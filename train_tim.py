import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import torch.optim as optim
import snntorch.functional as SF
from imu_dataset import IMUTremorDataset
from model_tim import TIMTremorSNN
from sklearn.model_selection import StratifiedKFold
import numpy as np
from collections import Counter
import os
from datetime import datetime

# Try to import mixed precision training
try:
    from torch.cuda.amp import GradScaler, autocast
    MIXED_PRECISION_AVAILABLE = True
except ImportError:
    MIXED_PRECISION_AVAILABLE = False
    print("⚠️  Mixed precision training not available")

def compute_class_weights(labels):
    """Compute balanced class weights with safety checks"""
    class_counts = np.bincount(labels, minlength=4)
    total = len(labels)
    
    # Safety check: if any class has 0 samples
    if np.any(class_counts == 0):
        print("⚠️  WARNING: Some classes have 0 samples! Using uniform weights.")
        return torch.ones(4, dtype=torch.float32) * 0.25
    
    class_weights = np.sqrt(total / (len(class_counts) * class_counts))
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = np.nan_to_num(class_weights, nan=1.0, posinf=1.0, neginf=1.0)
    
    return torch.tensor(class_weights, dtype=torch.float32)

def create_balanced_sampler(labels):
    """Create a weighted sampler with safety checks"""
    class_counts = np.bincount(labels, minlength=4)
    
    if np.any(class_counts == 0):
        print("⚠️  WARNING: Some classes missing! Using uniform sampler.")
        return None
    
    class_weights = 1. / (class_counts ** 0.5 + 1e-8)
    class_weights = class_weights / class_weights.sum()
    sample_weights = class_weights[labels]
    sample_weights = np.maximum(sample_weights, 1e-8)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights) * 2,
        replacement=True
    )
    return sampler

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs=10, base_lr=0.001):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_epoch = 0
    
    def step(self, epoch=None):
        if epoch is not None:
            self.current_epoch = epoch
        if self.current_epoch < self.warmup_epochs:
            lr_scale = (self.current_epoch + 1) / self.warmup_epochs
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.initial_lrs[i] * lr_scale

def save_results_file(filename, best_balanced_acc, best_epoch, best_per_class, confusion_matrix, training_completed=True):
    """Save results to file with error handling"""
    try:
        with open(filename, "w", encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("PARKINSON'S TREMOR DETECTION - TRAINING RESULTS\n")
            f.write("="*60 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Training Completed: {training_completed}\n\n")
            
            f.write(f"Best Balanced Accuracy: {best_balanced_acc:.2f}% at epoch {best_epoch}\n")
            f.write(f"Per-class accuracies:\n")
            for i in range(4):
                f.write(f"  Class {i}: {best_per_class[i]:.1f}%\n")
            
            f.write(f"\nConfusion Matrix:\n{confusion_matrix}\n")
            
            if training_completed:
                f.write("\n✅ Training completed successfully!\n")
            else:
                f.write("\n⚠️  Training interrupted or failed\n")
        
        print(f"\n💾 Results saved to '{filename}'")
        return True
    except Exception as e:
        print(f"\n❌ ERROR saving results file: {e}")
        return False

def train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=None, epoch=0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (left_spk, right_spk, ctx_spk, targets) in enumerate(train_loader):
        try:
            left_spk = left_spk.to(device)
            right_spk = right_spk.to(device)
            ctx_spk = ctx_spk.to(device)
            targets = targets.to(device)
            
            left_spk = left_spk.transpose(0, 1)
            right_spk = right_spk.transpose(0, 1)
            ctx_spk = ctx_spk.transpose(0, 1)
            
            if scaler is not None and MIXED_PRECISION_AVAILABLE:
                with autocast():
                    spk_out = model(left_spk, right_spk, ctx_spk)
                    spike_sum = spk_out.sum(dim=0)
                    loss = criterion(spike_sum, targets)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                spk_out = model(left_spk, right_spk, ctx_spk)
                spike_sum = spk_out.sum(dim=0)
                loss = criterion(spike_sum, targets)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            _, predicted = spike_sum.max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            total_loss += loss.item()
            
            if batch_idx % 50 == 0 and batch_idx > 0:
                print(f"    Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        except Exception as e:
            print(f"\n❌ Error in batch {batch_idx}: {e}")
            continue
    
    return total_loss / max(len(train_loader), 1), 100 * correct / max(total, 1)

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    val_correct = [0] * 4
    val_total = [0] * 4
    
    with torch.no_grad():
        for left_spk, right_spk, ctx_spk, targets in val_loader:
            try:
                left_spk = left_spk.to(device)
                right_spk = right_spk.to(device)
                ctx_spk = ctx_spk.to(device)
                targets = targets.to(device)
                
                left_spk = left_spk.transpose(0, 1)
                right_spk = right_spk.transpose(0, 1)
                ctx_spk = ctx_spk.transpose(0, 1)
                
                spk_out = model(left_spk, right_spk, ctx_spk)
                spike_sum = spk_out.sum(dim=0)
                
                loss = criterion(spike_sum, targets)
                val_loss += loss.item()
                
                _, predicted = spike_sum.max(1)
                
                for i in range(len(targets)):
                    label = targets[i].item()
                    pred = predicted[i].item()
                    if label == pred:
                        val_correct[label] += 1
                    val_total[label] += 1
            except Exception as e:
                print(f"⚠️  Validation error: {e}")
                continue
    
    val_acc_per_class = []
    for i in range(4):
        if val_total[i] > 0:
            acc = 100 * val_correct[i] / val_total[i]
            val_acc_per_class.append(acc)
        else:
            val_acc_per_class.append(0)
    
    val_acc = 100 * sum(val_correct) / max(sum(val_total), 1)
    avg_val_loss = val_loss / max(len(val_loader), 1)
    balanced_acc = np.mean(val_acc_per_class)
    
    return avg_val_loss, val_acc, balanced_acc, val_acc_per_class



def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Configuration
    data_dir = r"D:\Parkinson_SNN\real_pd_data"
    window_size = 200
    step_size = 100
    batch_size = 32
    num_epochs = 200
    learning_rate = 0.001
    warmup_epochs = 10
    
    # Results file name
    RESULTS_FILE = "training_results_improved.txt"
    BEST_MODEL_FILE = "best_tim_tremor_improved.pth"
    
    print("\n" + "="*60)
    print("🚀 IMPROVED SNN TRAINING FOR PARKINSON'S TREMOR DETECTION")
    print("="*60)
    
    # Initialize tracking variables
    best_balanced_acc = 0
    best_epoch = 0
    best_per_class = [0, 0, 0, 0]
    confusion_matrix = np.zeros((4, 4), dtype=int)
    training_completed = False
    
    try:
        # Load dataset
        print("\n[1] Loading Dataset...")
        full_dataset = IMUTremorDataset(
            data_dir=data_dir,
            window_size=window_size,
            step_size=step_size
        )
        
        # Get all labels
        all_labels = [full_dataset[i][3].item() for i in range(len(full_dataset))]
        print(f"\n📊 Overall label distribution: {Counter(all_labels)}")
        
        # Check if all samples are one class
        if len(Counter(all_labels)) < 2:
            print("\n❌ CRITICAL: All samples belong to ONE class!")
            print("   Generating results file with error message...")
            save_results_file(RESULTS_FILE, 0, 0, [0, 0, 0, 0], confusion_matrix, training_completed=False)
            return
        
        # Stratified split
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, val_idx = next(skf.split(np.zeros(len(all_labels)), all_labels))
        
        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)
        train_labels = [all_labels[i] for i in train_idx]
        
        train_sampler = create_balanced_sampler(train_labels)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=train_sampler if train_sampler else None,
            shuffle=(train_sampler is None),
            drop_last=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        print(f"\n📊 Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        print(f"📊 Train label distribution: {Counter(train_labels)}")
        
        # Initialize model
        print("\n[2] Creating SNN model...")
        model = TIMTremorSNN(num_classes=4, use_asymmetry_features=True).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,}")
        
        # Class weights
        class_weights = compute_class_weights(train_labels).to(device)
        print(f"📊 Class weights: {class_weights}")
        
        criterion = FocalLoss(gamma=3, alpha=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=10, factor=0.5, min_lr=1e-6
        )
        
        warmup_scheduler = WarmupScheduler(optimizer, warmup_epochs=warmup_epochs, base_lr=learning_rate)
        scaler = GradScaler() if MIXED_PRECISION_AVAILABLE and device.type == 'cuda' else None
        
        print("\n[3] Starting training...")
        print("="*60)
        
        patience_counter = 0
        
        for epoch in range(num_epochs):
            warmup_scheduler.step(epoch)
            current_lr = optimizer.param_groups[0]['lr']
            
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler, epoch
            )
            
            val_loss, val_acc, balanced_acc, val_acc_per_class = validate(
                model, val_loader, criterion, device
            )
            
            scheduler.step(balanced_acc)
            
            for i in range(4):
                if val_acc_per_class[i] > best_per_class[i]:
                    best_per_class[i] = val_acc_per_class[i]
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"\nEpoch {epoch+1}/{num_epochs} (LR: {current_lr:.6f})")
                print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
                print(f"  Balanced Acc: {balanced_acc:.2f}%")
                print(f"  Per-class: 0:{val_acc_per_class[0]:.1f}% 1:{val_acc_per_class[1]:.1f}% "
                      f"2:{val_acc_per_class[2]:.1f}% 3:{val_acc_per_class[3]:.1f}%")
            
            if balanced_acc > best_balanced_acc:
                best_balanced_acc = balanced_acc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), BEST_MODEL_FILE)
                print(f"  🏆 NEW BEST! Balanced Accuracy: {balanced_acc:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= 30:
                print(f"\n⏹️ Early stopping at epoch {epoch+1}")
                break
        
        training_completed = True
        print("\n" + "="*60)
        print(f"🎯 FINAL BEST BALANCED ACCURACY: {best_balanced_acc:.2f}% at epoch {best_epoch}")
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Final evaluation (if model was saved)
    try:
        print("\n[4] Final evaluation...")
        if os.path.exists(BEST_MODEL_FILE):
            model.load_state_dict(torch.load(BEST_MODEL_FILE))
            model.eval()
            
            with torch.no_grad():
                for left_spk, right_spk, ctx_spk, targets in val_loader:
                    left_spk = left_spk.to(device).transpose(0, 1)
                    right_spk = right_spk.to(device).transpose(0, 1)
                    ctx_spk = ctx_spk.to(device).transpose(0, 1)
                    targets = targets.to(device)
                    
                    spk_out = model(left_spk, right_spk, ctx_spk)
                    spike_sum = spk_out.sum(dim=0)
                    _, predicted = spike_sum.max(1)
                    
                    for i in range(len(targets)):
                        true_label = targets[i].item()
                        pred_label = predicted[i].item()
                        confusion_matrix[true_label, pred_label] += 1
            
            print("\n📊 Confusion Matrix:")
            print("          Predicted")
            print("         0    1    2    3")
            for i in range(4):
                row = f"True {i}: "
                for j in range(4):
                    row += f"{confusion_matrix[i, j]:4d} "
                print(row)
            
            print("\n📈 Per-class Metrics:")
            for i in range(4):
                total = confusion_matrix[i].sum()
                if total > 0:
                    correct = confusion_matrix[i, i]
                    accuracy = 100 * correct / total
                    precision = 100 * confusion_matrix[i, i] / (confusion_matrix[:, i].sum() + 1e-8)
                    recall = 100 * confusion_matrix[i, i] / (confusion_matrix[i].sum() + 1e-8)
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)
                    
                    print(f"  Class {i}:")
                    print(f"    Accuracy:  {accuracy:.2f}% ({correct}/{total})")
                    print(f"    Precision: {precision:.2f}%")
                    print(f"    Recall:    {recall:.2f}%")
                    print(f"    F1-Score:  {f1:.2f}%")
            
            print("\n📊 Overall Metrics:")
            print(f"  Overall Accuracy: {100 * np.trace(confusion_matrix) / confusion_matrix.sum():.2f}%")
            print(f"  Balanced Accuracy: {best_balanced_acc:.2f}%")
        else:
            print(f"⚠️  Model file '{BEST_MODEL_FILE}' not found. Skipping evaluation.")
    
    except Exception as e:
        print(f"\n⚠️  Evaluation error: {e}")
    
    # ALWAYS save results file (even if training failed)
    save_results_file(
        RESULTS_FILE, 
        best_balanced_acc, 
        best_epoch, 
        best_per_class, 
        confusion_matrix, 
        training_completed
    )
    
    print("="*60)
    
    

if __name__ == "__main__":
    main()