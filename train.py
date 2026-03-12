import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from snntorch import functional as SF
from tim_tremor_dataset import TIMTremorDataset
from model import MultiPathwayBilateralSNN
from sklearn.model_selection import StratifiedKFold
import numpy as np

def main():
    print("Initializing TIM-Tremor Dataset...")
    
    # Load TIM-Tremor dataset
    full_dataset = TIMTremorDataset("Parkinson-s-Disease-Tremor-Dataset", window_size=128)
    
    # Get all labels for stratified split
    all_labels = [full_dataset[i][3].item() for i in range(len(full_dataset))]
    
    # Use stratified k-fold for better evaluation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Use first fold for train/test split
    train_idx, test_idx = next(skf.split(np.zeros(len(all_labels)), all_labels))
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    test_dataset = torch.utils.data.Subset(full_dataset, test_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"📊 Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    print("Initializing Bilateral Context-Aware SNN...")
    model = MultiPathwayBilateralSNN(num_classes=4)
    
    # Calculate class weights for imbalanced data
    class_counts = np.bincount(all_labels)
    class_weights = 1.0 / (class_counts + 1e-8)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    loss_fn = SF.ce_rate_loss()
    
    num_epochs = 30
    print("\n🚀 Starting Training on TIM-Tremor Dataset...")
    
    best_accuracy = 0
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (left_spk, right_spk, ctx_spk, targets) in enumerate(train_loader):
            left_spk = left_spk.transpose(0, 1)
            right_spk = right_spk.transpose(0, 1)
            ctx_spk = ctx_spk.transpose(0, 1)
            
            spk_out = model(left_spk, right_spk, ctx_spk)
            loss = loss_fn(spk_out, targets)
            
            _, predicted = spk_out.sum(dim=0).max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        
        with torch.no_grad():
            for left_spk, right_spk, ctx_spk, targets in test_loader:
                left_spk = left_spk.transpose(0, 1)
                right_spk = right_spk.transpose(0, 1)
                ctx_spk = ctx_spk.transpose(0, 1)
                
                spk_out = model(left_spk, right_spk, ctx_spk)
                loss = loss_fn(spk_out, targets)
                val_loss += loss.item()
                
                _, predicted = spk_out.sum(dim=0).max(1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(test_loader)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), "best_tim_tremor_model.pth")
            print(f"   🏆 New best model! Accuracy: {val_acc:.2f}%")
        
        # Early stopping check
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break
    
    print(f"\n🎯 FINAL BEST VALIDATION ACCURACY: {best_accuracy:.2f}%")
    print("💾 Best model saved to 'best_tim_tremor_model.pth'")
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load("best_tim_tremor_model.pth"))
    model.eval()
    
    # Detailed per-class accuracy
    class_correct = [0] * 4
    class_total = [0] * 4
    
    with torch.no_grad():
        for left_spk, right_spk, ctx_spk, targets in test_loader:
            left_spk = left_spk.transpose(0, 1)
            right_spk = right_spk.transpose(0, 1)
            ctx_spk = ctx_spk.transpose(0, 1)
            
            spk_out = model(left_spk, right_spk, ctx_spk)
            _, predicted = spk_out.sum(dim=0).max(1)
            
            for i in range(len(targets)):
                label = targets[i].item()
                pred = predicted[i].item()
                if label == pred:
                    class_correct[label] += 1
                class_total[label] += 1
    
    print("\n📊 Per-class Accuracy:")
    for i in range(4):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f"   UPDRS {i}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")

if __name__ == "__main__":
    main()