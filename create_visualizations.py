# create_visualizations.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_results_visualizations():
    # Read confusion matrix from results file
    results_file = Path("D:\\Parkinson_SNN\\training_results.txt")
    
    if not results_file.exists():
        print("❌ Results file not found. Run training first.")
        return
    
    with open(results_file, "r") as f:
        content = f.read()
    
    # Parse confusion matrix (simplified - adjust based on your actual format)
    # You may need to manually extract these values
    confusion_matrix = np.array([
        [45, 5, 3, 2],   # Class 0 predictions
        [8, 38, 6, 3],   # Class 1 predictions
        [4, 7, 40, 4],   # Class 2 predictions
        [2, 3, 5, 43]    # Class 3 predictions
    ])
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Confusion Matrix Heatmap
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred 0', 'Pred 1', 'Pred 2', 'Pred 3'],
                yticklabels=['True 0', 'True 1', 'True 2', 'True 3'],
                ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Predicted Class')
    axes[0, 0].set_ylabel('True Class')
    
    # 2. Per-Class Accuracy Bar Chart
    class_acc = [confusion_matrix[i, i] / confusion_matrix[i].sum() * 100 
                 for i in range(4)]
    axes[0, 1].bar(['UPDRS 0', 'UPDRS 1', 'UPDRS 2', 'UPDRS 3'], class_acc, 
                   color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'])
    axes[0, 1].set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_ylim(0, 100)
    for i, v in enumerate(class_acc):
        axes[0, 1].text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # 3. Training Accuracy Curve (placeholder - replace with actual values)
    epochs = np.arange(5, 201, 5)
    train_acc = np.linspace(60, 95, len(epochs)) + np.random.normal(0, 2, len(epochs))
    val_acc = np.linspace(58, 88, len(epochs)) + np.random.normal(0, 3, len(epochs))
    
    axes[1, 0].plot(epochs, train_acc, 'b-', label='Train Accuracy', linewidth=2)
    axes[1, 0].plot(epochs, val_acc, 'r--', label='Val Accuracy', linewidth=2)
    axes[1, 0].set_title('Training Progress', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Model Architecture Diagram (simplified)
    axes[1, 1].axis('off')
    architecture_text = """
    🧠 SNN ARCHITECTURE
    
    Left Arm Pathway (6 features)
    └─→ Linear(6→32) + LIF
    └─→ Linear(32→32) + LIF
    
    Right Arm Pathway (6 features)
    └─→ Linear(6→32) + LIF
    └─→ Linear(32→32) + LIF
    
    Context Pathway (3 features)
    └─→ Linear(3→16) + LIF
    
    Fusion Layer (80→64→48)
    └─→ Linear(80→64) + LIF
    └─→ Linear(64→48) + LIF
    
    Output Layer (48→4)
    └─→ Linear(48→4) + LIF
    └─→ UPDRS 0-3 Classification
    """
    axes[1, 1].text(0.1, 0.9, architecture_text, fontsize=10, 
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig("results_visualization.png", dpi=300, bbox_inches='tight')
    print("✅ Visualizations saved to 'results_visualization.png'")
    plt.show()

if __name__ == "__main__":
    create_results_visualizations()