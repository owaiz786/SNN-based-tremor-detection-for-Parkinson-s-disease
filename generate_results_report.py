# generate_results_report.py
import torch
import numpy as np
from pathlib import Path
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import json

def generate_complete_results():
    print("="*60)
    print("📊 GENERATING COMPLETE RESULTS REPORT")
    print("="*60)
    
    # Load training results
    results_file = Path("training_results_improved.txt")
    
    if results_file.exists():
        with open(results_file, "r") as f:
            results_text = f.read()
        print("\n✅ Training results loaded")
    else:
        print("\n⚠️  No training results file found. Run training first.")
        return
    
    # Load best model
    model_file = Path("best_tim_tremor_improved.pth")
    if model_file.exists():
        model_state = torch.load(model_file)
        print(f"✅ Best model loaded: {model_file}")
    else:
        print("⚠️  No saved model found")
        return
    
    # Create comprehensive report
    report = {
        "project_title": "Parkinson's Tremor Detection Using Spiking Neural Networks",
        "data_type": "Synthetic IMU Data",
        "model_architecture": "Bilateral SNN with Asymmetry Features",
        "num_classes": 4,
        "classes": ["UPDRS 0 (No Tremor)", "UPDRS 1 (Slight)", "UPDRS 2 (Mild)", "UPDRS 3 (Moderate)"],
        "results_file": str(results_file),
        "model_file": str(model_file)
    }
    
    # Parse results from text file
    for line in results_text.split('\n'):
        if 'Best Balanced Accuracy' in line:
            report['best_balanced_accuracy'] = line.split(':')[1].strip()
        if 'Per-class accuracies' in line:
            report['per_class_accuracies'] = line.split(':')[1].strip()
    
    # Save as JSON
    with open("results_summary.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n📄 Results summary saved to 'results_summary.json'")
    
    # Print summary
    print("\n" + "="*60)
    print("📈 RESULTS SUMMARY")
    print("="*60)
    print(f"Project: {report['project_title']}")
    print(f"Data: {report['data_type']}")
    print(f"Model: {report['model_architecture']}")
    print(f"Classes: {report['num_classes']} (UPDRS 0-3)")
    print(f"\n{report.get('best_balanced_accuracy', 'N/A')}")
    print(f"{report.get('per_class_accuracies', 'N/A')}")
    print("="*60)
    
    return report

if __name__ == "__main__":
    generate_complete_results()