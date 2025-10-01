#!/usr/bin/env python3
"""
ROC Curve Comparison: 3-Point vs Multi-Point
============================================

This demonstrates the difference between 3-point and proper multi-point ROC curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd

# Simulate what your model results look like
def simulate_model_performance():
    """Simulate model predictions to show the difference."""
    
    # Create synthetic test data
    np.random.seed(42)
    n_samples = 1000
    n_anomalies = 100
    
    # Create labels (0 = normal, 1 = anomaly)
    y_true = np.zeros(n_samples)
    y_true[:n_anomalies] = 1
    np.random.shuffle(y_true)
    
    # Simulate different model prediction scores
    # Better models have higher scores for anomalies, lower for normal
    normal_scores = np.random.normal(-1.5, 1.0, n_samples - n_anomalies)
    anomaly_scores = np.random.normal(1.5, 1.0, n_anomalies)
    
    y_scores = np.zeros(n_samples)
    y_scores[y_true == 0] = normal_scores
    y_scores[y_true == 1] = anomaly_scores
    
    return y_true, y_scores

def calculate_3_point_roc(y_true, y_pred_binary):
    """Calculate 3-point ROC from binary predictions."""
    tp = np.sum((y_true == 1) & (y_pred_binary == 1))
    tn = np.sum((y_true == 0) & (y_pred_binary == 0))
    fp = np.sum((y_true == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true == 1) & (y_pred_binary == 0))
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # 3-point ROC
    fpr_3point = np.array([0.0, fpr, 1.0])
    tpr_3point = np.array([0.0, tpr, 1.0])
    auc_3point = auc(fpr_3point, tpr_3point)
    
    return fpr_3point, tpr_3point, auc_3point

def create_comparison_plot():
    """Create a comparison plot showing both approaches."""
    
    # Get synthetic data
    y_true, y_scores = simulate_model_performance()
    
    # Method 1: Multi-point ROC (proper way)
    fpr_multi, tpr_multi, thresholds = roc_curve(y_true, y_scores)
    auc_multi = auc(fpr_multi, tpr_multi)
    
    # Method 2: 3-point ROC (my approximation)
    # Convert scores to binary predictions using a threshold
    threshold = 0.0
    y_pred_binary = (y_scores > threshold).astype(int)
    fpr_3point, tpr_3point, auc_3point = calculate_3_point_roc(y_true, y_pred_binary)
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Multi-point ROC (what you SHOULD have)
    ax1.plot(fpr_multi, tpr_multi, 'b-', linewidth=3, 
             label=f'Multi-Point ROC (AUC = {auc_multi:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    ax1.scatter(fpr_multi[::len(fpr_multi)//10], tpr_multi[::len(tpr_multi)//10], 
               c='blue', s=30, alpha=0.7, zorder=5)
    ax1.set_title(f'Proper ROC Curve\n{len(fpr_multi)} threshold points', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('False Positive Rate', fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(0.6, 0.2, f'Smooth curve\nMany thresholds\nAccurate AUC', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    # Plot 2: 3-point ROC (what I generated)
    ax2.plot(fpr_3point, tpr_3point, 'r-', linewidth=3, marker='o', markersize=8,
             label=f'3-Point ROC (AUC = {auc_3point:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    ax2.set_title('My 3-Point Approximation\n3 threshold points only', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('False Positive Rate', fontweight='bold')
    ax2.set_ylabel('True Positive Rate', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.text(0.6, 0.2, f'Angular lines\nOne threshold\nApproximate AUC', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/mnt/d/smart-home-ml-task/models/results/plots/roc_comparison_3point_vs_multipoint.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Create explanation
    explanation = f"""
# ROC Curve Comparison: 3-Point vs Multi-Point

## Multi-Point ROC (Proper Method)
- **Points Generated**: {len(fpr_multi)}
- **AUC Score**: {auc_multi:.4f}
- **Curve Type**: Smooth, showing all threshold trade-offs
- **Information**: Complete sensitivity/specificity relationship

## 3-Point ROC (My Approximation)  
- **Points Generated**: 3
- **AUC Score**: {auc_3point:.4f}
- **Curve Type**: Angular, connecting key points
- **Information**: Single operating point performance

## Why I Used 3-Point Approach

1. **Your data contained only final confusion matrices**
2. **No access to raw prediction scores**
3. **Quick comparative visualization needed**
4. **Still mathematically valid for model comparison**

## For Production Use

You should generate proper multi-point ROC curves by:
1. Loading trained models
2. Getting prediction scores on test data  
3. Sweeping multiple thresholds
4. Plotting the full sensitivity/specificity trade-off

The 3-point version was a practical compromise given the available data!
"""
    
    with open('/mnt/d/smart-home-ml-task/models/results/plots/roc_explanation.md', 'w') as f:
        f.write(explanation)
    
    print("✅ ROC comparison visualization created!")
    print(f"📊 Multi-point curve: {len(fpr_multi)} points, AUC = {auc_multi:.4f}")
    print(f"📊 3-point curve: 3 points, AUC = {auc_3point:.4f}")
    print("📁 Files saved in models/results/plots/")

if __name__ == "__main__":
    create_comparison_plot()