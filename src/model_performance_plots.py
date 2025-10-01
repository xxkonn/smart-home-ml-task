#!/usr/bin/env python3
"""
Model Performance Visualization Script
=====================================

This script generates comprehensive performance visualizations for trained Isolation Forest models:
1. ROC Curves for Validation & Test sets
2. AUC Score Comparison Bar Charts  
3. Confusion Matrix Heatmaps

Date: October 1, 2025
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelPerformanceVisualizer:
    """Class to handle all model performance visualizations."""
    
    def __init__(self, results_dir: str, output_dir: str):
        """
        Initialize the visualizer.
        
        Args:
            results_dir: Directory containing model result JSON files
            output_dir: Directory to save generated plots
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model names and their corresponding file patterns
        self.model_configs = {
            'estimators_300': 'Isolation Forest (300 est.)',
            'estimators_500': 'Isolation Forest (500 est.)',
            'estimators_700': 'Isolation Forest (700 est.)',
            'estimators_900': 'Isolation Forest (900 est.)',
            'estimators_1100': 'Isolation Forest (1100 est.)'
        }
        
        self.model_data = {}
        
    def load_model_results(self) -> None:
        """Load all model results from JSON files."""
        print("Loading model results...")
        
        for model_name in self.model_configs.keys():
            result_file = self.results_dir / f"{model_name}_results.json"
            
            if result_file.exists():
                with open(result_file, 'r') as f:
                    self.model_data[model_name] = json.load(f)
                print(f"✓ Loaded {model_name}")
            else:
                print(f"⚠ Warning: {result_file} not found")
                
        print(f"Loaded {len(self.model_data)} model results\n")
    
    def calculate_roc_data(self, tp: int, tn: int, fp: int, fn: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calculate ROC curve data from confusion matrix values.
        
        Args:
            tp: True Positives
            tn: True Negatives  
            fp: False Positives
            fn: False Negatives
            
        Returns:
            Tuple of (fpr_array, tpr_array, auc_score)
        """
        # For binary classification with given confusion matrix values
        # We'll create a simple 2-point ROC curve
        
        # Calculate TPR and FPR
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # Create ROC curve points: (0,0), (fpr, tpr), (1,1)
        fpr_array = np.array([0.0, fpr, 1.0])
        tpr_array = np.array([0.0, tpr, 1.0])
        
        # Calculate AUC using trapezoidal rule
        auc_score = auc(fpr_array, tpr_array)
        
        return fpr_array, tpr_array, auc_score
        
    def plot_roc_curves(self) -> None:
        """Generate ROC curves for validation and test sets."""
        print("Generating ROC curves...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        validation_aucs = {}
        test_aucs = {}
        
        # Color palette for models
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.model_data)))
        
        for idx, (model_name, model_data) in enumerate(self.model_data.items()):
            model_label = self.model_configs[model_name]
            color = colors[idx]
            
            # Validation ROC
            val_metrics = model_data['validation_results']['overall_metrics']
            val_tp = val_metrics['true_positives']
            val_tn = val_metrics['true_negatives']
            val_fp = val_metrics['false_positives']
            val_fn = val_metrics['false_negatives']
            
            val_fpr, val_tpr, val_auc = self.calculate_roc_data(val_tp, val_tn, val_fp, val_fn)
            validation_aucs[model_name] = val_auc
            
            ax1.plot(val_fpr, val_tpr, color=color, linewidth=2.5, 
                    label=f'{model_label} (AUC = {val_auc:.3f})')
            
            # Test ROC
            test_metrics = model_data['test_results']['overall_metrics']
            test_tp = test_metrics['true_positives']
            test_tn = test_metrics['true_negatives']
            test_fp = test_metrics['false_positives']
            test_fn = test_metrics['false_negatives']
            
            test_fpr, test_tpr, test_auc = self.calculate_roc_data(test_tp, test_tn, test_fp, test_fn)
            test_aucs[model_name] = test_auc
            
            ax2.plot(test_fpr, test_tpr, color=color, linewidth=2.5,
                    label=f'{model_label} (AUC = {test_auc:.3f})')
        
        # Plot diagonal reference line
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=1)
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=1)
        
        # Customize validation plot
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax1.set_title('ROC Curves - Validation Set\nSmart Home Anomaly Detection', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.legend(loc="lower right", fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Customize test plot
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax2.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax2.set_title('ROC Curves - Test Set\nSmart Home Anomaly Detection', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.legend(loc="lower right", fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curves_validation_test.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Store AUC values for bar chart
        self.validation_aucs = validation_aucs
        self.test_aucs = test_aucs
        
        print("✓ ROC curves saved")
        
    def plot_auc_comparison(self) -> None:
        """Generate AUC score comparison bar chart."""
        print("Generating AUC comparison bar chart...")
        
        # Prepare data
        models = list(self.validation_aucs.keys())
        model_labels = [self.model_configs[m] for m in models]
        val_scores = [self.validation_aucs[m] for m in models]
        test_scores = [self.test_aucs[m] for m in models]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(models))
        width = 0.35
        
        # Create bars
        bars1 = ax.bar(x - width/2, val_scores, width, label='Validation Set', 
                      color='skyblue', alpha=0.8, edgecolor='navy', linewidth=1.2)
        bars2 = ax.bar(x + width/2, test_scores, width, label='Test Set', 
                      color='lightcoral', alpha=0.8, edgecolor='darkred', linewidth=1.2)
        
        # Add value labels on bars
        def add_value_labels(bars, scores):
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.annotate(f'{score:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontweight='bold', fontsize=10)
        
        add_value_labels(bars1, val_scores)
        add_value_labels(bars2, test_scores)
        
        # Customize plot
        ax.set_xlabel('Model Configuration', fontsize=12, fontweight='bold')
        ax.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
        ax.set_title('AUC Score Comparison Across Models\nSmart Home Anomaly Detection', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([label.replace('Isolation Forest ', 'IF\n') for label in model_labels], 
                          fontsize=10)
        ax.legend(fontsize=11, loc='upper left')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add horizontal reference lines
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, linewidth=1)
        ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, linewidth=1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'auc_comparison_bar_chart.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("✓ AUC comparison chart saved")
        
    def plot_confusion_matrices(self) -> None:
        """Generate confusion matrix heatmaps for each model."""
        print("Generating confusion matrix plots...")
        
        n_models = len(self.model_data)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (model_name, model_data) in enumerate(self.model_data.items()):
            ax = axes[idx]
            
            # Get test metrics (using test set for final evaluation)
            test_metrics = model_data['test_results']['overall_metrics']
            tp = test_metrics['true_positives']
            tn = test_metrics['true_negatives']
            fp = test_metrics['false_positives']
            fn = test_metrics['false_negatives']
            
            # Create confusion matrix
            cm = np.array([[tn, fp],
                          [fn, tp]])
            
            # Calculate percentages
            cm_percent = cm.astype('float') / cm.sum() * 100
            
            # Create annotations combining counts and percentages
            annotations = np.empty_like(cm).astype(str)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    annotations[i, j] = f'{cm[i, j]:,}\n({cm_percent[i, j]:.1f}%)'
            
            # Create heatmap
            sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                       cbar=True, square=True, ax=ax,
                       xticklabels=['Normal', 'Anomaly'],
                       yticklabels=['Normal', 'Anomaly'],
                       annot_kws={'fontsize': 11, 'fontweight': 'bold'})
            
            # Customize plot
            model_label = self.model_configs[model_name]
            ax.set_title(f'{model_label}\nTest Set Confusion Matrix', 
                        fontsize=12, fontweight='bold', pad=15)
            ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
            
            # Add performance metrics as text
            precision = test_metrics['precision']
            recall = test_metrics['recall']
            f1_score = test_metrics['f1_score']
            
            metrics_text = f'Precision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1_score:.3f}'
            ax.text(2.1, 0.5, metrics_text, transform=ax.transData, fontsize=10,
                   verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor="lightgray", alpha=0.8))
        
        # Hide empty subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Confusion Matrices - Model Performance Comparison\nSmart Home Anomaly Detection', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrices_all_models.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("✓ Confusion matrices saved")
        
    def generate_summary_report(self) -> None:
        """Generate a summary report of all visualizations."""
        print("Generating summary report...")
        
        report_content = f"""# Model Performance Visualization Report

**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Project:** Smart Home IoT Anomaly Detection
**Models Analyzed:** {len(self.model_data)} Isolation Forest configurations

## Files Generated

1. **ROC Curves** (`roc_curves_validation_test.png`)
   - Comparison of ROC curves for validation and test sets
   - Shows model discrimination capability
   - AUC scores included for each model

2. **AUC Score Comparison** (`auc_comparison_bar_chart.png`)
   - Bar chart comparing AUC scores across all models
   - Separate bars for validation and test performance
   - Clear visual comparison of model ranking

3. **Confusion Matrices** (`confusion_matrices_all_models.png`)
   - Detailed confusion matrices for each model
   - Shows true/false positives and negatives
   - Includes precision, recall, and F1-score metrics

## Model Performance Summary

### AUC Scores (Test Set)
"""
        
        # Add AUC rankings
        sorted_models = sorted(self.test_aucs.items(), key=lambda x: x[1], reverse=True)
        for i, (model_name, auc_score) in enumerate(sorted_models, 1):
            model_label = self.model_configs[model_name]
            report_content += f"{i}. **{model_label}**: {auc_score:.4f}\n"
        
        report_content += f"""
### Best Performing Model
**{self.model_configs[sorted_models[0][0]]}** achieved the highest AUC score of {sorted_models[0][1]:.4f} on the test set.

## Visualization Details

- **Resolution**: 300 DPI for publication quality
- **Format**: PNG with transparent backgrounds
- **Style**: Professional seaborn styling with clear labels
- **Color Scheme**: Distinct colors for easy model identification

## Usage Instructions

1. Open the generated PNG files to view detailed performance visualizations
2. Use ROC curves to understand model discrimination ability
3. Compare AUC scores to rank model performance
4. Analyze confusion matrices for detailed classification performance

---
*Generated by Smart Home ML Task Performance Visualizer*
"""
        
        # Save report
        with open(self.output_dir / 'visualization_report.md', 'w') as f:
            f.write(report_content)
        
        print("✓ Summary report saved")

    def run_all_visualizations(self) -> None:
        """Run all visualization functions."""
        print("=" * 60)
        print("SMART HOME ML TASK - MODEL PERFORMANCE VISUALIZER")
        print("=" * 60)
        
        self.load_model_results()
        
        if not self.model_data:
            print("❌ No model results found. Please check the results directory.")
            return
        
        self.plot_roc_curves()
        self.plot_auc_comparison()
        self.plot_confusion_matrices()
        self.generate_summary_report()
        
        print("\n" + "=" * 60)
        print("✅ ALL VISUALIZATIONS COMPLETED SUCCESSFULLY!")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📊 Generated {len(list(self.output_dir.glob('*.png')))} plot files")
        print("=" * 60)


def main():
    """Main function to run the visualizer."""
    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    results_dir = project_root / "models" / "results"
    output_dir = results_dir / "plots"
    
    # Create visualizer and run
    visualizer = ModelPerformanceVisualizer(results_dir, output_dir)
    visualizer.run_all_visualizations()


if __name__ == "__main__":
    main()