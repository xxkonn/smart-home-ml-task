#!/usr/bin/env python3
"""
Example usage of the AUC Feature Diagnostic Tool

This script demonstrates how to use the AUC feature diagnostic tool
to analyze feature quality and importance for anomaly detection.
"""

import sys
import os
from pathlib import Path

# Add the current directory to the path (since we're now in src/)
sys.path.append(os.path.dirname(__file__))

from auc_feature_diagnostic import AUCFeatureDiagnostic


def example_single_file_analysis(data_prefix=""):
    """
    Example: Analyze a single test file
    """
    print("=== Example 1: Single File Analysis ===")
    
    # Create diagnostic tool
    diagnostic = AUCFeatureDiagnostic(
        output_dir="results_single_file",
        random_state=42
    )
    
    # Run analysis on a single file
    data_path = f"{data_prefix}data/test/house_091_smart_home.csv"
    diagnostic.run_full_diagnostic(
        data_path=data_path,
        sample_size=2000  # Use 2000 samples for faster analysis
    )
    
    print(f"Results saved to: {diagnostic.output_dir}")


def example_multiple_files_analysis(data_prefix=""):
    """
    Example: Analyze multiple test files
    """
    print("\n=== Example 2: Multiple Files Analysis ===")
    
    # Create diagnostic tool
    diagnostic = AUCFeatureDiagnostic(
        output_dir="results_multiple_files",
        random_state=42
    )
    
    # Analyze multiple files
    test_files = [
        f"{data_prefix}data/test/house_091_smart_home.csv",
        f"{data_prefix}data/test/house_092_smart_home.csv",
        f"{data_prefix}data/test/house_093_smart_home.csv"
    ]
    
    diagnostic.run_full_diagnostic(
        data_path=test_files,
        sample_size=5000  # Sample across all files
    )
    
    print(f"Results saved to: {diagnostic.output_dir}")


def example_directory_analysis(data_prefix=""):
    """
    Example: Analyze entire directory
    """
    print("\n=== Example 3: Directory Analysis ===")
    
    # Create diagnostic tool
    diagnostic = AUCFeatureDiagnostic(
        output_dir="results_directory",
        random_state=42
    )
    
    # Analyze entire test directory
    diagnostic.run_full_diagnostic(
        data_path=f"{data_prefix}data/test",
        sample_size=10000  # Larger sample from all files
    )
    
    print(f"Results saved to: {diagnostic.output_dir}")


def example_programmatic_access(data_prefix=""):
    """
    Example: Access results programmatically
    """
    print("\n=== Example 4: Programmatic Access ===")
    
    # Create diagnostic tool
    diagnostic = AUCFeatureDiagnostic(
        output_dir="results_programmatic",
        random_state=42
    )
    
    # Run analysis
    diagnostic.run_full_diagnostic(
        data_path=f"{data_prefix}data/test/house_091_smart_home.csv",
        sample_size=1000
    )
    
    # Access results programmatically
    feature_aucs = diagnostic.results['feature_aucs']
    feature_ranking = diagnostic.results['feature_ranking']
    
    # Find best features
    best_features = feature_ranking.head(10)
    print("\nTop 10 features:")
    for i, row in best_features.iterrows():
        print(f"{i+1:2d}. {row['feature']:<30} AUC: {row['auc']:.3f}")
    
    # Find features with high AUC
    high_auc_features = [
        feature for feature, stats in feature_aucs.items() 
        if stats['auc'] > 0.7
    ]
    print(f"\nFeatures with AUC > 0.7: {len(high_auc_features)}")
    for feature in high_auc_features:
        print(f"  {feature}: {feature_aucs[feature]['auc']:.3f}")
    
    # Find potentially problematic features
    poor_features = [
        feature for feature, stats in feature_aucs.items()
        if abs(stats['auc'] - 0.5) < 0.02 or stats['missing_rate'] > 0.1
    ]
    print(f"\nPotentially problematic features: {len(poor_features)}")


def main():
    """
    Run all examples
    """
    print("AUC Feature Diagnostic Tool - Usage Examples")
    print("=" * 60)
    
    # Make sure we're in the right directory
    if not os.path.exists("../data/test") and not os.path.exists("data/test"):
        print("Error: Please run this script from the src/ directory or project root")
        print("Current directory:", os.getcwd())
        return
    
    # Adjust data paths based on current directory
    if os.path.exists("data/test"):
        # Running from project root
        data_prefix = ""
    else:
        # Running from src/ directory  
        data_prefix = "../"
    
    try:
        # Run examples
        example_single_file_analysis(data_prefix)
        example_multiple_files_analysis(data_prefix)
        example_directory_analysis(data_prefix)
        example_programmatic_access(data_prefix)
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("\nGenerated outputs:")
        print("  - results_single_file/")
        print("  - results_multiple_files/")
        print("  - results_directory/")
        print("  - results_programmatic/")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have all required dependencies installed:")
        print("  pip install pandas numpy matplotlib seaborn scikit-learn")


if __name__ == "__main__":
    main()