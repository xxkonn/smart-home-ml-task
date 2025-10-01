#!/usr/bin/env python3
"""
AUC Feature Diagnostic Tool for Smart Home IoT Anomaly Detection

This script analyzes the quality and importance of engineered features using AUC metrics.
It provides comprehensive diagnostic information about individual features and their
ability to distinguish between normal and anomalous behavior.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Import our feature engineering module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from feature_engineering import engineer_all_features


class AUCFeatureDiagnostic:
    """
    Comprehensive feature diagnostic tool using AUC metrics.
    """
    
    def __init__(self, output_dir: str = "feature_diagnostic_results", 
                 random_state: int = 42):
        """
        Initialize the diagnostic tool.
        
        Args:
            output_dir: Directory to save diagnostic results
            random_state: Random state for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.random_state = random_state
        self.results = {}
        
        # Setup plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_and_prepare_data(self, data_path: Union[str, List[str]], 
                            sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load and prepare data for analysis.
        
        Args:
            data_path: Path to CSV file or list of paths
            sample_size: Optional sample size for large datasets
            
        Returns:
            Prepared DataFrame with engineered features
        """
        print("Loading data...")
        
        # Handle single file or multiple files
        if isinstance(data_path, str):
            if os.path.isfile(data_path):
                df = pd.read_csv(data_path)
            elif os.path.isdir(data_path):
                # Load all CSV files in directory
                csv_files = list(Path(data_path).glob("*.csv"))
                if not csv_files:
                    raise ValueError(f"No CSV files found in {data_path}")
                
                dfs = []
                for file in csv_files[:10]:  # Limit to first 10 files for memory
                    dfs.append(pd.read_csv(file))
                df = pd.concat(dfs, ignore_index=True)
            else:
                raise ValueError(f"Invalid data path: {data_path}")
        else:
            # List of file paths
            dfs = []
            for file in data_path:
                dfs.append(pd.read_csv(file))
            df = pd.concat(dfs, ignore_index=True)
        
        print(f"Loaded data shape: {df.shape}")
        
        # Sample if requested
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=self.random_state)
            print(f"Sampled to {sample_size} rows")
        
        # Check for required columns
        required_cols = ['label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Engineer features
        print("Engineering features...")
        df_engineered = engineer_all_features(df)
        
        print(f"Final data shape: {df_engineered.shape}")
        print(f"Anomaly rate: {df_engineered['label'].mean():.3f}")
        
        return df_engineered
    
    def calculate_feature_auc(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Calculate AUC scores for all features.
        
        Args:
            df: DataFrame with features and labels
            
        Returns:
            Dictionary with feature AUC information
        """
        print("Calculating individual feature AUC scores...")
        
        # Get feature columns (exclude metadata)
        exclude_cols = ['timestamp', 'label', 'anomaly_type', 'house_id']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        feature_aucs = {}
        y = df['label'].values
        
        for col in feature_cols:
            try:
                X = df[col].values
                
                # Handle missing values
                if np.isnan(X).any():
                    X = np.nan_to_num(X, nan=0.0)
                
                # Handle constant features
                if np.std(X) == 0:
                    auc_score = 0.5
                    auc_std = 0.0
                else:
                    # Calculate AUC
                    auc_score = roc_auc_score(y, X)
                    
                    # Calculate cross-validated AUC for stability
                    try:
                        lr = LogisticRegression(random_state=self.random_state)
                        cv_scores = cross_val_score(lr, X.reshape(-1, 1), y, 
                                                  cv=3, scoring='roc_auc')
                        auc_std = np.std(cv_scores)
                    except:
                        auc_std = 0.0
                
                # Feature statistics
                feature_stats = {
                    'auc': auc_score,
                    'auc_std': auc_std,
                    'mean': np.mean(X),
                    'std': np.std(X),
                    'min': np.min(X),
                    'max': np.max(X),
                    'missing_rate': np.isnan(df[col]).mean(),
                    'zero_rate': (X == 0).mean(),
                    'unique_values': len(np.unique(X))
                }
                
                feature_aucs[col] = feature_stats
                
            except Exception as e:
                print(f"Error calculating AUC for {col}: {e}")
                feature_aucs[col] = {
                    'auc': 0.5, 'auc_std': 0.0, 'mean': 0, 'std': 0,
                    'min': 0, 'max': 0, 'missing_rate': 1.0, 'zero_rate': 1.0,
                    'unique_values': 1
                }
        
        return feature_aucs
    
    def identify_feature_groups(self, feature_aucs: Dict[str, Dict]) -> Dict[str, List[str]]:
        """
        Group features by their origin/type.
        
        Args:
            feature_aucs: Feature AUC dictionary
            
        Returns:
            Dictionary of feature groups
        """
        groups = {
            'time_features': [],
            'temperature_features': [],
            'fridge_features': [],
            'motion_features': [],
            'door_window_features': [],
            'cross_sensor_features': [],
            'original_features': []
        }
        
        for feature in feature_aucs.keys():
            if any(x in feature.lower() for x in ['hour', 'day', 'month', 'weekend', 'night', 'business', 'peak']):
                groups['time_features'].append(feature)
            elif 'temp' in feature.lower():
                groups['temperature_features'].append(feature)
            elif 'fridge' in feature.lower():
                groups['fridge_features'].append(feature)
            elif 'motion' in feature.lower():
                groups['motion_features'].append(feature)
            elif any(x in feature.lower() for x in ['door', 'window']):
                groups['door_window_features'].append(feature)
            elif any(x in feature.lower() for x in ['interaction', 'with', 'and', 'ratio']):
                groups['cross_sensor_features'].append(feature)
            else:
                groups['original_features'].append(feature)
        
        return groups
    
    def generate_feature_ranking(self, feature_aucs: Dict[str, Dict], 
                                top_n: int = 20) -> pd.DataFrame:
        """
        Generate ranked list of top features.
        
        Args:
            feature_aucs: Feature AUC dictionary
            top_n: Number of top features to return
            
        Returns:
            DataFrame with ranked features
        """
        ranking_data = []
        
        for feature, stats in feature_aucs.items():
            ranking_data.append({
                'feature': feature,
                'auc': stats['auc'],
                'auc_std': stats['auc_std'],
                'stability': 1 - stats['auc_std'],  # Higher = more stable
                'discriminative_power': abs(stats['auc'] - 0.5) * 2,  # 0-1 scale
                'data_quality': 1 - stats['missing_rate'] - stats['zero_rate'],
                'uniqueness': min(stats['unique_values'] / 100, 1.0),  # Normalized
            })
        
        df_ranking = pd.DataFrame(ranking_data)
        
        # Calculate composite score
        df_ranking['composite_score'] = (
            0.4 * df_ranking['discriminative_power'] + 
            0.3 * df_ranking['stability'] +
            0.2 * df_ranking['data_quality'] +
            0.1 * df_ranking['uniqueness']
        )
        
        # Sort by composite score
        df_ranking = df_ranking.sort_values('composite_score', ascending=False)
        
        return df_ranking.head(top_n)
    
    def plot_feature_distributions(self, df: pd.DataFrame, top_features: List[str], 
                                 max_features: int = 12):
        """
        Plot distributions of top features split by label.
        """
        n_features = min(len(top_features), max_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, feature in enumerate(top_features[:n_features]):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Plot distributions
            normal_data = df[df['label'] == 0][feature].dropna()
            anomaly_data = df[df['label'] == 1][feature].dropna()
            
            ax.hist(normal_data, bins=30, alpha=0.7, label='Normal', density=True)
            ax.hist(anomaly_data, bins=30, alpha=0.7, label='Anomaly', density=True)
            
            ax.set_title(f'{feature}\n(AUC: {self.results["feature_aucs"][feature]["auc"]:.3f})')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(n_features, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_auc_by_group(self, feature_aucs: Dict[str, Dict], 
                         feature_groups: Dict[str, List[str]]):
        """
        Plot AUC scores grouped by feature type.
        """
        group_aucs = {}
        
        for group_name, features in feature_groups.items():
            if features:
                aucs = [feature_aucs[f]['auc'] for f in features]
                group_aucs[group_name] = aucs
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create box plot
        box_data = []
        labels = []
        for group_name, aucs in group_aucs.items():
            if aucs:
                box_data.append(aucs)
                labels.append(f'{group_name}\n(n={len(aucs)})')
        
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
        
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('AUC Score')
        ax.set_title('Feature AUC Distribution by Group')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random (0.5)')
        ax.legend()
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'auc_by_group.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_correlation_matrix(self, df: pd.DataFrame, top_features: List[str]):
        """
        Plot correlation matrix of top features.
        """
        # Select top features and compute correlation
        feature_data = df[top_features + ['label']].corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(feature_data, dtype=bool))
        
        sns.heatmap(feature_data, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f')
        
        plt.title('Feature Correlation Matrix (Top Features)')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_detailed_report(self, df: pd.DataFrame) -> str:
        """
        Generate detailed text report.
        """
        feature_aucs = self.results['feature_aucs']
        feature_groups = self.results['feature_groups']
        ranking = self.results['feature_ranking']
        
        report = []
        report.append("=" * 80)
        report.append("AUC FEATURE DIAGNOSTIC REPORT")
        report.append("=" * 80)
        report.append(f"Generated on: {pd.Timestamp.now()}")
        report.append(f"Dataset shape: {df.shape}")
        report.append(f"Anomaly rate: {df['label'].mean():.3f}")
        report.append(f"Total features analyzed: {len(feature_aucs)}")
        report.append("")
        
        # Summary statistics
        all_aucs = [stats['auc'] for stats in feature_aucs.values()]
        report.append("OVERALL FEATURE QUALITY SUMMARY")
        report.append("-" * 40)
        report.append(f"Mean AUC: {np.mean(all_aucs):.3f}")
        report.append(f"Median AUC: {np.median(all_aucs):.3f}")
        report.append(f"Std AUC: {np.std(all_aucs):.3f}")
        report.append(f"Min AUC: {np.min(all_aucs):.3f}")
        report.append(f"Max AUC: {np.max(all_aucs):.3f}")
        report.append(f"Features with AUC > 0.7: {sum(1 for auc in all_aucs if auc > 0.7)}")
        report.append(f"Features with AUC > 0.6: {sum(1 for auc in all_aucs if auc > 0.6)}")
        report.append("")
        
        # Top features
        report.append("TOP 20 FEATURES BY COMPOSITE SCORE")
        report.append("-" * 40)
        for i, row in ranking.iterrows():
            report.append(f"{i+1:2d}. {row['feature']:<30} "
                         f"AUC: {row['auc']:.3f} "
                         f"Score: {row['composite_score']:.3f}")
        report.append("")
        
        # Group analysis
        report.append("FEATURE GROUP ANALYSIS")
        report.append("-" * 40)
        for group_name, features in feature_groups.items():
            if features:
                group_aucs = [feature_aucs[f]['auc'] for f in features]
                report.append(f"{group_name.upper()}:")
                report.append(f"  Count: {len(features)}")
                report.append(f"  Mean AUC: {np.mean(group_aucs):.3f}")
                report.append(f"  Max AUC: {np.max(group_aucs):.3f}")
                report.append(f"  Best feature: {features[np.argmax(group_aucs)]}")
                report.append("")
        
        # Data quality issues
        report.append("DATA QUALITY ISSUES")
        report.append("-" * 40)
        high_missing = [(f, s['missing_rate']) for f, s in feature_aucs.items() 
                       if s['missing_rate'] > 0.1]
        if high_missing:
            report.append("Features with >10% missing values:")
            for feature, rate in sorted(high_missing, key=lambda x: x[1], reverse=True)[:10]:
                report.append(f"  {feature}: {rate:.1%}")
        else:
            report.append("No significant missing value issues detected.")
        report.append("")
        
        high_zero = [(f, s['zero_rate']) for f, s in feature_aucs.items() 
                    if s['zero_rate'] > 0.9]
        if high_zero:
            report.append("Features with >90% zero values:")
            for feature, rate in sorted(high_zero, key=lambda x: x[1], reverse=True)[:10]:
                report.append(f"  {feature}: {rate:.1%}")
        else:
            report.append("No sparse features detected.")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        
        best_features = ranking.head(10)['feature'].tolist()
        report.append("Recommended features for model training:")
        for i, feature in enumerate(best_features, 1):
            auc = feature_aucs[feature]['auc']
            report.append(f"  {i}. {feature} (AUC: {auc:.3f})")
        report.append("")
        
        poor_features = [f for f, s in feature_aucs.items() 
                        if abs(s['auc'] - 0.5) < 0.02]
        if poor_features:
            report.append(f"Consider removing {len(poor_features)} features with AUC ≈ 0.5 (no discriminative power)")
        
        redundant_candidates = ranking[ranking['composite_score'] < 0.3]['feature'].tolist()
        if redundant_candidates:
            report.append(f"Consider reviewing {len(redundant_candidates)} features with low composite scores")
        
        return "\n".join(report)
    
    def save_results(self, df: pd.DataFrame):
        """
        Save all results to files.
        """
        print("Saving results...")
        
        # Save feature ranking CSV
        self.results['feature_ranking'].to_csv(
            self.output_dir / 'feature_ranking.csv', index=False
        )
        
        # Save detailed feature statistics
        feature_stats_df = pd.DataFrame(self.results['feature_aucs']).T
        feature_stats_df.to_csv(self.output_dir / 'feature_statistics.csv')
        
        # Save results JSON
        # Convert numpy types for JSON serialization
        json_results = {}
        for key, value in self.results.items():
            if key == 'feature_aucs':
                json_results[key] = {
                    feature: {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                             for k, v in stats.items()}
                    for feature, stats in value.items()
                }
            elif key == 'feature_ranking':
                # Convert DataFrame to dictionary
                json_results[key] = value.to_dict('records')
            elif key == 'feature_groups':
                # Feature groups is already serializable
                json_results[key] = value
            else:
                json_results[key] = value
        
        with open(self.output_dir / 'diagnostic_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save detailed report
        report = self.generate_detailed_report(df)
        with open(self.output_dir / 'diagnostic_report.txt', 'w') as f:
            f.write(report)
        
        print(f"Results saved to {self.output_dir}")
    
    def run_full_diagnostic(self, data_path: Union[str, List[str]], 
                          sample_size: Optional[int] = None):
        """
        Run complete diagnostic analysis.
        
        Args:
            data_path: Path to data file(s)
            sample_size: Optional sample size for large datasets
        """
        # Load and prepare data
        df = self.load_and_prepare_data(data_path, sample_size)
        
        # Calculate feature AUCs
        feature_aucs = self.calculate_feature_auc(df)
        self.results['feature_aucs'] = feature_aucs
        
        # Group features
        feature_groups = self.identify_feature_groups(feature_aucs)
        self.results['feature_groups'] = feature_groups
        
        # Generate ranking
        feature_ranking = self.generate_feature_ranking(feature_aucs)
        self.results['feature_ranking'] = feature_ranking
        
        # Get top features for plotting
        top_features = feature_ranking.head(15)['feature'].tolist()
        
        # Generate plots
        print("Generating visualizations...")
        self.plot_feature_distributions(df, top_features)
        self.plot_auc_by_group(feature_aucs, feature_groups)
        self.plot_feature_correlation_matrix(df, top_features[:10])
        
        # Save all results
        self.save_results(df)
        
        # Print summary
        print("\n" + "="*60)
        print("DIAGNOSTIC SUMMARY")
        print("="*60)
        print(f"Total features analyzed: {len(feature_aucs)}")
        print(f"Best feature: {feature_ranking.iloc[0]['feature']} "
              f"(AUC: {feature_ranking.iloc[0]['auc']:.3f})")
        print(f"Mean AUC: {np.mean([s['auc'] for s in feature_aucs.values()]):.3f}")
        print(f"Features with AUC > 0.7: {sum(1 for s in feature_aucs.values() if s['auc'] > 0.7)}")
        print(f"Results saved to: {self.output_dir}")


def main():
    """
    Main function for command-line usage.
    """
    parser = argparse.ArgumentParser(description='AUC Feature Diagnostic Tool')
    parser.add_argument('data_path', help='Path to CSV file or directory containing CSV files')
    parser.add_argument('--output-dir', default='feature_diagnostic_results',
                       help='Output directory for results')
    parser.add_argument('--sample-size', type=int, 
                       help='Sample size for large datasets')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create diagnostic tool
    diagnostic = AUCFeatureDiagnostic(
        output_dir=args.output_dir,
        random_state=args.random_seed
    )
    
    # Run diagnostic
    diagnostic.run_full_diagnostic(
        data_path=args.data_path,
        sample_size=args.sample_size
    )


if __name__ == "__main__":
    main()