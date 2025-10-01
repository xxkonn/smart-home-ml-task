"""
Isolation Forest Training Pipeline for Smart Home IoT Anomaly Detection

This script provides comprehensive training, validation, and evaluation of Isolation Forest models
with detailed logging, parameter tuning, and anomaly type-specific metrics.
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import joblib

# Import our feature engineering module
from feature_engineering import engineer_all_features


class IsolationForestTrainer:
    """
    Comprehensive Isolation Forest trainer with logging, validation, and metrics tracking.
    """
    
    def __init__(self, project_root: str = None, **model_params):
        if project_root is None:
            project_root = Path(__file__).parent.parent  # Go up from src/ to project root
        self.project_root = Path(project_root)
        self.data_root = self.project_root / "data"
        self.models_root = self.project_root / "models"
        self.logs_root = self.project_root / "models" / "logs"
        self.results_root = self.project_root / "models" / "results"
        
        # Create directories
        self.models_root.mkdir(exist_ok=True)
        self.logs_root.mkdir(exist_ok=True)
        self.results_root.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Model parameters (exposed for fine-tuning)
        self.model_params = {
            'n_estimators': 100,
            'contamination': 0.1,
            'max_samples': 'auto',
            'max_features': 1.0,
            'bootstrap': False,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Update with any provided parameters
        if model_params:
            self.model_params.update(model_params)
            self.logger.info(f"Updated model parameters: {model_params}")
        
        # Feature engineering settings
        self.feature_config = {
            'use_time_features': True,
            'use_rolling_stats': True,
            'rolling_windows': [6, 12, 24],  # 30min, 1hr, 2hr windows
            'use_lag_features': True,
            'lag_periods': [1, 2, 3],
            'use_statistical_features': True
        }
        
        self.features_used = []
        self.scaler = StandardScaler()
        
    def setup_logging(self):
        """Setup comprehensive logging system."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.logs_root / f"isolation_forest_training_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized. Log file: {log_file}")
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer comprehensive features using the dedicated feature engineering module.
        
        Args:
            df: Input dataframe with sensor data
            
        Returns:
            DataFrame with engineered features
        """
        self.logger.info("Starting feature engineering...")
        
        # Use the dedicated feature engineering module
        df_features = engineer_all_features(df)
        
        # Remove timestamp and label columns for model training
        feature_columns = [col for col in df_features.columns if col not in ['timestamp', 'label', 'anomaly_type', 'house_id']]
        self.features_used = feature_columns
        
        self.logger.info(f"Feature engineering complete. Generated {len(feature_columns)} features")
        self.logger.info(f"Features used: {feature_columns}")
        
        return df_features
    
    def load_training_data(self) -> pd.DataFrame:
        """Load and combine all training data."""
        self.logger.info("Loading training data...")
        
        train_files = list((self.data_root / "unsupervised" / "train").glob("*.csv"))
        train_data = []
        
        for file_path in train_files:
            df = pd.read_csv(file_path)
            train_data.append(df)
            
        combined_data = pd.concat(train_data, ignore_index=True)
        self.logger.info(f"Loaded {len(train_data)} training files with {len(combined_data)} total samples")
        
        return combined_data
    
    def load_validation_data(self) -> pd.DataFrame:
        """Load and combine all validation data."""
        self.logger.info("Loading validation data...")
        
        val_files = list((self.data_root / "unsupervised" / "val").glob("*.csv"))
        val_data = []
        
        for file_path in val_files:
            df = pd.read_csv(file_path)
            val_data.append(df)
            
        combined_data = pd.concat(val_data, ignore_index=True)
        self.logger.info(f"Loaded {len(val_data)} validation files with {len(combined_data)} total samples")
        
        return combined_data
    
    def load_test_data(self) -> pd.DataFrame:
        """Load and combine all test data."""
        self.logger.info("Loading test data...")
        
        test_files = list((self.data_root / "test").glob("*.csv"))
        test_data = []
        
        for file_path in test_files:
            df = pd.read_csv(file_path)
            test_data.append(df)
            
        combined_data = pd.concat(test_data, ignore_index=True)
        self.logger.info(f"Loaded {len(test_data)} test files with {len(combined_data)} total samples")
        
        return combined_data
    
    def train_model(self, train_data: pd.DataFrame) -> IsolationForest:
        """
        Train Isolation Forest model.
        
        Args:
            train_data: Training dataset (normal data only)
            
        Returns:
            Trained Isolation Forest model
        """
        self.logger.info("Starting model training...")
        self.logger.info(f"Model parameters: {self.model_params}")
        
        # Engineer features
        train_features = self.engineer_features(train_data)
        
        # Prepare training data (only normal samples)
        X_train = train_features[self.features_used]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        model = IsolationForest(**self.model_params)
        model.fit(X_train_scaled)
        
        self.logger.info(f"Model training completed with {len(X_train)} samples")
        self.logger.info(f"Feature scaling applied with scaler: {type(self.scaler).__name__}")
        
        return model
    
    def evaluate_model(self, model: IsolationForest, eval_data: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Evaluate model performance with detailed metrics.
        
        Args:
            model: Trained Isolation Forest model
            eval_data: Evaluation dataset
            dataset_name: Name of the dataset (for logging)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.logger.info(f"Evaluating model on {dataset_name} data...")
        
        # Engineer features
        eval_features = self.engineer_features(eval_data)
        
        # Prepare evaluation data
        X_eval = eval_features[self.features_used]
        X_eval_scaled = self.scaler.transform(X_eval)
        
        # Get true labels
        y_true = eval_data['label'].values
        anomaly_types = eval_data['anomaly_type'].values
        
        # Predict anomalies (-1 for anomaly, 1 for normal)
        predictions = model.predict(X_eval_scaled)
        
        # Convert to binary format (1 for anomaly, 0 for normal)
        y_pred = (predictions == -1).astype(int)
        
        # Calculate overall metrics
        if y_true.sum() > 0:  # If there are anomalies
            precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        else:  # If no anomalies (validation set case)
            precision, recall, f1, support = 0.0, 0.0, 0.0, 0
            if y_pred.sum() == 0:  # No false positives
                precision = 1.0
        
        # Calculate anomaly type-specific metrics
        anomaly_type_metrics = {}
        for anomaly_type in range(1, 7):  # Types 1-6
            mask = anomaly_types == anomaly_type
            if mask.sum() > 0:
                type_precision, type_recall, type_f1, type_support = precision_recall_fscore_support(
                    y_true[mask], y_pred[mask], average='binary', zero_division=0
                )
                anomaly_type_metrics[f'type_{anomaly_type}'] = {
                    'precision': float(type_precision),
                    'recall': float(type_recall),
                    'f1_score': float(type_f1),
                    'support': int(type_support) if type_support is not None else 0,
                    'samples': int(mask.sum())
                }
        
        # Calculate confusion matrix
        if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        else:
            # Handle case where all labels are the same
            if y_true.sum() == 0:  # All normal
                tn = len(y_true) - y_pred.sum()
                fp = y_pred.sum()
                fn = 0
                tp = 0
            else:  # All anomalies (shouldn't happen in our case)
                tn = 0
                fp = 0
                fn = len(y_true) - y_pred.sum()
                tp = y_pred.sum()
        
        metrics = {
            'dataset': dataset_name,
            'overall_metrics': {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'support': int(support) if support is not None else 0,
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'total_samples': len(y_true),
                'total_anomalies': int(y_true.sum()),
                'total_normal': int((y_true == 0).sum())
            },
            'anomaly_type_metrics': anomaly_type_metrics,
            'model_parameters': self.model_params,
            'features_used': self.features_used,
            'feature_config': self.feature_config
        }
        
        self.logger.info(f"{dataset_name} Evaluation Results:")
        self.logger.info(f"  Overall Precision: {precision:.4f}")
        self.logger.info(f"  Overall Recall: {recall:.4f}")
        self.logger.info(f"  Overall F1-Score: {f1:.4f}")
        self.logger.info(f"  Total Samples: {len(y_true)}")
        self.logger.info(f"  True Anomalies: {y_true.sum()}")
        
        for anomaly_type, type_metrics in anomaly_type_metrics.items():
            self.logger.info(f"  {anomaly_type.upper()} - Recall: {type_metrics['recall']:.4f}, "
                           f"Samples: {type_metrics['samples']}")
        
        return metrics
    
    def save_model_and_results(self, model: IsolationForest, results: Dict[str, Any], 
                              model_name: str = None) -> str:
        """
        Save trained model and results.
        
        Args:
            model: Trained model
            results: Evaluation results
            model_name: Custom model name
            
        Returns:
            Path to saved model
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if model_name is None:
            model_name = f"isolation_forest_{timestamp}"
        
        # Save model
        model_path = self.models_root / f"{model_name}.joblib"
        joblib.dump({
            'model': model,
            'scaler': self.scaler,
            'features_used': self.features_used,
            'model_params': self.model_params,
            'feature_config': self.feature_config
        }, model_path)
        
        # Save results
        results_path = self.results_root / f"{model_name}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Model saved to: {model_path}")
        self.logger.info(f"Results saved to: {results_path}")
        
        return str(model_path)
    
    def train_and_validate(self, model_name: str = None) -> Dict[str, Any]:
        """
        Complete training and validation pipeline.
        
        Args:
            model_name: Custom name for the model
            
        Returns:
            Combined results from training and validation
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING ISOLATION FOREST TRAINING AND VALIDATION")
        self.logger.info("=" * 80)
        
        # Load data
        train_data = self.load_training_data()
        val_data = self.load_validation_data()
        test_data = self.load_test_data()
        
        # Train model
        model = self.train_model(train_data)
        
        # Evaluate on validation and test sets
        val_results = self.evaluate_model(model, val_data, "validation")
        test_results = self.evaluate_model(model, test_data, "test")
        
        # Combine results
        combined_results = {
            'training_info': {
                'timestamp': datetime.now().isoformat(),
                'training_samples': len(train_data),
                'validation_samples': len(val_data),
                'test_samples': len(test_data),
                'model_type': 'IsolationForest'
            },
            'validation_results': val_results,
            'test_results': test_results
        }
        
        # Save model and results
        model_path = self.save_model_and_results(model, combined_results, model_name)
        
        self.logger.info("=" * 80)
        self.logger.info("TRAINING AND VALIDATION COMPLETED")
        self.logger.info("=" * 80)
        
        return combined_results


def parse_arguments():
    """Parse command line arguments for model parameters."""
    parser = argparse.ArgumentParser(
        description='Train Isolation Forest for Smart Home IoT Anomaly Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model parameters
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='Number of base estimators in the ensemble')
    parser.add_argument('--contamination', type=float, default=0.1,
                       help='Expected proportion of outliers in the data (0.0 to 0.5)')
    parser.add_argument('--max-samples', type=str, default='auto',
                       help='Number of samples to draw to train each base estimator (int or "auto")')
    parser.add_argument('--max-features', type=float, default=1.0,
                       help='Number of features to draw to train each base estimator (0.0 to 1.0)')
    parser.add_argument('--bootstrap', action='store_true', default=False,
                       help='Whether samples are drawn with replacement')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility')
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of jobs to run in parallel (-1 means using all processors)')
    
    # Model naming
    parser.add_argument('--model-name', type=str, default=None,
                       help='Custom name for the saved model')
    
    # Feature engineering options
    parser.add_argument('--no-time-features', action='store_true', default=False,
                       help='Disable time-based features')
    parser.add_argument('--no-rolling-stats', action='store_true', default=False,
                       help='Disable rolling statistics features')
    parser.add_argument('--no-lag-features', action='store_true', default=False,
                       help='Disable lag features')
    parser.add_argument('--no-statistical-features', action='store_true', default=False,
                       help='Disable statistical interaction features')
    
    # Rolling window configuration
    parser.add_argument('--rolling-windows', nargs='+', type=int, default=[6, 12, 24],
                       help='Rolling window sizes (in 5-min intervals)')
    parser.add_argument('--lag-periods', nargs='+', type=int, default=[1, 2, 3],
                       help='Lag periods for lag features')
    
    return parser.parse_args()


def main():
    """Main training script with command-line parameter support."""
    args = parse_arguments()
    
    # Convert max_samples to appropriate type
    max_samples = args.max_samples
    if max_samples != 'auto':
        try:
            max_samples = int(max_samples)
        except ValueError:
            print(f"Warning: Invalid max_samples value '{max_samples}', using 'auto'")
            max_samples = 'auto'
    
    # Prepare model parameters
    model_params = {
        'n_estimators': args.n_estimators,
        'contamination': args.contamination,
        'max_samples': max_samples,
        'max_features': args.max_features,
        'bootstrap': args.bootstrap,
        'random_state': args.random_state,
        'n_jobs': args.n_jobs
    }
    
    # Create trainer with custom parameters
    trainer = IsolationForestTrainer(**model_params)
    
    # Configure feature engineering
    trainer.feature_config.update({
        'use_time_features': not args.no_time_features,
        'use_rolling_stats': not args.no_rolling_stats,
        'rolling_windows': args.rolling_windows,
        'use_lag_features': not args.no_lag_features,
        'lag_periods': args.lag_periods,
        'use_statistical_features': not args.no_statistical_features
    })
    
    # Generate model name if not provided
    model_name = args.model_name
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"isolation_forest_{timestamp}"
    
    print("="*80)
    print("ISOLATION FOREST TRAINING CONFIGURATION")
    print("="*80)
    print(f"Model Parameters:")
    for key, value in model_params.items():
        print(f"  {key}: {value}")
    print(f"\nFeature Configuration:")
    for key, value in trainer.feature_config.items():
        print(f"  {key}: {value}")
    print(f"\nModel Name: {model_name}")
    print("="*80)
    
    # Run training and validation
    results = trainer.train_and_validate(model_name)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED - SUMMARY RESULTS")
    print("="*80)
    
    # Print validation results
    val_metrics = results['validation_results']['overall_metrics']
    print(f"\n📊 VALIDATION RESULTS:")
    print(f"   Precision: {val_metrics['precision']:.4f}")
    print(f"   Recall: {val_metrics['recall']:.4f}")
    print(f"   F1-Score: {val_metrics['f1_score']:.4f}")
    print(f"   Total Samples: {val_metrics['total_samples']:,}")
    print(f"   False Positives: {val_metrics['false_positives']}")
    
    # Print test results
    test_metrics = results['test_results']['overall_metrics'] 
    print(f"\n🎯 TEST RESULTS:")
    print(f"   Precision: {test_metrics['precision']:.4f}")
    print(f"   Recall: {test_metrics['recall']:.4f}")
    print(f"   F1-Score: {test_metrics['f1_score']:.4f}")
    print(f"   Total Samples: {test_metrics['total_samples']:,}")
    print(f"   Anomalies Detected: {test_metrics['true_positives']}/{test_metrics['total_anomalies']}")
    
    # Print anomaly type results
    print(f"\n🔍 ANOMALY TYPE RECALL (TEST SET):")
    for anomaly_type, metrics in results['test_results']['anomaly_type_metrics'].items():
        type_name = {
            'type_1': 'AC Failure',
            'type_2': 'Fridge Breakdown', 
            'type_3': 'Nighttime Intrusion',
            'type_4': 'Door Left Open',
            'type_5': 'Window Stuck',
            'type_6': 'Fridge Door Open'
        }.get(anomaly_type, anomaly_type)
        print(f"   {type_name}: {metrics['recall']:.4f} ({metrics['samples']} samples)")
    
    # Print parameter recommendations
    print(f"\n💡 PARAMETER INSIGHTS:")
    if test_metrics['precision'] < 0.1:
        print("   • Low precision - consider decreasing contamination parameter")
    if test_metrics['recall'] < 0.3:
        print("   • Low recall - consider increasing contamination parameter")
    if test_metrics['f1_score'] < 0.2:
        print("   • Low F1-score - try different max_samples or more estimators")
    
    print(f"\n📁 Model saved as: {model_name}")
    print(f"📊 Results saved to: models/{model_name}_results.json")


def main_simple():
    """Simple main function for backward compatibility."""
    trainer = IsolationForestTrainer()
    
    # Run training and validation
    results = trainer.train_and_validate("isolation_forest_baseline")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED - SUMMARY RESULTS") 
    print("="*80)
    
    # Print validation results
    val_metrics = results['validation_results']['overall_metrics']
    print(f"\n📊 VALIDATION RESULTS:")
    print(f"   Precision: {val_metrics['precision']:.4f}")
    print(f"   Recall: {val_metrics['recall']:.4f}")
    print(f"   F1-Score: {val_metrics['f1_score']:.4f}")
    print(f"   Total Samples: {val_metrics['total_samples']:,}")
    print(f"   Anomalies Detected: {val_metrics['true_positives']}/{val_metrics['total_anomalies']}")
    
    # Print test results
    test_metrics = results['test_results']['overall_metrics'] 
    print(f"\n🎯 TEST RESULTS:")
    print(f"   Precision: {test_metrics['precision']:.4f}")
    print(f"   Recall: {test_metrics['recall']:.4f}")
    print(f"   F1-Score: {test_metrics['f1_score']:.4f}")
    print(f"   Total Samples: {test_metrics['total_samples']:,}")
    print(f"   Anomalies Detected: {test_metrics['true_positives']}/{test_metrics['total_anomalies']}")
    
    # Print anomaly type results
    print(f"\n🔍 ANOMALY TYPE RECALL (TEST SET):")
    for anomaly_type, metrics in results['test_results']['anomaly_type_metrics'].items():
        print(f"   {anomaly_type.upper()}: {metrics['recall']:.4f} ({metrics['samples']} samples)")


if __name__ == "__main__":
    main()