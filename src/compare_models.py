"""
Model Comparison Script for Isolation Forest Parameter Tuning

This script allows testing multiple Isolation Forest configurations and compares
their performance across different parameter combinations.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from train_isolation_forest import IsolationForestTrainer


class ModelComparison:
    """
    Framework for comparing multiple Isolation Forest model configurations.
    """
    
    def __init__(self, project_root: str = None):
        if project_root is None:
            project_root = Path(__file__).parent.parent  # Go up from src/ to project root
        self.project_root = Path(project_root)
        self.results_root = self.project_root / "models" / "results" / "comparisons"
        self.results_root.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Define parameter combinations to test
        self.parameter_grid = [
            # Baseline model
            {
                'name': 'baseline',
                'n_estimators': 100,
                'contamination': 0.1,
                'max_samples': 'auto',
                'max_features': 1.0,
                'bootstrap': False
            },
            # High precision model (lower contamination)
            {
                'name': 'high_precision',
                'n_estimators': 100,
                'contamination': 0.05,
                'max_samples': 'auto',
                'max_features': 1.0,
                'bootstrap': False
            },
            # High recall model (higher contamination)
            {
                'name': 'high_recall',
                'n_estimators': 100,
                'contamination': 0.15,
                'max_samples': 'auto',
                'max_features': 1.0,
                'bootstrap': False
            },
            # More estimators for stability
            {
                'name': 'stable_200_trees',
                'n_estimators': 200,
                'contamination': 0.1,
                'max_samples': 'auto',
                'max_features': 1.0,
                'bootstrap': False
            },
            # Limited features for speed
            {
                'name': 'fast_limited_features',
                'n_estimators': 100,
                'contamination': 0.1,
                'max_samples': 'auto',
                'max_features': 0.7,
                'bootstrap': False
            },
            # Bootstrap for variance reduction
            {
                'name': 'bootstrap_ensemble',
                'n_estimators': 150,
                'contamination': 0.1,
                'max_samples': 'auto',
                'max_features': 1.0,
                'bootstrap': True
            },
            # Limited samples for speed
            {
                'name': 'fast_limited_samples',
                'n_estimators': 100,
                'contamination': 0.1,
                'max_samples': 256,
                'max_features': 1.0,
                'bootstrap': False
            },
            # Conservative detection
            {
                'name': 'conservative',
                'n_estimators': 200,
                'contamination': 0.05,
                'max_samples': 'auto',
                'max_features': 0.8,
                'bootstrap': True
            },
            # Aggressive detection
            {
                'name': 'aggressive',
                'n_estimators': 150,
                'contamination': 0.2,
                'max_samples': 'auto',
                'max_features': 1.0,
                'bootstrap': False
            }
        ]
        
        # Feature configuration combinations
        self.feature_configs = [
            {
                'name': 'full_features',
                'use_time_features': True,
                'use_rolling_stats': True,
                'rolling_windows': [6, 12, 24],
                'use_lag_features': True,
                'lag_periods': [1, 2, 3],
                'use_statistical_features': True
            },
            {
                'name': 'minimal_features',
                'use_time_features': True,
                'use_rolling_stats': True,
                'rolling_windows': [12],
                'use_lag_features': False,
                'lag_periods': [],
                'use_statistical_features': False
            },
            {
                'name': 'time_only',
                'use_time_features': True,
                'use_rolling_stats': False,
                'rolling_windows': [],
                'use_lag_features': False,
                'lag_periods': [],
                'use_statistical_features': False
            },
            {
                'name': 'stats_heavy',
                'use_time_features': True,
                'use_rolling_stats': True,
                'rolling_windows': [6, 12, 24, 48],
                'use_lag_features': True,
                'lag_periods': [1, 2, 3, 6, 12],
                'use_statistical_features': True
            }
        ]
        
    def setup_logging(self):
        """Setup logging for model comparison."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.results_root / f"model_comparison_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Model comparison logging initialized. Log file: {log_file}")
        
    def run_single_experiment(self, model_params: Dict, feature_config: Dict) -> Dict[str, Any]:
        """
        Run a single model training experiment.
        
        Args:
            model_params: Model parameters to use
            feature_config: Feature engineering configuration
            
        Returns:
            Experiment results
        """
        experiment_name = f"{model_params['name']}_{feature_config['name']}"
        self.logger.info(f"Running experiment: {experiment_name}")
        
        # Create trainer with custom configuration
        trainer = IsolationForestTrainer()
        trainer.model_params.update({k: v for k, v in model_params.items() if k != 'name'})
        trainer.feature_config.update({k: v for k, v in feature_config.items() if k != 'name'})
        
        # Run training and validation
        try:
            results = trainer.train_and_validate(experiment_name)
            results['experiment_config'] = {
                'model_params': model_params,
                'feature_config': feature_config,
                'experiment_name': experiment_name
            }
            
            self.logger.info(f"Experiment {experiment_name} completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Experiment {experiment_name} failed: {str(e)}")
            return {
                'experiment_config': {
                    'model_params': model_params,
                    'feature_config': feature_config,
                    'experiment_name': experiment_name
                },
                'error': str(e),
                'status': 'failed'
            }
    
    def run_all_experiments(self, max_experiments: int = None) -> List[Dict[str, Any]]:
        """
        Run all parameter combinations.
        
        Args:
            max_experiments: Maximum number of experiments to run (None for all)
            
        Returns:
            List of all experiment results
        """
        self.logger.info("=" * 100)
        self.logger.info("STARTING MODEL COMPARISON EXPERIMENTS")
        self.logger.info("=" * 100)
        
        all_results = []
        experiment_count = 0
        
        for model_params in self.parameter_grid:
            for feature_config in self.feature_configs:
                if max_experiments and experiment_count >= max_experiments:
                    break
                    
                experiment_count += 1
                self.logger.info(f"Experiment {experiment_count}: {model_params['name']} + {feature_config['name']}")
                
                results = self.run_single_experiment(model_params, feature_config)
                all_results.append(results)
                
                # Save intermediate results
                self.save_comparison_results(all_results, f"intermediate_{experiment_count}")
                
            if max_experiments and experiment_count >= max_experiments:
                break
        
        # Save final results
        self.save_comparison_results(all_results, "final")
        
        self.logger.info("=" * 100)
        self.logger.info("ALL EXPERIMENTS COMPLETED")
        self.logger.info("=" * 100)
        
        return all_results
    
    def save_comparison_results(self, results: List[Dict[str, Any]], suffix: str = ""):
        """Save comparison results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_comparison_{timestamp}"
        if suffix:
            filename += f"_{suffix}"
        filename += ".json"
        
        results_path = self.results_root / filename
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Comparison results saved to: {results_path}")
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze and rank experiment results.
        
        Args:
            results: List of experiment results
            
        Returns:
            Analysis summary
        """
        self.logger.info("Analyzing experiment results...")
        
        successful_results = [r for r in results if 'error' not in r]
        
        if not successful_results:
            self.logger.warning("No successful experiments to analyze")
            return {'error': 'No successful experiments'}
        
        # Extract key metrics for ranking
        rankings = []
        for result in successful_results:
            test_metrics = result.get('test_results', {}).get('overall_metrics', {})
            val_metrics = result.get('validation_results', {}).get('overall_metrics', {})
            
            config = result.get('experiment_config', {})
            experiment_name = config.get('experiment_name', 'unknown')
            
            # Calculate combined score (weighted F1 + recall)
            test_f1 = test_metrics.get('f1_score', 0)
            test_recall = test_metrics.get('recall', 0)
            val_f1 = val_metrics.get('f1_score', 0)
            val_recall = val_metrics.get('recall', 0)
            
            combined_score = (test_f1 * 0.4 + test_recall * 0.3 + val_f1 * 0.2 + val_recall * 0.1)
            
            rankings.append({
                'experiment_name': experiment_name,
                'combined_score': combined_score,
                'test_f1': test_f1,
                'test_recall': test_recall,
                'test_precision': test_metrics.get('precision', 0),
                'val_f1': val_f1,
                'val_recall': val_recall,
                'val_precision': val_metrics.get('precision', 0),
                'model_params': config.get('model_params', {}),
                'feature_config': config.get('feature_config', {})
            })
        
        # Sort by combined score
        rankings.sort(key=lambda x: x['combined_score'], reverse=True)
        
        analysis = {
            'total_experiments': len(results),
            'successful_experiments': len(successful_results),
            'failed_experiments': len(results) - len(successful_results),
            'best_model': rankings[0] if rankings else None,
            'top_5_models': rankings[:5],
            'all_rankings': rankings
        }
        
        # Log top results
        self.logger.info("TOP 5 MODEL CONFIGURATIONS:")
        for i, model in enumerate(rankings[:5], 1):
            self.logger.info(f"  {i}. {model['experiment_name']}")
            self.logger.info(f"     Combined Score: {model['combined_score']:.4f}")
            self.logger.info(f"     Test F1: {model['test_f1']:.4f}, Test Recall: {model['test_recall']:.4f}")
            self.logger.info(f"     Val F1: {model['val_f1']:.4f}, Val Recall: {model['val_recall']:.4f}")
        
        return analysis
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """Print a comprehensive summary of all experiments."""
        analysis = self.analyze_results(results)
        
        print("\n" + "="*100)
        print("MODEL COMPARISON SUMMARY")
        print("="*100)
        
        print(f"\n📊 EXPERIMENT OVERVIEW:")
        print(f"   Total Experiments: {analysis['total_experiments']}")
        print(f"   Successful: {analysis['successful_experiments']}")
        print(f"   Failed: {analysis['failed_experiments']}")
        
        if analysis['best_model']:
            best = analysis['best_model']
            print(f"\n🏆 BEST MODEL: {best['experiment_name']}")
            print(f"   Combined Score: {best['combined_score']:.4f}")
            print(f"   Test Results - F1: {best['test_f1']:.4f}, Recall: {best['test_recall']:.4f}, Precision: {best['test_precision']:.4f}")
            print(f"   Validation Results - F1: {best['val_f1']:.4f}, Recall: {best['val_recall']:.4f}, Precision: {best['val_precision']:.4f}")
        
        print(f"\n🥇 TOP 5 MODELS:")
        for i, model in enumerate(analysis['top_5_models'], 1):
            print(f"   {i}. {model['experiment_name']} (Score: {model['combined_score']:.4f})")


def main():
    """Main comparison script."""
    comparison = ModelComparison()
    
    print("Starting model comparison experiments...")
    print("This will test multiple parameter combinations and feature configurations.")
    print("Each experiment trains a new Isolation Forest model and evaluates it.")
    
    # Run all experiments (or limit for testing)
    results = comparison.run_all_experiments(max_experiments=None)  # Set to small number for testing
    
    # Print summary
    comparison.print_summary(results)
    
    # Save analysis
    analysis = comparison.analyze_results(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_path = comparison.results_root / f"analysis_{timestamp}.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n📁 Results saved to: {comparison.results_root}")
    print(f"📈 Analysis saved to: {analysis_path}")


if __name__ == "__main__":
    main()