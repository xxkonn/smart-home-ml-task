"""
Quick Model Comparison Test - Test 2 configurations for validation
"""

from compare_models import ModelComparison

def main():
    """Run a quick test with just 2 model configurations."""
    comparison = ModelComparison()
    
    # Limit to just 2 configurations for testing
    comparison.parameter_grid = [
        {
            'name': 'baseline',
            'n_estimators': 100,
            'contamination': 0.1,
            'max_samples': 'auto',
            'max_features': 1.0,
            'bootstrap': False
        },
        {
            'name': 'high_recall',
            'n_estimators': 100,
            'contamination': 0.15,
            'max_samples': 'auto',
            'max_features': 1.0,
            'bootstrap': False
        }
    ]
    
    # Use only minimal features for speed
    comparison.feature_configs = [
        {
            'name': 'minimal_features',
            'use_time_features': True,
            'use_rolling_stats': True,
            'rolling_windows': [12],
            'use_lag_features': False,
            'lag_periods': [],
            'use_statistical_features': False
        }
    ]
    
    print("Running quick model comparison test...")
    print("Testing 2 model configurations with minimal features...")
    
    # Run limited experiments
    results = comparison.run_all_experiments(max_experiments=2)
    
    # Print summary
    comparison.print_summary(results)

if __name__ == "__main__":
    main()