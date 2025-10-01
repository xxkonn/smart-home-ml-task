#!/bin/bash

# Isolation Forest Training - 3 Optimized Models
# Run this script to train 3 specific model configurations

echo "=== ISOLATION FOREST - 3 OPTIMIZED MODELS ==="
echo ""

# Activate virtual environment
source venv/bin/activate

# Clean up previous models, results, and logs (preserve results_v0.md)
echo "Cleaning up previous training artifacts..."
rm -f models/*.joblib
rm -f models/results/*.json
rm -f models/logs/*.log
echo "Cleanup completed"
echo ""

echo "Training 3 optimized models with bootstrap and warm-start enabled:"
echo ""

echo "1. Model: estimators_300 (Conservative precision)"
echo "   Parameters: 300 estimators, 0.7 max-samples, 0.03 contamination"
python src/train_isolation_forest.py --n-estimators 300 --max-samples 0.7 --contamination 0.03 --bootstrap --warm-start --model-name estimators_300
echo ""

echo "2. Model: estimators_500 (Balanced performance)" 
echo "   Parameters: 500 estimators, 0.9 max-samples, 0.02 contamination"
python src/train_isolation_forest.py --n-estimators 500 --max-samples 0.9 --contamination 0.02 --bootstrap --warm-start --model-name estimators_500
echo ""

echo "3. Model: estimators_700 (Maximum precision)"
echo "   Parameters: 700 estimators, 0.9 max-samples, 0.02 contamination"
python src/train_isolation_forest.py --n-estimators 700 --max-samples 0.9 --contamination 0.02 --bootstrap --warm-start --model-name estimators_700
echo ""

echo "=== ALL 3 MODELS TRAINING COMPLETED ==="
echo ""
echo "Models saved:"
echo "  • estimators_300.joblib"
echo "  • estimators_500.joblib" 
echo "  • estimators_700.joblib"
echo ""
echo "Results saved in models/results/"
echo "Logs saved in models/logs/"