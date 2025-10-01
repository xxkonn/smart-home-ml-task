#!/bin/bash

# Isolation Forest Training Examples
# Run this script to test different parameter combinations

echo "=== ISOLATION FOREST PARAMETER EXAMPLES ==="
echo ""

# Activate virtual environment
source venv/bin/activate

echo "1. BASELINE MODEL (Default parameters)"
echo "Command: python src/train_isolation_forest.py --model-name baseline"
echo ""

echo "2. HIGH PRECISION MODEL (Lower contamination)"
echo "Command: python src/train_isolation_forest.py --contamination 0.05 --model-name high_precision"
echo ""

echo "3. HIGH RECALL MODEL (Higher contamination)"
echo "Command: python src/train_isolation_forest.py --contamination 0.15 --model-name high_recall"
echo ""

echo "4. STABLE MODEL (More estimators)"
echo "Command: python src/train_isolation_forest.py --n-estimators 200 --model-name stable"
echo ""

echo "5. FAST MODEL (Limited samples and features)"
echo "Command: python src/train_isolation_forest.py --max-samples 256 --max-features 0.7 --model-name fast"
echo ""

echo "6. CONSERVATIVE MODEL (Lower contamination + bootstrap)"
echo "Command: python src/train_isolation_forest.py --contamination 0.05 --bootstrap --n-estimators 150 --model-name conservative"
echo ""

echo "7. AGGRESSIVE MODEL (Higher contamination)"
echo "Command: python src/train_isolation_forest.py --contamination 0.2 --model-name aggressive"
echo ""

echo "8. MINIMAL FEATURES MODEL (No lag features, minimal rolling windows)"
echo "Command: python src/train_isolation_forest.py --no-lag-features --rolling-windows 12 --model-name minimal_features"
echo ""

echo "9. TIME-FOCUSED MODEL (Only time features)"
echo "Command: python src/train_isolation_forest.py --no-rolling-stats --no-lag-features --no-statistical-features --model-name time_only"
echo ""

echo "10. CUSTOM HIGH PRECISION MODEL (300 estimators, conservative sampling)"
echo "Command: python src/train_isolation_forest.py --n-estimators 300 --max-samples 0.7 --contamination 0.03 --model-name custom_high_precision"
echo ""

echo "11. CUSTOM ULTRA PRECISION MODEL (500 estimators, extensive sampling)"
echo "Command: python src/train_isolation_forest.py --n-estimators 500 --max-samples 0.9 --contamination 0.02 --model-name custom_ultra_precision"
echo ""

echo "12. CUSTOM MAXIMUM PRECISION MODEL (700 estimators, maximum sampling)"
echo "Command: python src/train_isolation_forest.py --n-estimators 700 --max-samples 0.9 --contamination 0.02 --model-name custom_max_precision"
echo ""

echo "=== Parameter Explanations ==="
echo ""
echo "--contamination: Expected fraction of anomalies (0.05-0.2)"
echo "  • Lower values → Higher precision, Lower recall"
echo "  • Higher values → Lower precision, Higher recall"
echo ""
echo "--n-estimators: Number of trees in ensemble (50-300)"
echo "  • More trees → More stable but slower"
echo ""
echo "--max-samples: Samples per tree ('auto' or integer)"
echo "  • 'auto' uses min(256, n_samples)"
echo "  • Lower values → Faster training"
echo ""
echo "--max-features: Feature fraction per tree (0.5-1.0)"
echo "  • Lower values → More diversity, potentially better generalization"
echo ""
echo "--bootstrap: Use bootstrap sampling"
echo "  • Can improve stability but may reduce diversity"
echo ""
echo "Feature Engineering Options:"
echo "  --no-time-features: Disable hour/day features"
echo "  --no-rolling-stats: Disable rolling mean/std/min/max"
echo "  --no-lag-features: Disable previous timestep features"
echo "  --no-statistical-features: Disable interaction features"
echo ""

# Ask user which one to run
echo "Which example would you like to run? (Enter number 1-12, or 'all' for all):"
read choice

case $choice in
    1)
        python src/train_isolation_forest.py --model-name baseline
        ;;
    2)
        python src/train_isolation_forest.py --contamination 0.05 --model-name high_precision
        ;;
    3)
        python src/train_isolation_forest.py --contamination 0.15 --model-name high_recall
        ;;
    4)
        python src/train_isolation_forest.py --n-estimators 200 --model-name stable
        ;;
    5)
        python src/train_isolation_forest.py --max-samples 256 --max-features 0.7 --model-name fast
        ;;
    6)
        python src/train_isolation_forest.py --contamination 0.05 --bootstrap --n-estimators 150 --model-name conservative
        ;;
    7)
        python src/train_isolation_forest.py --contamination 0.2 --model-name aggressive
        ;;
    8)
        python src/train_isolation_forest.py --no-lag-features --rolling-windows 12 --model-name minimal_features
        ;;
    9)
        python src/train_isolation_forest.py --no-rolling-stats --no-lag-features --no-statistical-features --model-name time_only
        ;;
    10)
        python src/train_isolation_forest.py --n-estimators 300 --max-samples 0.7 --contamination 0.03 --model-name custom_high_precision
        ;;
    11)
        python src/train_isolation_forest.py --n-estimators 500 --max-samples 0.9 --contamination 0.02 --model-name custom_ultra_precision
        ;;
    12)
        python src/train_isolation_forest.py --n-estimators 700 --max-samples 0.9 --contamination 0.02 --model-name custom_max_precision
        ;;
    all)
        echo "Running all examples..."
        python src/train_isolation_forest.py --model-name baseline
        python src/train_isolation_forest.py --contamination 0.05 --model-name high_precision
        python src/train_isolation_forest.py --contamination 0.15 --model-name high_recall
        python src/train_isolation_forest.py --n-estimators 200 --model-name stable
        python src/train_isolation_forest.py --max-samples 256 --max-features 0.7 --model-name fast
        python src/train_isolation_forest.py --contamination 0.05 --bootstrap --n-estimators 150 --model-name conservative
        python src/train_isolation_forest.py --contamination 0.2 --model-name aggressive
        python src/train_isolation_forest.py --no-lag-features --rolling-windows 12 --model-name minimal_features
        python src/train_isolation_forest.py --no-rolling-stats --no-lag-features --no-statistical-features --model-name time_only
        python src/train_isolation_forest.py --n-estimators 300 --max-samples 0.7 --contamination 0.03 --model-name custom_high_precision
        python src/train_isolation_forest.py --n-estimators 500 --max-samples 0.9 --contamination 0.02 --model-name custom_ultra_precision
        python src/train_isolation_forest.py --n-estimators 700 --max-samples 0.9 --contamination 0.02 --model-name custom_max_precision
        ;;
    *)
        echo "Invalid choice. Please run the script again."
        ;;
esac