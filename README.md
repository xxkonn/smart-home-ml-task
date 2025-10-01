# Smart Home IoT Anomaly Detection Dataset

## Overview

This repository contains a synthetic smart home IoT dataset designed for anomaly detection research and machine learning applications. The dataset simulates 2 weeks of sensor data from 100 different houses in Sydney, Australia during summer conditions (December 2024 - February 2025).

## Dataset Specifications

### Environment Context
- **Location**: Sydney, Australia
- **Season**: Summer (December-February)
- **House Type**: residential homes
- **Duration**: 2 weeks per house
- **Sampling Rate**: 5-minute intervals
- **Training set**:80 weeks traning set
    - No anomaly appear all possitive data
- **Validation set**: 10 weeks validation set
- **Testing set**: 10 weeks default anomaly appear rate

### Data Structure

Each house dataset contains with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | DateTime | 5-minute interval timestamps |
| `temperature_living_room` | Float | Room temperature in Celsius (20-33°C range) |
| `window_open` | Integer | Window state (0=closed, 1=open) |
| `power_consumption_fridge` | Float | Refrigerator power consumption in watts |
| `motion_detected_hallway` | Integer | Motion sensor state (0=no motion, 1=motion detected) |
| `door_state_front` | Integer | Front door state (0=closed, 1=open) |
| `label` | Integer | Anomaly flag (0=normal, 1=anomaly) |
| `anomaly_type` | Integer | Specific anomaly identifier (0=normal, 1-6=anomaly types) |
| `house_id` | Integer | Unique house identifier (1-100) |

### Normal Behavior Patterns

The dataset simulates realistic household patterns:

- **Temperature**: Daily cycles with peak heat during afternoon, cooler nights
- **Windows**: Morning and evening ventilation, increased weekend opening
- **Refrigerator**: Compressor cycles, defrost patterns, higher summer load
- **Motion**: Activity peaks around 8AM and 7PM, reduced nighttime activity
- **Doors**: Regular entry/exit times (7:30AM, 12PM, 6PM)

### Anomaly Types

Each house experiences 6 different anomaly types at randomized times:

| ID | Anomaly Type | Description | Impact |
|----|--------------|-------------|--------|
| 1 | **AC Failure Heatwave** | Air conditioning system fails during hot day | Temperature spikes to 38°C+ |
| 2 | **Fridge Summer Breakdown** | Refrigerator fails during peak heat | Power drops to 5W, food safety risk |
| 3 | **Nighttime Intrusion** | Security breach with persistent motion | Motion detected 2-5 hours at night && door or window open at the midnight |
| 4 | **Door Left Open** | Front door left open during day | Energy waste, security risk |
| 5 | **Window Stuck by storm** | Window cannot close during weather event | window open and Temperature drops|
| 6 | **Fridge Door Left Open** | Family member forgets to close fridge | Power consumption increases 2.5x |

### Anomaly Sensor Patterns

| Type | Temperature | Window | Power | Motion | Door |
|------|-------------|--------|-------|--------|------|
| 1 | 38-43°C plateau | Often 1 | Normal cycles | Normal | Normal |
| 2 | Baseline | Normal | 3-8W floor | Normal | Normal |
| 3 | Baseline | 0↔1 (xor) | Normal | Stuck 1 | 1↔0 (xor) |
| 4 | +2-5°C drift | Normal | Normal | Mostly 0 | Stuck 1 |
| 5 | -5°C dip | Stuck 1 | Normal | Normal | Normal |
| 6 | Slight + | Normal | 2.5× spike | Normal | Normal |

### Sample Anomaly Rows

| Type | Timestamp | Temp °C | Window | Power W | Motion | Door |
|------|-----------|---------|--------|---------|--------|------|
| 1 | 2024-12-02 14:20 | 40.91 | 0 | 143.48 | 0 | 0 |
| 1 | 2024-12-02 14:25 | 40.22 | 0 | 142.66 | 0 | 0 |
| 1 | 2024-12-02 14:30 | 39.00 | 1 | 140.44 | 0 | 0 |
| 2 | 2024-12-01 08:45 | 26.23 | 1 | 4.91 | 0 | 0 |
| 2 | 2024-12-01 08:50 | 26.15 | 1 | 4.97 | 0 | 0 |
| 2 | 2024-12-01 08:50 | 24.63 | 1 | 4.74 | 1 | 0 |
| 3 | 2024-12-01 00:10 | 33.00 | 0 | 151.60 | 1 | 1 |
| 3 | 2024-12-01 00:15 | 33.00 | 1 | 146.94 | 1 | 0 |
| 3 | 2024-12-01 00:20 | 31.71 | 0 | 162.58 | 1 | 1 |
| 4 | 2024-12-02 09:00 | 26.54 | 0 | 171.35 | 0 | 1 |
| 4 | 2024-12-02 09:05 | 27.62 | 0 | 177.82 | 0 | 1 |
| 4 | 2024-12-02 09:10 | 28.31 | 0 | 175.74 | 0 | 1 |
| 5 | 2024-12-02 18:25 | 22.14 | 1 | 147.60 | 0 | 0 |
| 5 | 2024-12-02 18:30 | 21.98 | 1 | 155.11 | 0 | 0 |
| 5 | 2024-12-02 18:35 | 25.38 | 1 | 154.14 | 1 | 0 |
| 6 | 2024-12-04 00:20 | 22.71 | 1 | 407.84 | 0 | 0 |
| 6 | 2024-12-04 00:25 | 23.08 | 0 | 415.18 | 0 | 0 |
| 6 | 2024-12-04 00:30 | 23.29 | 0 | 419.34 | 0 | 0 |

### Randomization

Each house has unique characteristics:
- **Random Seeds**: Each house uses a unique seed (base_seed + house_id)
- **Anomaly Timing**: All 6 anomalies occur at different random times within the 2-week period
- **Duration Variance**: Anomaly durations vary within realistic ranges
- **Intensity Variation**: Sensor values have natural variation and noise

## Anomaly Type Discussion

In the real world, it's impossible to cover all possible anomaly types. I designed these anomalies to be difficult to detect, creating a challenge for anomaly detection models. Each anomaly type involves 1-3 sensors showing abnormal behavior.

**Dataset Limitations:**
- Human behavior patterns change over time
- People sometimes act unpredictably, which is hard to simulate
- Only covers 6 common anomaly types, not all possible scenarios

**Model use in this task**
For model picking here are some key points I would consider:
1. light model: could potentially running on the edge device
2. prefer unsupervised model: because we don't have labeled data for anomaly detection.
3. real-time detect, the ealier the better to trigger alarm

- Isolation forest
    - Unsupervised model, which make sense because in real world, we don't have labeled data for anomaly detection.
    - It's not a time series model, I will enigneer the feature help the model pick up the sensor pattern regards time.

## Quick Start

### Prerequisites
```bash
# Activate virtual environment
source venv/bin/activate

# Install required packages (if not already installed)
pip install -r requirements.txt
```

### Training Models

#### Option 1: Quick Training (3 Optimized Models)
```bash
# Run the automated training script for 3 optimized models
bash src/example_training.sh
```

This will train 3 models with different configurations:
- **estimators_300**: Conservative (300 estimators, high recall)
- **estimators_500**: Balanced (500 estimators, good performance)  
- **estimators_700**: Precision-focused (700 estimators, best F1-score)

#### Option 2: Custom Training
```bash
# Train a custom model with specific parameters
python src/train_isolation_forest.py \
  --n-estimators 500 \
  --max-samples 0.9 \
  --contamination 0.02 \
  --max-features 0.8 \
  --bootstrap \
  --warm-start \
  --model-name my_custom_model
```

### Feature Analysis

Run AUC feature diagnostic to analyze feature quality:
```bash
# Run feature analysis examples
python src/example_auc_diagnostic_usage.py
```

This will generate feature importance reports and visualizations in various output directories.

### Results

- **Models**: Saved in `models/` directory as `.joblib` files
- **Performance Results**: JSON files in `models/results/`
- **Training Logs**: Detailed logs in `models/logs/`
- **Comprehensive Report**: See `models/results/results_v1.md` for detailed analysis

### Model Performance Summary

Based on our optimization tests, the recommended model configurations are:

| Model | Accuracy | F1-Score | Use Case |
|-------|----------|----------|----------|
| **estimators_300** | 94.81% | 72.10% | Maximum recall (99.12%) - catch all anomalies |
| **estimators_500** | 96.23% | 77.97% | Balanced performance |
| **estimators_700** | 96.28% | 78.19% | Best precision (64.83%) |
| **estimators_900** | 96.30% | 78.28% | **Optimal overall performance** |

**Recommendation**: Use `estimators_900` for production - it achieves the best balance of 96.30% accuracy, 78.28% F1-score, and computational efficiency.
