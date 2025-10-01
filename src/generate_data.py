"""Synthesize smart home IoT datasets aligned with README specification."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil
from typing import Callable, Dict

import numpy as np
import pandas as pd


BASE_SEED = 2024_12_01
TOTAL_HOUSES = 100
START_DATE = datetime(2024, 12, 1)
DURATION_DAYS = 14
FREQ_MINUTES = 5
PERIODS_PER_DAY = int(24 * 60 / FREQ_MINUTES)
TOTAL_PERIODS = DURATION_DAYS * PERIODS_PER_DAY


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    house_ids: range
    balance: bool


UNSUPERVISED_SPLITS = {
    "train": DatasetSplit("train", range(1, 81), False),  # 80 houses
    "val": DatasetSplit("val", range(81, 91), False),     # 10 houses
}

TEST_HOUSES = range(91, 101)  # 10 houses


def clear_directory(path: Path) -> None:
    """Clear all CSV files in directory, create if doesn't exist."""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    for csv in path.glob("*.csv"):
        csv.unlink()


def generate_house_data(house_id: int, base_seed: int) -> pd.DataFrame:
    """Generate realistic smart home sensor data for one house."""
    rng = np.random.default_rng(base_seed + house_id)
    timestamps = pd.date_range(START_DATE, periods=TOTAL_PERIODS, freq=f"{FREQ_MINUTES}min")
    hour_values = timestamps.hour.to_numpy()
    minute_values = timestamps.minute.to_numpy()
    hours = hour_values + minute_values / 60
    weekdays = timestamps.dayofweek.to_numpy()
    weekend_mask = weekdays >= 5
    idx = np.arange(TOTAL_PERIODS)

    # Temperature: Daily cycles with peak heat during afternoon, cooler nights
    temp_base = rng.uniform(24.0, 28.0)
    temp_amp = rng.uniform(3.0, 6.0)
    temp_phase = rng.uniform(0, 2 * np.pi)
    daily_cycle = np.sin((2 * np.pi / 24) * hours + temp_phase)
    temperature = temp_base + temp_amp * daily_cycle
    temperature += weekend_mask * rng.uniform(0.5, 1.2)
    temperature += rng.normal(0, 0.6, TOTAL_PERIODS)
    temperature = np.clip(temperature, 20.0, 33.0)

    # Windows: Morning and evening ventilation, increased weekend opening
    window_prob = np.full(TOTAL_PERIODS, 0.08)
    morning = (hours >= 6) & (hours < 9)
    evening = (hours >= 18) & (hours < 22)
    midday = (hours >= 12) & (hours < 15)
    window_prob[morning] = 0.55
    window_prob[evening] = 0.45
    window_prob[midday] = 0.25
    window_prob += weekend_mask * 0.12
    window_prob += rng.normal(0, 0.03, TOTAL_PERIODS)
    window_prob = np.clip(window_prob, 0.02, 0.92)
    window_open = rng.binomial(1, window_prob)

    # Refrigerator: Compressor cycles, defrost patterns, higher summer load
    fridge_base = rng.normal(150.0, 5.0)
    daily_load = rng.uniform(10.0, 18.0)
    fridge_cycle = np.sin(2 * np.pi * idx / 36)
    daily_bias = np.sin((2 * np.pi / 24) * (hours - rng.uniform(3, 5)))
    power_consumption = fridge_base + daily_load * np.maximum(daily_bias, 0)
    power_consumption += 8 * fridge_cycle + rng.normal(0, 3.0, TOTAL_PERIODS)
    power_consumption = np.clip(power_consumption, 120.0, 220.0)

    # Motion: Activity peaks around 8AM and 7PM, reduced nighttime activity
    motion_prob = np.full(TOTAL_PERIODS, 0.05)
    morning_activity = (hours >= 6) & (hours < 9)
    lunch_activity = (hours >= 11) & (hours < 14)
    evening_activity = (hours >= 17) & (hours < 22)
    night_quiet = (hours >= 0) & (hours < 5)
    motion_prob[morning_activity] = 0.5
    motion_prob[lunch_activity] = 0.33
    motion_prob[evening_activity] = 0.6
    motion_prob[night_quiet] = 0.02
    motion_prob[weekend_mask & (hours >= 10) & (hours < 23)] += 0.08
    motion_prob = np.clip(motion_prob + rng.normal(0, 0.02, TOTAL_PERIODS), 0.0, 0.95)
    motion_detected = rng.binomial(1, motion_prob)

    # Doors: Regular entry/exit times (7:30AM, 12PM, 6PM)
    door_state = np.zeros(TOTAL_PERIODS, dtype=int)
    door_events_minutes = [7 * 60 + 30, 12 * 60, 18 * 60]
    for day in range(DURATION_DAYS):
        day_offset = day * PERIODS_PER_DAY
        for event_minute in door_events_minutes:
            base_idx = day_offset + int(event_minute / FREQ_MINUTES)
            if base_idx >= TOTAL_PERIODS:
                continue
            jitter = rng.integers(-2, 3)
            duration = rng.integers(1, 4)
            start = np.clip(base_idx + jitter, day_offset, day_offset + PERIODS_PER_DAY - 1)
            end = min(start + duration, day_offset + PERIODS_PER_DAY)
            door_state[start:end] = 1
    # Add some random door openings
    extra_opens = rng.integers(0, 3)
    for _ in range(extra_opens):
        start = rng.integers(0, TOTAL_PERIODS - 3)
        door_state[start : start + rng.integers(1, 4)] = 1

    # Initialize anomaly tracking
    label = np.zeros(TOTAL_PERIODS, dtype=int)
    anomaly_type = np.zeros(TOTAL_PERIODS, dtype=int)

    def schedule_anomaly(duration: int, condition: Callable[[int], bool]) -> slice:
        """Schedule an anomaly without overlapping existing anomalies."""
        for _ in range(400):
            start_idx = int(rng.integers(0, TOTAL_PERIODS - duration))
            if not condition(start_idx):
                continue
            segment = slice(start_idx, start_idx + duration)
            if label[segment].any():
                continue
            return segment
        raise RuntimeError("Unable to schedule anomaly without overlap")

    def ac_failure_heatwave() -> None:
        """Anomaly Type 1: AC Failure Heatwave - Temperature spikes to 38°C+"""
        duration = int(rng.integers(36, 72))
        segment = schedule_anomaly(duration, lambda idx_: 12 <= hours[idx_] < 18)
        temp_values = rng.normal(40.0, 0.8, duration)
        temperature[segment] = np.clip(temp_values, 38.0, 43.0)
        window_open[segment] = np.maximum(window_open[segment], rng.binomial(1, 0.4, duration))
        label[segment] = 1
        anomaly_type[segment] = 1

    def fridge_summer_breakdown() -> None:
        """Anomaly Type 2: Fridge Summer Breakdown - Power drops to 5W"""
        duration = int(rng.integers(48, 96))
        segment = schedule_anomaly(duration, lambda idx_: 8 <= hours[idx_] < 20)
        broken_power = rng.normal(5.0, 0.8, duration)
        power_consumption[segment] = np.clip(broken_power, 3.0, 8.5)
        label[segment] = 1
        anomaly_type[segment] = 2

    def nighttime_intrusion() -> None:
        """Anomaly Type 3: Nighttime Intrusion - Motion detected 2-5 hours at night"""
        duration = int(rng.integers(24, 60))
        segment = schedule_anomaly(duration, lambda idx_: hours[idx_] < 3)
        motion_detected[segment] = 1
        # XOR pattern: door or window open, but not both
        door_vs_window = rng.binomial(1, 0.5, duration)
        door_vs_window[0] = 0
        if door_vs_window.all():
            door_vs_window[rng.integers(0, duration)] = 0
        elif (~door_vs_window.astype(bool)).all():
            index = rng.integers(0, duration)
            if index == 0 and duration > 1:
                index = 1
            door_vs_window[index] = 1
        window_open[segment] = door_vs_window.astype(int)
        door_state[segment] = 1 - window_open[segment]
        label[segment] = 1
        anomaly_type[segment] = 3

    def door_left_open() -> None:
        """Anomaly Type 4: Door Left Open - Front door left open during day"""
        duration = int(rng.integers(18, 42))
        segment = schedule_anomaly(duration, lambda idx_: 9 <= hours[idx_] < 17)
        door_state[segment] = 1
        motion_detected[segment] = np.minimum(motion_detected[segment], rng.binomial(1, 0.2, duration))
        temperature[segment] = np.clip(temperature[segment] + rng.normal(2.5, 0.5, duration), 20.0, 36.0)
        label[segment] = 1
        anomaly_type[segment] = 4

    def window_stuck_storm() -> None:
        """Anomaly Type 5: Window Stuck by Storm - Window cannot close, temperature drops"""
        duration = int(rng.integers(24, 60))
        segment = schedule_anomaly(duration, lambda idx_: 16 <= hours[idx_] < 23)
        window_open[segment] = 1
        temperature[segment] = np.clip(temperature[segment] - rng.normal(5.0, 1.2, duration), 15.0, 30.0)
        label[segment] = 1
        anomaly_type[segment] = 5

    def fridge_door_left_open() -> None:
        """Anomaly Type 6: Fridge Door Left Open - Power consumption increases 2.5x"""
        duration = int(rng.integers(12, 36))
        segment = schedule_anomaly(duration, lambda idx_: True)
        power_consumption[segment] = np.clip(power_consumption[segment] * 2.5 + rng.normal(15, 5, duration), 250.0, 600.0)
        label[segment] = 1
        anomaly_type[segment] = 6

    # Generate all 6 anomaly types for each house
    ac_failure_heatwave()
    fridge_summer_breakdown()
    nighttime_intrusion()
    door_left_open()
    window_stuck_storm()
    fridge_door_left_open()

    # Final bounds enforcement
    temperature = np.clip(temperature, 15.0, 45.0)
    power_consumption = np.clip(power_consumption, 0.0, 650.0)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temperature_living_room": temperature,
            "window_open": window_open.astype(int),
            "power_consumption_fridge": power_consumption,
            "motion_detected_hallway": motion_detected.astype(int),
            "door_state_front": door_state.astype(int),
            "label": label.astype(int),
            "anomaly_type": anomaly_type.astype(int),
            "house_id": np.full(TOTAL_PERIODS, house_id, dtype=int),
        }
    )

    # Round float columns to 2 decimal places
    float_cols = ["temperature_living_room", "power_consumption_fridge"]
    df[float_cols] = df[float_cols].round(2)
    return df


def write_dataset(df: pd.DataFrame, destination: Path, filename: str) -> None:
    """Write dataset to CSV file."""
    destination.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination / filename, index=False)


def main() -> None:
    """Generate smart home IoT datasets for unsupervised anomaly detection."""
    project_root = Path(__file__).resolve().parent.parent  # Go up from src/ to project root
    data_root = project_root / "data"

    # Create directory structure
    (data_root / "unsupervised").mkdir(parents=True, exist_ok=True)
    (data_root / "test").mkdir(parents=True, exist_ok=True)

    # Clear existing data
    for split in UNSUPERVISED_SPLITS.values():
        clear_directory(data_root / "unsupervised" / split.name)
    clear_directory(data_root / "test")

    # Generate data for all houses
    house_frames: Dict[int, pd.DataFrame] = {}
    print("Generating house data...")
    for house_id in range(1, TOTAL_HOUSES + 1):
        house_frames[house_id] = generate_house_data(house_id, BASE_SEED)
        if house_id % 20 == 0:
            print(f"Generated data for {house_id} houses...")

    # Create unsupervised training and validation sets
    print("Creating unsupervised datasets...")
    for split_name, split in UNSUPERVISED_SPLITS.items():
        for house_id in split.house_ids:
            df = house_frames[house_id]
            if split_name == "train":
                # For training: keep only normal data (no anomalies)
                normal_only = df[df["label"] == 0].copy()
                normal_only.loc[:, "label"] = 0
                normal_only.loc[:, "anomaly_type"] = 0
                filename = f"house_{house_id:03d}_smart_home.csv"
                write_dataset(normal_only, data_root / "unsupervised" / split_name, filename)
            else:
                # For validation: keep all data (including anomalies for proper evaluation)
                filename = f"house_{house_id:03d}_smart_home.csv"
                write_dataset(df, data_root / "unsupervised" / split_name, filename)

    # Create test set (with all anomalies for detection evaluation)
    print("Creating test dataset...")
    for house_id in TEST_HOUSES:
        df = house_frames[house_id]
        filename = f"house_{house_id:03d}_smart_home.csv"
        write_dataset(df, data_root / "test", filename)

    # Generate summary statistics
    summary_rows = []
    for house_id, df in house_frames.items():
        summary_rows.append((house_id, int(df["label"].sum())))
    summary = pd.DataFrame(summary_rows, columns=["house_id", "anomaly_points"])
    total_anomalies = int(summary["anomaly_points"].sum())
    
    print(f"\n✅ Dataset Generation Complete!")
    print(f"📊 Generated data for {len(house_frames)} houses")
    print(f"🚨 Total anomaly points: {total_anomalies}")
    print(f"📁 Data structure:")
    print(f"   - Training: {len(UNSUPERVISED_SPLITS['train'].house_ids)} houses (normal data only)")
    print(f"   - Validation: {len(UNSUPERVISED_SPLITS['val'].house_ids)} houses (with anomalies for evaluation)")
    print(f"   - Testing: {len(TEST_HOUSES)} houses (with anomalies)")
    print(f"📂 Data saved to: {data_root}")


if __name__ == "__main__":
    main()