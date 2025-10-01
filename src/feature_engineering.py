#!/usr/bin/env python3
"""
Enhanced feature engineering for Smart Home IoT anomaly detection.

This module provides comprehensive feature engineering specifically designed
for detecting various types of anomalies in smart home IoT sensor data.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np
import pandas as pd


# Raw columns
COL_TS = "timestamp"
COL_TEMP = "temperature_living_room"
COL_WINDOW = "window_open"
COL_FRIDGE = "power_consumption_fridge"
COL_MOTION = "motion_detected_hallway"
COL_DOOR = "door_state_front"


def safe_rolling_stat(series: pd.Series, window: int, stat: str, min_periods: int = 1) -> pd.Series:
    """Safe rolling statistics with proper handling of edge cases."""
    if stat == "mean":
        return series.rolling(window, min_periods=min_periods).mean()
    elif stat == "std":
        return series.rolling(window, min_periods=min_periods).std()
    elif stat == "median":
        return series.rolling(window, min_periods=min_periods).median()
    elif stat == "min":
        return series.rolling(window, min_periods=min_periods).min()
    elif stat == "max":
        return series.rolling(window, min_periods=min_periods).max()
    elif stat == "quantile_25":
        return series.rolling(window, min_periods=min_periods).quantile(0.25)
    elif stat == "quantile_75":
        return series.rolling(window, min_periods=min_periods).quantile(0.75)
    else:
        raise ValueError(f"Unsupported rolling stat: {stat}")


def compute_streak(values: pd.Series) -> pd.Series:
    """Compute consecutive streaks of 1s."""
    array = values.to_numpy(dtype=int, copy=False)
    streak = np.zeros(len(array), dtype=int)
    count = 0
    for idx, val in enumerate(array):
        if val:
            count += 1
        else:
            count = 0
        streak[idx] = count
    return pd.Series(streak, index=values.index)


def compute_time_since_last(values: pd.Series) -> pd.Series:
    """Compute time steps since last occurrence of 1."""
    array = values.to_numpy(dtype=int, copy=False)
    time_since = np.zeros(len(array), dtype=int)
    last_idx = -1
    
    for idx, val in enumerate(array):
        if val:
            last_idx = idx
            time_since[idx] = 0
        else:
            if last_idx >= 0:
                time_since[idx] = idx - last_idx
            else:
                time_since[idx] = idx + 1
    return pd.Series(time_since, index=values.index)


def add_enhanced_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive time-based features."""
    df = df.copy()
    
    if COL_TS not in df.columns:
        return df
    
    dt = pd.to_datetime(df[COL_TS])
    
    # Basic time features
    df["hour"] = dt.dt.hour.astype(np.int16)
    df["day_of_week"] = dt.dt.dayofweek.astype(np.int8)
    df["month"] = dt.dt.month.astype(np.int8)
    df["day_of_year"] = dt.dt.dayofyear.astype(np.int16)
    
    # Cyclical time features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)
    
    # Contextual time features
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(np.int8)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] < 6)).astype(np.int8)
    df["is_business_hours"] = ((df["hour"] >= 9) & (df["hour"] < 17) & (df["day_of_week"] < 5)).astype(np.int8)
    df["is_peak_hours"] = ((df["hour"] >= 17) & (df["hour"] < 21)).astype(np.int8)
    df["is_early_morning"] = ((df["hour"] >= 6) & (df["hour"] < 9)).astype(np.int8)
    df["is_late_evening"] = ((df["hour"] >= 21) & (df["hour"] < 24)).astype(np.int8)
    
    return df


def add_enhanced_temperature_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive temperature features focusing on AC failure detection."""
    df = df.copy()
    
    if COL_TEMP not in df.columns:
        return df
    
    # Basic window sizes for different temporal patterns  
    windows_basic = [6, 12, 24, 72]  # 30min, 1h, 2h, 6h
    
    grouped = df.groupby("house_id", group_keys=False) if "house_id" in df.columns else None
    
    for window in windows_basic:
        minutes = window * 5
        
        if grouped is not None:
            df[f"temp_mean_{minutes}m"] = grouped[COL_TEMP].transform(
                lambda s: safe_rolling_stat(s, window, "mean")
            )
            df[f"temp_std_{minutes}m"] = grouped[COL_TEMP].transform(
                lambda s: safe_rolling_stat(s, window, "std", min_periods=2)
            )
            df[f"temp_max_{minutes}m"] = grouped[COL_TEMP].transform(
                lambda s: safe_rolling_stat(s, window, "max")
            )
            df[f"temp_min_{minutes}m"] = grouped[COL_TEMP].transform(
                lambda s: safe_rolling_stat(s, window, "min")
            )
        else:
            df[f"temp_mean_{minutes}m"] = safe_rolling_stat(df[COL_TEMP], window, "mean")
            df[f"temp_std_{minutes}m"] = safe_rolling_stat(df[COL_TEMP], window, "std", min_periods=2)
            df[f"temp_max_{minutes}m"] = safe_rolling_stat(df[COL_TEMP], window, "max")
            df[f"temp_min_{minutes}m"] = safe_rolling_stat(df[COL_TEMP], window, "min")
        
        # Temperature differences and ranges
        df[f"temp_diff_now_vs_mean_{minutes}m"] = df[COL_TEMP] - df[f"temp_mean_{minutes}m"]
        df[f"temp_range_{minutes}m"] = df[f"temp_max_{minutes}m"] - df[f"temp_min_{minutes}m"]
        
        # Z-scores
        temp_std = df[f"temp_std_{minutes}m"].replace(0, np.nan)
        df[f"temp_zscore_{minutes}m"] = (df[f"temp_diff_now_vs_mean_{minutes}m"] / temp_std).fillna(0.0)
    
    # Temperature velocity and acceleration
    for lag in [1, 3, 6]:
        if grouped is not None:
            df[f"temp_velocity_{lag*5}m"] = grouped[COL_TEMP].transform(lambda s: s.diff(lag) / lag)
            df[f"temp_acceleration_{lag*5}m"] = grouped[f"temp_velocity_{lag*5}m"].transform(lambda s: s.diff(1))
        else:
            df[f"temp_velocity_{lag*5}m"] = df[COL_TEMP].diff(lag) / lag
            df[f"temp_acceleration_{lag*5}m"] = df[f"temp_velocity_{lag*5}m"].diff(1)
    
    # Temperature anomaly flags
    df["temp_extreme_high"] = (df[COL_TEMP] > 35).astype(np.int8)
    df["temp_very_high"] = (df[COL_TEMP] > 32).astype(np.int8)
    df["temp_high"] = (df[COL_TEMP] > 28).astype(np.int8)
    df["temp_very_low"] = (df[COL_TEMP] < 15).astype(np.int8)
    
    # Temperature stability measures
    for window in [6, 12, 24]:  # 30m, 1h, 2h
        minutes = window * 5
        temp_std_col = f"temp_std_{minutes}m"
        if temp_std_col in df.columns:
            temp_std = df[temp_std_col]
            df[f"temp_stable_{minutes}m"] = (temp_std < 0.5).astype(np.int8)
            df[f"temp_volatile_{minutes}m"] = (temp_std > 2.0).astype(np.int8)
    
    return df


def add_enhanced_fridge_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add enhanced fridge power consumption features."""
    df = df.copy()
    
    if COL_FRIDGE not in df.columns:
        return df
    
    grouped = df.groupby("house_id", group_keys=False) if "house_id" in df.columns else None
    
    # Basic rolling statistics
    for window in [6, 12, 24, 72]:  # 30m, 1h, 2h, 6h
        minutes = window * 5
        
        if grouped is not None:
            df[f"fridge_mean_{minutes}m"] = grouped[COL_FRIDGE].transform(
                lambda s: safe_rolling_stat(s, window, "mean")
            )
            df[f"fridge_std_{minutes}m"] = grouped[COL_FRIDGE].transform(
                lambda s: safe_rolling_stat(s, window, "std", min_periods=2)
            )
            df[f"fridge_median_{minutes}m"] = grouped[COL_FRIDGE].transform(
                lambda s: safe_rolling_stat(s, window, "median")
            )
        else:
            df[f"fridge_mean_{minutes}m"] = safe_rolling_stat(df[COL_FRIDGE], window, "mean")
            df[f"fridge_std_{minutes}m"] = safe_rolling_stat(df[COL_FRIDGE], window, "std", min_periods=2)
            df[f"fridge_median_{minutes}m"] = safe_rolling_stat(df[COL_FRIDGE], window, "median")
    
    # Power consumption ratios
    if "fridge_median_360m" in df.columns:  # 6h baseline
        baseline = df["fridge_median_360m"].replace(0, np.nan)
        df["fridge_power_ratio_6h"] = (df[COL_FRIDGE] / baseline).fillna(1.0)
    
    if "fridge_mean_60m" in df.columns:  # 1h baseline
        baseline = df["fridge_mean_60m"].replace(0, np.nan)
        df["fridge_power_ratio_1h"] = (df[COL_FRIDGE] / baseline).fillna(1.0)
    
    # Coefficient of variation (measure of stability)
    for window in [6, 12, 24]:  # 30m, 1h, 2h
        minutes = window * 5
        mean_col = f"fridge_mean_{minutes}m"
        std_col = f"fridge_std_{minutes}m"
        if mean_col in df.columns and std_col in df.columns:
            df[f"fridge_cv_{minutes}m"] = (df[std_col] / df[mean_col].replace(0, np.nan)).fillna(0.0)
    
    # Power consumption changes
    for lag in [1, 3, 6]:
        if grouped is not None:
            df[f"fridge_diff_{lag*5}m"] = grouped[COL_FRIDGE].transform(lambda s: s.diff(lag))
            df[f"fridge_pct_change_{lag*5}m"] = grouped[COL_FRIDGE].transform(lambda s: s.pct_change(lag))
        else:
            df[f"fridge_diff_{lag*5}m"] = df[COL_FRIDGE].diff(lag)
            df[f"fridge_pct_change_{lag*5}m"] = df[COL_FRIDGE].pct_change(lag)
    
    # Spike detection
    for window in [6, 12]:
        minutes = window * 5
        mean_col = f"fridge_mean_{minutes}m"
        std_col = f"fridge_std_{minutes}m"
        if mean_col in df.columns and std_col in df.columns:
            threshold = df[mean_col] + 2 * df[std_col]
            df[f"fridge_spike_{minutes}m"] = (df[COL_FRIDGE] > threshold).astype(np.int8)
    
    # Power level categories
    df["fridge_power_very_low"] = (df[COL_FRIDGE] < 100).astype(np.int8)
    df["fridge_power_low"] = (df[COL_FRIDGE] < 130).astype(np.int8)
    df["fridge_power_normal"] = ((df[COL_FRIDGE] >= 130) & (df[COL_FRIDGE] < 180)).astype(np.int8)
    df["fridge_power_high"] = (df[COL_FRIDGE] >= 180).astype(np.int8)
    df["fridge_power_very_high"] = (df[COL_FRIDGE] >= 220).astype(np.int8)
    
    return df


def add_enhanced_motion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add enhanced motion detection features."""
    df = df.copy()
    
    if COL_MOTION not in df.columns:
        return df
    
    grouped = df.groupby("house_id", group_keys=False) if "house_id" in df.columns else None
    
    # Motion streaks and time since last motion
    if grouped is not None:
        df["motion_streak"] = grouped[COL_MOTION].transform(compute_streak)
        df["time_since_last_motion"] = grouped[COL_MOTION].transform(compute_time_since_last)
    else:
        df["motion_streak"] = compute_streak(df[COL_MOTION])
        df["time_since_last_motion"] = compute_time_since_last(df[COL_MOTION])
    
    # Motion activity rates over different windows
    for window in [6, 12, 36, 72]:  # 30m, 1h, 2h, 3h, 6h
        minutes = window * 5
        
        if grouped is not None:
            df[f"motion_rate_{minutes}m"] = grouped[COL_MOTION].transform(
                lambda s: safe_rolling_stat(s, window, "mean")
            )
            df[f"motion_sum_{minutes}m"] = grouped[COL_MOTION].transform(
                lambda s: s.rolling(window, min_periods=1).sum()
            )
        else:
            df[f"motion_rate_{minutes}m"] = safe_rolling_stat(df[COL_MOTION], window, "mean")
            df[f"motion_sum_{minutes}m"] = df[COL_MOTION].rolling(window, min_periods=1).sum()
    
    # Contextual motion features
    if "hour" in df.columns or "is_night" in df.columns:
        if "is_night" in df.columns:
            df["night_motion"] = df[COL_MOTION] * df["is_night"]
        elif "hour" in df.columns:
            is_night = ((df["hour"] >= 22) | (df["hour"] < 6)).astype(np.int8)
            df["night_motion"] = df[COL_MOTION] * is_night
    
    return df


def add_enhanced_door_window_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add enhanced door and window features."""
    df = df.copy()
    
    grouped = df.groupby("house_id", group_keys=False) if "house_id" in df.columns else None
    
    # Door features
    if COL_DOOR in df.columns:
        if grouped is not None:
            df["door_open_streak"] = grouped[COL_DOOR].transform(compute_streak)
            df["time_since_door_opened"] = grouped[COL_DOOR].transform(compute_time_since_last)
        else:
            df["door_open_streak"] = compute_streak(df[COL_DOOR])
            df["time_since_door_opened"] = compute_time_since_last(df[COL_DOOR])
        
        for window in [6, 12, 36]:  # 30m, 1h, 3h
            minutes = window * 5
            
            if grouped is not None:
                df[f"door_open_rate_{minutes}m"] = grouped[COL_DOOR].transform(
                    lambda s: safe_rolling_stat(s, window, "mean")
                )
            else:
                df[f"door_open_rate_{minutes}m"] = safe_rolling_stat(df[COL_DOOR], window, "mean")
    
    # Window features
    if COL_WINDOW in df.columns:
        if grouped is not None:
            df["window_open_streak"] = grouped[COL_WINDOW].transform(compute_streak)
            df["time_since_window_opened"] = grouped[COL_WINDOW].transform(compute_time_since_last)
        else:
            df["window_open_streak"] = compute_streak(df[COL_WINDOW])
            df["time_since_window_opened"] = compute_time_since_last(df[COL_WINDOW])
        
        for window in [6, 12, 24]:  # 30m, 1h, 3h
            minutes = window * 5
            
            if grouped is not None:
                df[f"window_open_rate_{minutes}m"] = grouped[COL_WINDOW].transform(
                    lambda s: safe_rolling_stat(s, window, "mean")
                )
            else:
                df[f"window_open_rate_{minutes}m"] = safe_rolling_stat(df[COL_WINDOW], window, "mean")
    
    return df


def add_cross_sensor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-sensor interaction features."""
    df = df.copy()
    
    # Temperature-window interactions
    if COL_TEMP in df.columns and COL_WINDOW in df.columns:
        # Temperature when window is open/closed
        df["temp_when_window_open"] = df[COL_TEMP] * df[COL_WINDOW]
        df["temp_when_window_closed"] = df[COL_TEMP] * (1 - df[COL_WINDOW])
        
        # Temperature differential with window state changes
        grouped = df.groupby("house_id", group_keys=False) if "house_id" in df.columns else [("all", df)]
        
        # Look for temperature changes around window events
        for lag in [1, 3, 6]:
            window_change = grouped[COL_WINDOW].transform(
                lambda s: s.diff(lag).abs()
            ) if "house_id" in df.columns else df[COL_WINDOW].diff(lag).abs()
            
            temp_change = grouped[COL_TEMP].transform(
                lambda s: s.diff(lag).abs()
            ) if "house_id" in df.columns else df[COL_TEMP].diff(lag).abs()
            
            df[f"temp_change_with_window_{lag*5}m"] = temp_change * window_change
    
    # Motion-door/window combinations
    if COL_MOTION in df.columns:
        if COL_DOOR in df.columns:
            df["motion_with_door_open"] = df[COL_MOTION] * df[COL_DOOR]
            df["motion_with_door_closed"] = df[COL_MOTION] * (1 - df[COL_DOOR])
        
        if COL_WINDOW in df.columns:
            df["motion_with_window_open"] = df[COL_MOTION] * df[COL_WINDOW]
    
    # Door-window combinations
    if COL_DOOR in df.columns and COL_WINDOW in df.columns:
        df["door_and_window_open"] = df[COL_DOOR] * df[COL_WINDOW]
        df["door_or_window_open"] = ((df[COL_DOOR] + df[COL_WINDOW]) > 0).astype(np.int8)
    
    # Temperature-power interactions
    if COL_TEMP in df.columns and COL_FRIDGE in df.columns:
        # Power efficiency ratio (high temp, low power might indicate AC issue)
        df["power_temp_ratio"] = df[COL_FRIDGE] / (df[COL_TEMP] + 0.1)  # avoid division by zero
        
        # Temperature vs expected cooling load
        df["temp_power_interaction"] = df[COL_TEMP] * df[COL_FRIDGE] / 1000  # scaled down
    
    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Main function to engineer all features."""
    print("Starting enhanced feature engineering...")
    
    # Make a copy to avoid modifying original
    df_result = df.copy()
    
    # Sort by house and timestamp for proper temporal feature calculation
    if "house_id" in df_result.columns and COL_TS in df_result.columns:
        df_result = df_result.sort_values(["house_id", COL_TS]).reset_index(drop=True)
    elif COL_TS in df_result.columns:
        df_result = df_result.sort_values(COL_TS).reset_index(drop=True)
    
    # Add feature groups
    print("Adding time features...")
    df_result = add_enhanced_time_features(df_result)
    
    print("Adding temperature features...")
    df_result = add_enhanced_temperature_features(df_result)
    
    print("Adding fridge features...")
    df_result = add_enhanced_fridge_features(df_result)
    
    print("Adding motion features...")
    df_result = add_enhanced_motion_features(df_result)
    
    print("Adding door/window features...")
    df_result = add_enhanced_door_window_features(df_result)
    
    print("Adding cross-sensor features...")
    df_result = add_cross_sensor_features(df_result)
    
    # Fill any remaining NaN values
    df_result = df_result.fillna(0.0)
    
    print(f"Feature engineering complete. Total features: {len(df_result.columns)}")
    
    return df_result


if __name__ == "__main__":
    # Simple test
    test_data = {
        COL_TS: pd.date_range("2024-01-01 00:00:00", periods=100, freq="5min"),
        COL_TEMP: np.random.normal(25, 3, 100),
        COL_WINDOW: np.random.choice([0, 1], 100),
        COL_FRIDGE: np.random.normal(150, 20, 100),
        COL_MOTION: np.random.choice([0, 1], 100),
        COL_DOOR: np.random.choice([0, 1], 100),
        "house_id": [1] * 100
    }
    
    test_df = pd.DataFrame(test_data)
    result_df = engineer_all_features(test_df)
    
    print(f"Input shape: {test_df.shape}")
    print(f"Output shape: {result_df.shape}")
    print(f"New features added: {result_df.shape[1] - test_df.shape[1]}")
    print(f"Feature columns: {list(result_df.columns)}")