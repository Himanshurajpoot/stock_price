"""
Utility functions for Stock Price Predictor
"""

import os
import pickle
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ============ FILE MANAGEMENT ============

def ensure_directory(directory: str) -> None:
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_model(model: Any, filename: str, directory: str = "models") -> str:
    """
    Save trained model to disk using pickle.

    Args:
        model: Model to save
        filename: Name of the file
        directory: Directory to save to

    Returns:
        Full path to saved model
    """
    try:
        ensure_directory(directory)
        filepath = os.path.join(directory, filename)

        with open(filepath, "wb") as f:
            pickle.dump(model, f)

        logger.info(f"Model saved to {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise


def load_model(filepath: str) -> Any:
    """
    Load trained model from disk.

    Args:
        filepath: Path to saved model

    Returns:
        Loaded model
    """
    try:
        if not os.path.exists(filepath):
            logger.warning(f"Model file not found: {filepath}")
            return None

        with open(filepath, "rb") as f:
            model = pickle.load(f)

        logger.info(f"Model loaded from {filepath}")
        return model

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


def save_data_cache(data: pd.DataFrame, symbol: str, directory: str = "cache") -> str:
    """
    Save dataframe to cache with timestamp.

    Args:
        data: Data to cache
        symbol: Stock symbol
        directory: Cache directory

    Returns:
        Path to cached file
    """
    try:
        ensure_directory(directory)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(directory, f"{symbol}_{timestamp}.parquet")

        data.to_parquet(filepath)
        logger.info(f"Data cached to {filepath}")

        return filepath

    except Exception as e:
        logger.error(f"Error caching data: {e}")
        raise


def load_cached_data(symbol: str, max_age_hours: int = 24, directory: str = "cache") -> Tuple[pd.DataFrame, bool]:
    """
    Load cached data if it exists and is fresh.

    Args:
        symbol: Stock symbol
        max_age_hours: Maximum age of cache in hours
        directory: Cache directory

    Returns:
        Tuple of (data, is_valid)
    """
    try:
        if not os.path.exists(directory):
            return None, False

        # Find most recent file for this symbol
        files = sorted(
            [f for f in os.listdir(directory) if f.startswith(symbol)],
            reverse=True
        )

        if not files:
            return None, False

        latest_file = os.path.join(directory, files[0])
        file_time = datetime.fromtimestamp(os.path.getmtime(latest_file))
        age_hours = (datetime.now() - file_time).total_seconds() / 3600

        if age_hours > max_age_hours:
            logger.info(f"Cache expired for {symbol} ({age_hours:.1f} hours old)")
            return None, False

        data = pd.read_parquet(latest_file)
        logger.info(f"Loaded cached data for {symbol} ({age_hours:.1f} hours old)")

        return data, True

    except Exception as e:
        logger.error(f"Error loading cache: {e}")
        return None, False


# ============ FEATURE IMPORTANCE ============

def calculate_feature_importance(model, feature_names: list) -> Dict[str, float]:
    """
    Calculate feature importance from XGBoost model.

    Args:
        model: Trained XGBoost model
        feature_names: Names of features

    Returns:
        Dictionary of feature: importance pairs
    """
    try:
        if not hasattr(model, "feature_importances_"):
            logger.warning("Model does not have feature_importances_ attribute")
            return {}

        importances = model.feature_importances_
        importance_dict = dict(zip(feature_names, importances))

        # Sort by importance
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )

        return importance_dict

    except Exception as e:
        logger.error(f"Error calculating feature importance: {e}")
        return {}


# ============ ANOMALY DETECTION ============

def detect_anomalies(data, threshold: float = 2.5):
    """
    Detect anomalies using Z-score method.

    Args:
        data: 1D array of values
        threshold: Z-score threshold (default: 2.5)

    Returns:
        Boolean array indicating anomalies
    """
    try:
        if isinstance(data, pd.DataFrame):
            return detect_price_anomalies(data, threshold)

        z_scores = np.abs((data - np.mean(data)) / (np.std(data) + 1e-10))
        anomalies = z_scores > threshold

        return anomalies

    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        return np.array([])


def detect_price_anomalies(df: pd.DataFrame, threshold: float = 2.5) -> pd.DataFrame:
    """
    Detect anomalies in price data.

    Args:
        df: DataFrame with price data
        threshold: Z-score threshold

    Returns:
        DataFrame with anomaly flags
    """
    try:
        df = df.copy()

        # Daily returns
        df["daily_return"] = df["Close"].pct_change()

        # Volume change
        if "Volume" in df.columns:
            df["volume_change"] = df["Volume"].pct_change()

        # Detect anomalies in returns
        return_anomalies = detect_anomalies(df["daily_return"].dropna().values, threshold)
        df["is_return_anomaly"] = False
        df.loc[df["daily_return"].notna(), "is_return_anomaly"] = return_anomalies

        # Detect anomalies in volume
        if "volume_change" in df.columns:
            volume_anomalies = detect_anomalies(df["volume_change"].dropna().values, threshold)
            df["is_volume_anomaly"] = False
            df.loc[df["volume_change"].notna(), "is_volume_anomaly"] = volume_anomalies

        return df

    except Exception as e:
        logger.error(f"Error in price anomaly detection: {e}")
        return df


# ============ CORRELATION ANALYSIS ============

def calculate_correlation_matrix(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """
    Calculate correlation matrix for features.

    Args:
        df: DataFrame with data
        columns: Specific columns to correlate (default: all numeric)

    Returns:
        Correlation matrix
    """
    try:
        if columns:
            return df[columns].corr()
        else:
            return df.select_dtypes(include=[np.number]).corr()

    except Exception as e:
        logger.error(f"Error calculating correlation: {e}")
        return pd.DataFrame()


def find_high_correlations(corr_matrix: pd.DataFrame, threshold: float = 0.8) -> list:
    """
    Find feature pairs with high correlation.

    Args:
        corr_matrix: Correlation matrix
        threshold: Correlation threshold

    Returns:
        List of high correlation pairs
    """
    try:
        high_corr = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr.append({
                        "feature1": corr_matrix.columns[i],
                        "feature2": corr_matrix.columns[j],
                        "correlation": corr_matrix.iloc[i, j]
                    })

        return high_corr

    except Exception as e:
        logger.error(f"Error finding high correlations: {e}")
        return []


# ============ PERFORMANCE TRACKING ============

def calculate_rolling_performance(
    actual: np.ndarray,
    predicted: np.ndarray,
    window_size: int = 30
) -> Dict[str, list]:
    """
    Calculate rolling window performance metrics.

    Args:
        actual: Actual values
        predicted: Predicted values
        window_size: Window size for rolling metrics

    Returns:
        Dictionary with rolling metrics
    """
    try:
        from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

        rmse_values = []
        mape_values = []
        dates = []

        for i in range(len(actual) - window_size + 1):
            window_actual = actual[i:i + window_size]
            window_pred = predicted[i:i + window_size]

            rmse = np.sqrt(mean_squared_error(window_actual, window_pred))
            mape = mean_absolute_percentage_error(window_actual, window_pred) * 100

            rmse_values.append(rmse)
            mape_values.append(mape)
            dates.append(i + window_size - 1)

        return {
            "rmse": rmse_values,
            "mape": mape_values,
            "dates": dates
        }

    except Exception as e:
        logger.error(f"Error calculating rolling performance: {e}")
        return {}


# ============ DATA VALIDATION ============

def validate_data(df: pd.DataFrame, min_rows: int = 100) -> Tuple[bool, str]:
    """
    Validate dataframe for model training.

    Args:
        df: DataFrame to validate
        min_rows: Minimum required rows

    Returns:
        Tuple of (is_valid, message)
    """
    try:
        if df is None or df.empty:
            return False, "Data is empty"

        if len(df) < min_rows:
            return False, f"Insufficient data: {len(df)} rows (need {min_rows})"

        if df.isnull().sum().sum() > 0:
            null_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            if null_pct > 10:  # Allow up to 10% nulls
                return False, f"Too many null values: {null_pct:.1f}%"

        return True, "Data validation passed"

    except Exception as e:
        return False, f"Validation error: {str(e)}"


# ============ STATISTICS ============

def calculate_stats(values: np.ndarray) -> Dict[str, float]:
    """
    Calculate summary statistics.

    Args:
        values: Array of values

    Returns:
        Dictionary of statistics
    """
    try:
        return {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
        }

    except Exception as e:
        logger.error(f"Error calculating statistics: {e}")
        return {}
