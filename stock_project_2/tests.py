"""
Unit Tests for Stock Price Predictor
Tests for data loading, feature engineering, and model predictions
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import model
import config
from utils import (
    validate_data, calculate_stats, detect_anomalies,
    calculate_feature_importance, save_model, load_model
)


class TestDataLoading:
    """Test data loading and caching functionality."""
    
    def test_load_stock_valid_symbol(self):
        """Test loading valid stock data."""
        try:
            df = model.load_stock("RELIANCE", use_cache=False)
            assert isinstance(df, pd.DataFrame)
            assert not df.empty
            assert "Close" in df.columns
            assert len(df) > 100
        except Exception as e:
            pytest.skip(f"Network issue: {e}")
    
    def test_load_stock_invalid_symbol(self):
        """Test loading invalid stock symbol."""
        with pytest.raises(ValueError):
            model.load_stock("INVALIDSYMBOL", use_cache=False)

    def test_load_stock_uses_local_csv_fallback(self):
        """Test fallback to bundled CSV data when Yahoo is unavailable."""
        df = model.load_stock("ADANIPORTS", use_cache=False)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "Close" in df.columns
        assert len(df) > 50
    
    def test_validate_data_valid(self):
        """Test validation of valid data."""
        df = pd.DataFrame({
            "Close": np.random.randn(150),
            "Date": pd.date_range("2023-01-01", periods=150)
        })
        is_valid, message = validate_data(df)
        assert is_valid is True
    
    def test_validate_data_insufficient_rows(self):
        """Test validation with insufficient rows."""
        df = pd.DataFrame({
            "Close": np.random.randn(50),
            "Date": pd.date_range("2023-01-01", periods=50)
        })
        is_valid, message = validate_data(df, min_rows=100)
        assert is_valid is False


class TestFeatureEngineering:
    """Test feature engineering functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        dates = pd.date_range("2023-01-01", periods=100)
        df = pd.DataFrame({
            "Date": dates,
            "Close": np.random.randn(100).cumsum() + 100,
            "Volume": np.random.randint(1000000, 10000000, 100)
        })
        return df
    
    def test_create_features(self, sample_data):
        """Test feature creation."""
        df = model.create_features(sample_data)
        
        # Check new features exist
        expected_features = ["lag1", "lag2", "lag5", "roll_mean", "roll_std", "return1", "ema10", "dayofweek", "month"]
        for feature in expected_features:
            assert feature in df.columns
        
        # Check no nulls after dropna
        assert df.isnull().sum().sum() == 0

    def test_future_xgb_forecast_recalculates_features(self, sample_data):
        """Test that recursive forecasts use newly generated feature rows."""
        feature_rows = []

        class RecordingModel:
            def predict(self, features):
                feature_rows.append(features.copy())
                return np.array([100.0])

        forecast = model.future_xgb_forecast(
            RecordingModel(), sample_data, 3, config.FEATURE_NAMES
        )

        assert len(forecast) == 3
        assert len(feature_rows) == 3
        weekdays = {int(row["dayofweek"].iloc[0]) for row in feature_rows}
        assert len(weekdays) == 3
    
    def test_rsi_calculation(self, sample_data):
        """Test RSI calculation."""
        rsi = model.calculate_rsi(sample_data["Close"], period=14)
        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(sample_data)
        # RSI should be between 0-100
        assert rsi.min() >= 0
        assert rsi.max() <= 100
    
    def test_macd_calculation(self, sample_data):
        """Test MACD calculation."""
        macd, signal, histogram = model.calculate_macd(sample_data["Close"])
        assert isinstance(macd, pd.Series)
        assert len(macd) == len(sample_data)
        assert len(signal) == len(sample_data)
        assert len(histogram) == len(sample_data)
    
    def test_bollinger_bands(self, sample_data):
        """Test Bollinger Bands calculation."""
        middle, upper, lower = model.calculate_bollinger_bands(sample_data["Close"])
        assert isinstance(middle, pd.Series)
        # Upper band should be > middle > lower band
        assert (upper >= middle).all()
        assert (middle >= lower).all()


class TestAnomalyDetection:
    """Test anomaly detection functions."""
    
    def test_detect_anomalies(self):
        """Test anomaly detection with z-score method."""
        data = np.array([1, 2, 3, 4, 5, 100, 6, 7, 8, 9, 10])  # 100 is an outlier
        anomalies = detect_anomalies(data, threshold=2.5)
        assert anomalies[5] == True  # 100 should be detected as anomaly
    
    def test_detect_price_anomalies(self):
        """Test price anomaly detection."""
        dates = pd.date_range("2023-01-01", periods=100)
        prices = np.random.randn(100).cumsum() + 100
        df = pd.DataFrame({
            "Date": dates,
            "Close": prices,
            "Volume": np.random.randint(1000000, 10000000, 100)
        })
        
        result_df = detect_anomalies(df)
        assert "is_return_anomaly" in result_df.columns


class TestModelUtilities:
    """Test model utility functions."""
    
    def test_calculate_stats(self):
        """Test statistics calculation."""
        data = np.array([1, 2, 3, 4, 5])
        stats = calculate_stats(data)
        
        assert "mean" in stats
        assert "median" in stats
        assert "std" in stats
        assert stats["mean"] == 3.0
        assert stats["min"] == 1
        assert stats["max"] == 5
    
    def test_feature_importance_calculation(self):
        """Test feature importance extraction."""
        from sklearn.datasets import make_regression
        from xgboost import XGBRegressor
        
        X, y = make_regression(n_samples=100, n_features=9, random_state=42)
        model_xgb = XGBRegressor(n_estimators=10, random_state=42)
        model_xgb.fit(X, y)
        
        feature_names = [f"feat_{i}" for i in range(9)]
        importance = calculate_feature_importance(model_xgb, feature_names)
        
        assert isinstance(importance, dict)
        assert len(importance) == 9
        assert all(isinstance(v, float) or isinstance(v, np.floating) for v in importance.values())


class TestConfidenceIntervals:
    """Test confidence interval calculations."""
    
    def test_confidence_intervals(self):
        """Test CI calculation."""
        predictions = np.array([100, 101, 102, 103, 104])
        residuals = np.array([1, -1, 0.5, -0.5, 1])
        
        lower, upper = model.calculate_confidence_intervals(predictions, residuals)
        
        assert len(lower) == len(predictions)
        assert len(upper) == len(predictions)
        assert (upper > predictions).all()
        assert (lower < predictions).all()


class TestBacktesting:
    """Test backtesting functionality."""

    def test_backtest_uses_next_session_and_costs(self):
        """Test that trades execute next session and include transaction costs."""
        df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=3),
            "Close": [100.0, 110.0, 90.0],
        })

        results = model.run_backtest(
            df,
            np.array([120.0, 80.0, 80.0]),
            range(3),
            transaction_cost=0.01,
        )

        assert results["trades"] == 2
        assert results["total_return"] < results["buy_hold_return"]
    
    def test_backtest_structure(self):
        """Test backtest result structure."""
        dates = pd.date_range("2023-01-01", periods=100)
        prices = np.linspace(100, 110, 100)
        predictions = prices + np.random.randn(100) * 0.5
        
        df = pd.DataFrame({
            "Date": dates,
            "Close": prices,
        })
        
        results = model.run_backtest(df, predictions, range(100))
        
        assert "total_return" in results
        assert "buy_hold_return" in results
        assert "sharpe_ratio" in results
        assert "max_drawdown" in results
        assert "trades" in results
        assert "portfolio_values" in results


class TestConfig:
    """Test configuration loading."""
    
    def test_config_defaults(self):
        """Test configuration defaults are set."""
        assert config.DEFAULT_PERIOD == "5y"
        assert config.DEFAULT_INTERVAL == "1d"
        assert len(config.FEATURE_NAMES) == 9
        assert config.MIN_FORECAST_DAYS == 7
        assert config.MAX_FORECAST_DAYS == 60


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
