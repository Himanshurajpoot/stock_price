"""
Hybrid Stock Price Prediction Model (Advanced Version)
Combines XGBoost and Facebook Prophet for accurate NSE stock forecasting.
Features: Caching, Feature Importance, Anomaly Detection, Rolling Performance, Correlations
"""

import logging
import logging.handlers
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
from typing import Dict, List
from xgboost import XGBRegressor
from prophet import Prophet
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import os
from pathlib import Path

# Import config and utilities
import config
from utils import (
    save_model, load_model, load_cached_data, save_data_cache,
    calculate_feature_importance, detect_price_anomalies,
    calculate_correlation_matrix, find_high_correlations,
    calculate_rolling_performance, validate_data, calculate_stats,
    ensure_directory
)

# Configure logging with rotation
def setup_logging():
    """Setup logging with file rotation."""
    ensure_directory("logs")
    
    logger = logging.getLogger(__name__)
    logger.setLevel(config.LOG_LEVEL)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # File handler with rotation
    handler = logging.handlers.RotatingFileHandler(
        config.LOG_FILE,
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT
    )
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


logger = setup_logging()

# Import constants from config
DEFAULT_PERIOD = config.DEFAULT_PERIOD
DEFAULT_INTERVAL = config.DEFAULT_INTERVAL
LAG_FEATURES = config.LAG_FEATURES
ROLLING_WINDOW = config.ROLLING_WINDOW
EMA_SPAN = config.EMA_SPAN
XGB_PARAMS = config.XGB_PARAMS
FEATURE_NAMES = config.FEATURE_NAMES
RSI_PERIOD = config.RSI_PERIOD
MACD_FAST = config.MACD_FAST
MACD_SLOW = config.MACD_SLOW
MACD_SIGNAL = config.MACD_SIGNAL
BOLLINGER_BANDS_PERIOD = config.BOLLINGER_BANDS_PERIOD
BOLLINGER_BANDS_STD = config.BOLLINGER_BANDS_STD
ANOMALY_THRESHOLD = config.ANOMALY_THRESHOLD
ENABLE_ANOMALY_DETECTION = config.ENABLE_ANOMALY_DETECTION
ROLLING_WINDOW_DAYS = config.ROLLING_WINDOW_DAYS


# ===================================
# LOAD DATA WITH CACHING
# ===================================

def load_local_stock_csv(symbol: str) -> pd.DataFrame:
    """Load stock data from a bundled CSV file as a fallback when Yahoo Finance fails."""
    symbol_name = symbol.upper().strip().removesuffix(".NS")
    data_dir = Path(__file__).resolve().parent
    candidates = [
        data_dir / f"{symbol_name}.csv",
        data_dir / f"{symbol_name}.CSV",
    ]

    candidates.extend(
        path for path in data_dir.iterdir()
        if path.is_file() and path.name.upper() == f"{symbol_name}.CSV"
    )

    for candidate in candidates:
        if candidate.exists():
            try:
                df = pd.read_csv(candidate)
                required = ["Date", "Open", "High", "Low", "Close", "Volume"]
                if not set(required).issubset(df.columns):
                    logger.warning(f"CSV file {candidate} does not contain required columns")
                    continue

                df = df[required].copy()
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
                df["High"] = pd.to_numeric(df["High"], errors="coerce")
                df["Low"] = pd.to_numeric(df["Low"], errors="coerce")
                df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
                df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
                df = df.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"]).reset_index(drop=True)

                if not df.empty:
                    logger.info(f"Loaded fallback CSV data for {symbol} from {candidate}")
                    return df
            except Exception as exc:
                logger.warning(f"Failed reading fallback CSV {candidate}: {exc}")

    return pd.DataFrame()


def load_stock(symbol: str, use_cache: bool = True) -> pd.DataFrame:
    """Load historical stock data from Yahoo Finance with CSV fallback support."""
    try:
        symbol = symbol.upper().strip().removesuffix(".NS")

        # Try to load from cache first
        if use_cache:
            cached_data, is_valid = load_cached_data(
                symbol,
                max_age_hours=config.CACHE_DURATION_HOURS
            )
            if is_valid and cached_data is not None:
                logger.info(f"Using cached data for {symbol}")
                return cached_data

        ticker = f"{symbol}.NS"
        logger.info(f"Downloading stock data for {symbol}...")

        try:
            df = yf.download(
                ticker,
                period=DEFAULT_PERIOD,
                interval=DEFAULT_INTERVAL,
                auto_adjust=False,
                progress=False
            )
        except Exception:
            df = pd.DataFrame()

        if df.empty:
            logger.warning(f"Yahoo Finance returned no data for {symbol}; trying local CSV fallback")
            fallback_df = load_local_stock_csv(symbol)
            if not fallback_df.empty:
                try:
                    save_data_cache(fallback_df, symbol)
                except Exception as e:
                    logger.warning(f"Could not cache fallback data: {e}")
                return fallback_df
            raise ValueError(f"No data found for symbol {symbol}. Try a valid ticker or add a matching CSV file.")

        df.reset_index(inplace=True)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        keep_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        df = df[keep_cols].copy()

        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df.dropna(inplace=True)

        is_valid, message = validate_data(df, min_rows=100)
        if not is_valid:
            raise ValueError(message)

        logger.info(f"Loaded {len(df)} records for {symbol}")

        # Save to cache
        try:
            save_data_cache(df, symbol)
        except Exception as e:
            logger.warning(f"Could not cache data: {e}")

        return df

    except Exception as e:
        logger.error(f"Error loading stock data: {e}")
        raise


# ===================================
# FEATURE ENGINEERING
# ===================================

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for machine learning model."""
    try:
        df = df.copy()
        
        for lag in LAG_FEATURES:
            df[f"lag{lag}"] = df["Close"].shift(lag)
        
        df["roll_mean"] = df["Close"].rolling(ROLLING_WINDOW).mean()
        df["roll_std"] = df["Close"].rolling(ROLLING_WINDOW).std()
        
        df["return1"] = df["Close"].pct_change()
        df["ema10"] = df["Close"].ewm(span=EMA_SPAN).mean()
        
        df["dayofweek"] = df["Date"].dt.dayofweek
        df["month"] = df["Date"].dt.month
        
        df.dropna(inplace=True)
        
        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        return df
    
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise


# ===================================
# TECHNICAL INDICATORS
# ===================================

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    ema_fast = data.ewm(span=fast).mean()
    ema_slow = data.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram


def calculate_bollinger_bands(
    data: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple:
    """Calculate Bollinger Bands."""
    middle = data.rolling(window=period, min_periods=1).mean()
    std = data.rolling(window=period, min_periods=1).std().fillna(0)
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return middle, upper, lower


def calculate_confidence_intervals(
    predictions: np.ndarray, residuals: np.ndarray, confidence: float = 0.95
) -> tuple:
    """Calculate prediction confidence intervals using t-distribution."""
    std_error = np.std(residuals)
    df = len(residuals) - 1
    critical_value = stats.t.ppf((1 + confidence) / 2, df)
    margin = critical_value * std_error
    lower = predictions - margin
    upper = predictions + margin
    return lower, upper


# ===================================
# FUTURE XGB FORECAST
# ===================================

def future_xgb_forecast(
    model: XGBRegressor,
    df: pd.DataFrame,
    forecast_days: int,
    features: List[str]
) -> List[float]:
    """Generate future predictions using XGBoost model."""
    try:
        temp = df[["Date", "Close", "Volume"]].copy()
        preds = []
        
        for _ in range(forecast_days):
            feature_frame = create_features(temp)
            row = feature_frame.iloc[-1:]
            pred = model.predict(row[features])[0]
            preds.append(float(pred))
            
            next_date = temp["Date"].iloc[-1] + pd.offsets.BDay(1)
            temp = pd.concat(
                [
                    temp,
                    pd.DataFrame(
                        {
                            "Date": [next_date],
                            "Close": [pred],
                            "Volume": [temp["Volume"].iloc[-1]],
                        }
                    ),
                ],
                ignore_index=True,
            )
        
        logger.info(f"XGBoost forecast generated for {forecast_days} days")
        return preds
    
    except Exception as e:
        logger.error(f"Error in XGBoost forecasting: {e}")
        raise


# ===================================
# BACKTESTING
# ===================================

def run_backtest(
    df: pd.DataFrame,
    model_preds: np.ndarray,
    test_indices: np.ndarray,
    initial_capital: float = 100000,
    transaction_cost: float = config.TRANSACTION_COST,
) -> Dict:
    """Backtest predictions with next-session execution and trading costs."""
    try:
        actual = df.iloc[test_indices]["Close"].values

        if len(actual) < 2 or len(model_preds) < 2:
            return {}
        
        position = 0
        capital = initial_capital
        shares = 0
        trade_log = []
        portfolio_value = []
        
        for i in range(len(model_preds) - 1):
            signal_price = actual[i]
            execution_price = actual[i + 1]
            prediction = model_preds[i]
            
            if prediction > signal_price and position == 0 and capital > 0:
                shares = (capital * (1 - transaction_cost)) / execution_price
                capital = 0
                position = 1
                trade_log.append({"action": "BUY", "price": execution_price, "shares": shares})
            
            elif prediction < signal_price and position == 1:
                capital = shares * execution_price * (1 - transaction_cost)
                position = 0
                shares = 0
                trade_log.append({"action": "SELL", "price": execution_price, "capital": capital})
            
            if position == 1:
                current_value = shares * execution_price
            else:
                current_value = capital
            
            portfolio_value.append(current_value)
        
        if position == 1:
            capital = shares * actual[-1] * (1 - transaction_cost)
        
        total_return = ((capital - initial_capital) / initial_capital) * 100
        buy_hold_value = (initial_capital / actual[0]) * actual[-1]
        buy_hold_return = ((buy_hold_value - initial_capital) / initial_capital) * 100
        
        daily_returns = np.diff(portfolio_value) / portfolio_value[:-1]
        sharpe_ratio = np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252)
        
        cumulative = np.array(portfolio_value)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown) * 100
        
        logger.info(f"Backtest completed. Return: {total_return:.2f}%, Sharpe: {sharpe_ratio:.2f}")
        
        return {
            "total_return": round(total_return, 2),
            "buy_hold_return": round(buy_hold_return, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown": round(max_drawdown, 2),
            "final_capital": round(capital, 2),
            "trades": len(trade_log),
            "portfolio_values": portfolio_value,
        }
    
    except Exception as e:
        logger.error(f"Error in backtesting: {e}")
        return {}


# ===================================
# MAIN MODEL PIPELINE
# ===================================

def run_hybrid_model(
    symbol: str,
    forecast_days: int,
    validation_days: int,
    use_cache: bool = True
) -> Dict:
    """Execute hybrid XGBoost + Prophet forecasting pipeline with advanced features."""
    try:
        logger.info(f"Starting hybrid model for {symbol}")
        
        # Load and prepare data with caching
        df = load_stock(symbol, use_cache=use_cache)
        df = create_features(df)
        
        # Calculate technical indicators
        logger.info("Calculating technical indicators...")
        df["rsi"] = calculate_rsi(df["Close"])
        macd, signal, histogram = calculate_macd(df["Close"])
        df["macd"] = macd
        df["macd_signal"] = signal
        df["macd_histogram"] = histogram
        
        middle, upper, lower = calculate_bollinger_bands(df["Close"])
        df["bb_middle"] = middle
        df["bb_upper"] = upper
        df["bb_lower"] = lower
        
        # Detect anomalies
        logger.info("Detecting anomalies...")
        anomaly_df = detect_price_anomalies(df, threshold=ANOMALY_THRESHOLD)
        anomalies = anomaly_df[anomaly_df["is_return_anomaly"]].index.tolist()
        logger.info(f"Found {len(anomalies)} anomalies in price data")
        
        # Train-test split
        train = df.iloc[:-validation_days]
        test = df.iloc[-validation_days:]
        test_indices = range(len(df) - validation_days, len(df))
        
        logger.info(f"Train set: {len(train)}, Test set: {len(test)}")
        
        # ============ XGBOOST ============
        logger.info("Training XGBoost model...")
        
        xgb_model_path = f"models/{symbol}_xgb.pkl"
        xgb = None
        
        if use_cache:
            xgb = load_model(xgb_model_path)
        
        if xgb is None:
            xgb = XGBRegressor(**XGB_PARAMS)
            xgb.fit(train[FEATURE_NAMES], train["Close"])
            save_model(xgb, f"{symbol}_xgb.pkl")
        
        xgb_pred = xgb.predict(test[FEATURE_NAMES])
        
        # Calculate feature importance
        feature_importance = calculate_feature_importance(xgb, FEATURE_NAMES)
        logger.info(f"Top 3 features: {list(feature_importance.items())[:3]}")
        
        # ============ PROPHET ============
        logger.info("Training Prophet model...")

        validation_prophet = Prophet(**config.PROPHET_PARAMS)
        validation_prophet.fit(
            train[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
        )
        validation_future = validation_prophet.make_future_dataframe(
            periods=validation_days,
            freq="B"
        )
        prophet_validation = np.array(
            validation_prophet.predict(validation_future)["yhat"].tail(validation_days)
        )
        
        prophet_model_path = f"models/{symbol}_prophet.pkl"
        prophet = None
        
        if use_cache:
            prophet = load_model(prophet_model_path)
        
        if prophet is None:
            prophet_df = df[["Date", "Close"]].rename(
                columns={"Date": "ds", "Close": "y"}
            )
            
            prophet = Prophet(**config.PROPHET_PARAMS)
            prophet.fit(prophet_df)
            save_model(prophet, f"{symbol}_prophet.pkl")
        
        future = prophet.make_future_dataframe(periods=forecast_days, freq="B")
        forecast = prophet.predict(future)
        
        # ============ METRICS + WEIGHTS ============
        actual = np.array(test["Close"])
        
        xgb_rmse = np.sqrt(mean_squared_error(actual, xgb_pred))
        prophet_rmse = np.sqrt(mean_squared_error(actual, prophet_validation))
        
        xgb_score = 1 / (xgb_rmse + 1e-6)
        prophet_score = 1 / (prophet_rmse + 1e-6)
        
        total = xgb_score + prophet_score
        xgb_weight = xgb_score / total
        prophet_weight = prophet_score / total
        
        hybrid_validation = (
            xgb_weight * np.array(xgb_pred) +
            prophet_weight * prophet_validation
        )
        
        # Calculate confidence intervals
        logger.info("Calculating confidence intervals...")
        residuals = actual - hybrid_validation
        lower_ci, upper_ci = calculate_confidence_intervals(hybrid_validation, residuals)
        
        # Calculate rolling performance
        logger.info("Calculating rolling performance metrics...")
        rolling_perf = calculate_rolling_performance(actual, hybrid_validation, ROLLING_WINDOW_DAYS)
        
        # ============ FUTURE FORECASTS ============
        future_xgb = future_xgb_forecast(xgb, df, forecast_days, FEATURE_NAMES)
        future_prophet = np.array(forecast["yhat"].tail(forecast_days))
        
        hybrid_future = (
            xgb_weight * np.array(future_xgb) +
            prophet_weight * future_prophet
        )
        
        # Confidence intervals for future
        future_residuals = np.std(residuals)
        df_val = len(residuals) - 1
        cv = stats.t.ppf(0.975, df_val)
        future_margin = cv * future_residuals
        
        future_lower = hybrid_future - future_margin
        future_upper = hybrid_future + future_margin
        
        future_dates = pd.bdate_range(
            start=df["Date"].max() + timedelta(days=1),
            periods=forecast_days
        )
        
        latest_price = float(df["Close"].values[-1])
        
        # ============ FINAL METRICS ============
        hybrid_rmse = np.sqrt(mean_squared_error(actual, hybrid_validation))
        hybrid_mape = mean_absolute_percentage_error(actual, hybrid_validation) * 100
        
        actual_direction = np.sign(np.diff(actual))
        pred_direction = np.sign(np.diff(hybrid_validation))
        direction_acc = np.mean(actual_direction == pred_direction) * 100
        
        # ============ BACKTESTING ============
        logger.info("Running backtest...")
        backtest_results = run_backtest(df, hybrid_validation, test_indices)
        
        # ============ CORRELATION ANALYSIS ============
        logger.info("Analyzing correlations...")
        corr_matrix = calculate_correlation_matrix(train[FEATURE_NAMES])
        high_corr = find_high_correlations(corr_matrix, threshold=0.8)
        
        logger.info(
            f"Hybrid RMSE: {hybrid_rmse:.2f}, "
            f"MAPE: {hybrid_mape:.2f}%, "
            f"Direction Accuracy: {direction_acc:.2f}%"
        )
        
        return {
            # Basic metrics
            "latest_price": latest_price,
            "rmse": round(hybrid_rmse, 2),
            "mape": round(hybrid_mape, 2),
            "direction_acc": round(direction_acc, 2),
            "xgb_weight": float(xgb_weight),
            "prophet_weight": float(prophet_weight),
            
            # Historical and forecast data
            "historical_dates": df["Date"],
            "historical_prices": df["Close"],
            "future_dates": future_dates,
            "hybrid_forecast": hybrid_future,
            "xgb_forecast": future_xgb,
            "prophet_forecast": future_prophet,
            "xgb_rmse": round(xgb_rmse, 2),
            "prophet_rmse": round(prophet_rmse, 2),
            
            # Confidence intervals
            "validation_lower_ci": lower_ci,
            "validation_upper_ci": upper_ci,
            "future_lower_ci": future_lower,
            "future_upper_ci": future_upper,
            
            # Technical indicators
            "rsi": df["rsi"].tail(30),
            "macd": df["macd"].tail(30),
            "macd_signal": df["macd_signal"].tail(30),
            "macd_histogram": df["macd_histogram"].tail(30),
            "bb_upper": df["bb_upper"].tail(30),
            "bb_lower": df["bb_lower"].tail(30),
            "bb_middle": df["bb_middle"].tail(30),
            
            # Backtest results
            "backtest": backtest_results,
            
            # Anomaly detection
            "anomalies": anomalies,
            "anomaly_count": len(anomalies),
            
            # Feature importance
            "feature_importance": feature_importance,
            
            # Rolling performance
            "rolling_performance": rolling_perf,
            
            # Correlation analysis
            "correlations": {
                "matrix": corr_matrix.to_dict(),
                "high_correlations": high_corr
            }
        }
    
    except Exception as e:
        logger.error(f"Error in hybrid model: {e}")
        raise
