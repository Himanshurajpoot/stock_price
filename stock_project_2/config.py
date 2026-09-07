"""
Configuration and Constants for Stock Price Predictor
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============ DATA SETTINGS ============
DEFAULT_PERIOD = "5y"
DEFAULT_INTERVAL = "1d"
CACHE_DURATION_HOURS = 24  # Cache data for 24 hours
MODEL_CACHE_DIR = "models"  # Directory to store cached models

# ============ MODEL PARAMETERS ============
LAG_FEATURES = [1, 2, 5]
ROLLING_WINDOW = 10
EMA_SPAN = 10

XGB_PARAMS = {
    "n_estimators": int(os.getenv("XGB_N_ESTIMATORS", 300)),
    "learning_rate": float(os.getenv("XGB_LEARNING_RATE", 0.05)),
    "max_depth": int(os.getenv("XGB_MAX_DEPTH", 4)),
    "random_state": 42,
}

PROPHET_PARAMS = {
    "yearly_seasonality": True,
    "weekly_seasonality": True,
    "daily_seasonality": False,
    "interval_width": float(os.getenv("PROPHET_INTERVAL_WIDTH", 0.95)),
}

# ============ FEATURE ENGINEERING ============
FEATURE_NAMES = [
    "lag1", "lag2", "lag5",
    "roll_mean", "roll_std",
    "return1", "ema10",
    "dayofweek", "month"
]

TECHNICAL_INDICATORS = ["rsi", "macd", "macd_signal", "macd_histogram", "bb_upper", "bb_lower", "bb_middle"]

# ============ TECHNICAL INDICATOR PARAMETERS ============
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_BANDS_PERIOD = 20
BOLLINGER_BANDS_STD = 2.0

# ============ ANOMALY DETECTION ============
ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", 2.5))  # Z-score threshold
ENABLE_ANOMALY_DETECTION = os.getenv("ENABLE_ANOMALY_DETECTION", "true").lower() == "true"

# ============ BACKTEST PARAMETERS ============
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", 100000))
TRANSACTION_COST = float(os.getenv("TRANSACTION_COST", 0.001))  # 0.1% per trade

# ============ SUPPORTED STOCKS ============
SUPPORTED_STOCKS = [
    "ADANIPORTS",
    "ASIANPAINT",
    "AXISBANK",
    "HCLTECH",
    "RELIANCE"
]

# ============ UI SETTINGS ============
MIN_FORECAST_DAYS = 7
MAX_FORECAST_DAYS = 60
DEFAULT_FORECAST_DAYS = 30
MIN_VALIDATION_DAYS = 30
MAX_VALIDATION_DAYS = 180
DEFAULT_VALIDATION_DAYS = 120

# ============ DATABASE SETTINGS ============
DATABASE_URL = os.getenv("DATABASE_URL", None)
USE_DATABASE = DATABASE_URL is not None

# ============ LOGGING SETTINGS ============
LOG_FILE = "logs/stock_predictor.log"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5

# ============ API SETTINGS ============
ENABLE_API = os.getenv("ENABLE_API", "false").lower() == "true"
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))

# ============ PERFORMANCE WINDOW ============
ROLLING_WINDOW_DAYS = 30  # Track performance over 30-day windows
