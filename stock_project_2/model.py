import pandas as pd
import numpy as np
import yfinance as yf

from datetime import timedelta

from xgboost import XGBRegressor
from prophet import Prophet

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_percentage_error
)

# ===================================
# LOAD DATA
# ===================================

def load_stock(symbol):

    ticker = symbol + ".NS"

    df = yf.download(

        ticker,

        period="5y",

        interval="1d",

        auto_adjust=False

    )

    df.reset_index(inplace=True)

    # Fix MultiIndex issue from yfinance

    if isinstance(df.columns, pd.MultiIndex):

        df.columns = df.columns.get_level_values(0)

    keep_cols = [

        "Date",

        "Open",

        "High",

        "Low",

        "Close",

        "Volume"

    ]

    df = df[keep_cols].copy()

    df["Close"] = pd.to_numeric(

        df["Close"],

        errors="coerce"

    )

    df.dropna(inplace=True)

    return df


# ===================================
# FEATURE ENGINEERING
# ===================================

def create_features(df):

    df = df.copy()

    df["lag1"] = df["Close"].shift(1)

    df["lag2"] = df["Close"].shift(2)

    df["lag5"] = df["Close"].shift(5)

    df["roll_mean"] = (

        df["Close"]

        .rolling(10)

        .mean()

    )

    df["roll_std"] = (

        df["Close"]

        .rolling(10)

        .std()

    )

    df["return1"] = (

        df["Close"]

        .pct_change()

    )

    df["ema10"] = (

        df["Close"]

        .ewm(span=10)

        .mean()

    )

    df["dayofweek"] = (

        df["Date"]

        .dt.dayofweek

    )

    df["month"] = (

        df["Date"]

        .dt.month

    )

    df.dropna(inplace=True)

    return df


# ===================================
# FUTURE XGB FORECAST
# ===================================

def future_xgb_forecast(

        model,

        df,

        forecast_days,

        features):

    temp = df.copy()

    preds = []

    for _ in range(forecast_days):

        row = temp.iloc[-1:].copy()

        pred = model.predict(

            row[features]

        )[0]

        preds.append(float(pred))

        new_row = row.copy()

        new_row["Close"] = pred

        new_row["lag1"] = pred

        new_row["lag2"] = row["lag1"].values[0]

        new_row["lag5"] = row["lag2"].values[0]

        temp = pd.concat(

            [

                temp,

                new_row

            ],

            ignore_index=True

        )

    return preds


# ===================================
# MAIN MODEL PIPELINE
# ===================================

def run_hybrid_model(

        symbol,

        forecast_days,

        validation_days):

    df = load_stock(symbol)

    df = create_features(df)

    features = [

        "lag1",

        "lag2",

        "lag5",

        "roll_mean",

        "roll_std",

        "return1",

        "ema10",

        "dayofweek",

        "month"

    ]

    train = df.iloc[:-validation_days]

    test = df.iloc[-validation_days:]

    # -------------------
    # XGBOOST
    # -------------------

    xgb = XGBRegressor(

        n_estimators=300,

        learning_rate=0.05,

        max_depth=4,

        random_state=42

    )

    xgb.fit(

        train[features],

        train["Close"]

    )

    xgb_pred = xgb.predict(

        test[features]

    )

    # -------------------
    # PROPHET
    # -------------------

    prophet_df = df[

        [

            "Date",

            "Close"

        ]

    ].rename(

        columns={

            "Date": "ds",

            "Close": "y"

        }

    )

    prophet = Prophet(

        yearly_seasonality=True,

        weekly_seasonality=True

    )

    prophet.fit(

        prophet_df

    )

    future = prophet.make_future_dataframe(

        periods=forecast_days

    )

    forecast = prophet.predict(

        future

    )

    prophet_validation = np.array(

        forecast["yhat"]

        .tail(validation_days)

    )

    # -------------------
    # METRICS + WEIGHTS
    # -------------------

    actual = np.array(

        test["Close"]

    )

    xgb_rmse = np.sqrt(

        mean_squared_error(

            actual,

            xgb_pred

        )

    )

    prophet_rmse = np.sqrt(

        mean_squared_error(

            actual,

            prophet_validation

        )

    )

    xgb_score = 1 / xgb_rmse

    prophet_score = 1 / prophet_rmse

    total = (

        xgb_score +

        prophet_score

    )

    xgb_weight = (

        xgb_score /

        total

    )

    prophet_weight = (

        prophet_score /

        total

    )

    hybrid_validation = (

        xgb_weight *

        np.array(

            xgb_pred

        )

        +

        prophet_weight *

        prophet_validation

    )

    # -------------------
    # FUTURE FORECASTS
    # -------------------

    future_xgb = future_xgb_forecast(

        xgb,

        df,

        forecast_days,

        features

    )

    future_prophet = np.array(

        forecast["yhat"]

        .tail(forecast_days)

    )

    hybrid_future = (

        xgb_weight *

        np.array(

            future_xgb

        )

        +

        prophet_weight *

        future_prophet

    )

    future_dates = pd.date_range(

        start=

        df["Date"].max()

        + timedelta(days=1),

        periods=forecast_days

    )

    latest_price = float(

        df["Close"]

        .values[-1]

    )

    hybrid_rmse = np.sqrt(

        mean_squared_error(

            actual,

            hybrid_validation

        )

    )

    hybrid_mape = (

        mean_absolute_percentage_error(

            actual,

            hybrid_validation

        ) * 100

    )

    actual_direction = np.sign(

        np.diff(

            actual

        )

    )

    pred_direction = np.sign(

        np.diff(

            hybrid_validation

        )

    )

    direction_acc = (

        np.mean(

            actual_direction == pred_direction

        ) * 100

    )

    return {

        "latest_price":

        latest_price,

        "rmse":

        round(

            hybrid_rmse,

            2

        ),

        "mape":

        round(

            hybrid_mape,

            2

        ),

        "direction_acc":

        round(

            direction_acc,

            2

        ),

        "xgb_weight":

        float(

            xgb_weight

        ),

        "prophet_weight":

        float(

            prophet_weight

        ),

        "historical_dates":

        df["Date"],

        "historical_prices":

        df["Close"],

        "future_dates":

        future_dates,

        "hybrid_forecast":

        hybrid_future,

        "xgb_forecast":

        future_xgb,

        "prophet_forecast":

        future_prophet

    }