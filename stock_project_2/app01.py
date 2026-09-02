import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from model import run_hybrid_model

# --------------------------------
# PAGE CONFIG
# --------------------------------

st.set_page_config(

    page_title="Hybrid Stock Predictor",

    layout="wide"

)

# --------------------------------
# HEADER
# --------------------------------

st.title("📈 Hybrid NSE Stock Predictor")

st.write(

    "Live NSE Data + XGBoost + Prophet Hybrid Forecast"

)

# --------------------------------
# SIDEBAR
# --------------------------------

st.sidebar.header("Controls")

stock = st.sidebar.selectbox(

    "Choose Stock",

    [

        "RELIANCE",

        "ADANIPORTS",

        "TCS",

        "INFY",

        "SBIN",

        "HDFCBANK",

        "ITC"

    ]

)

forecast_days = st.sidebar.slider(

    "Forecast Days",

    min_value=7,

    max_value=60,

    value=30

)

validation_days = st.sidebar.slider(

    "Validation Days",

    min_value=30,

    max_value=180,

    value=120

)

run_button = st.sidebar.button(

    "Run Forecast"

)

# --------------------------------
# DEFAULT MESSAGE
# --------------------------------

if not run_button:

    st.info(

        "Select stock settings from sidebar and press Run Forecast."

    )

# --------------------------------
# RUN MODEL
# --------------------------------

if run_button:

    try:

        with st.spinner(

            "Running Hybrid Model..."

        ):

            results = run_hybrid_model(

                symbol=stock,

                forecast_days=forecast_days,

                validation_days=validation_days

            )

        # ------------------------
        # METRICS
        # ------------------------

        st.subheader("Metrics")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(

            "Latest Price",

            f"₹{results['latest_price']:.2f}"

        )

        c2.metric(

            "RMSE",

            f"{results['rmse']}"

        )

        c3.metric(

            "MAPE",

            f"{results['mape']}%"

        )

        c4.metric(

            "Direction Accuracy",

            f"{results['direction_acc']}%"

        )

        # ------------------------
        # MODEL WEIGHTS
        # ------------------------

        st.subheader(

            "Hybrid Weights"

        )

        st.write(

            f"XGBoost Weight: {results['xgb_weight']:.2%}"

        )

        st.write(

            f"Prophet Weight: {results['prophet_weight']:.2%}"

        )

        # ------------------------
        # CHART
        # ------------------------

        st.subheader(

            "Historical + Forecast"

        )

        fig = go.Figure()

        fig.add_trace(

            go.Scatter(

                x=results["historical_dates"],

                y=results["historical_prices"],

                mode="lines",

                name="Historical"

            )

        )

        fig.add_trace(

            go.Scatter(

                x=results["future_dates"],

                y=results["hybrid_forecast"],

                mode="lines",

                name="Hybrid Forecast"

            )

        )

        fig.add_trace(

            go.Scatter(

                x=results["future_dates"],

                y=results["xgb_forecast"],

                mode="lines",

                name="XGBoost"

            )

        )

        fig.add_trace(

            go.Scatter(

                x=results["future_dates"],

                y=results["prophet_forecast"],

                mode="lines",

                name="Prophet"

            )

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

        # ------------------------
        # FUTURE TABLE
        # ------------------------

        st.subheader(

            "Future Predictions"

        )

        pred_df = pd.DataFrame({

            "Date":

            results["future_dates"],

            "Hybrid Prediction":

            results["hybrid_forecast"]

        })

        st.dataframe(

            pred_df,

            use_container_width=True

        )

    except Exception as e:

        st.error(

            f"Error: {str(e)}"

        )