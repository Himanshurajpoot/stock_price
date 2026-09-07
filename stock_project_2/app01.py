import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import logging
from model import run_hybrid_model
import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants - Import from config
STOCKS = config.SUPPORTED_STOCKS
MIN_FORECAST_DAYS = config.MIN_FORECAST_DAYS
MAX_FORECAST_DAYS = config.MAX_FORECAST_DAYS
DEFAULT_FORECAST_DAYS = config.DEFAULT_FORECAST_DAYS
MIN_VALIDATION_DAYS = config.MIN_VALIDATION_DAYS
MAX_VALIDATION_DAYS = config.MAX_VALIDATION_DAYS
DEFAULT_VALIDATION_DAYS = config.DEFAULT_VALIDATION_DAYS


# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="Hybrid Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============ SIDEBAR CONTROLS ============
st.sidebar.header("⚙️ Controls")

stock = st.sidebar.selectbox(
    "Choose Stock",
    STOCKS,
    help="Select NSE stock ticker"
)

forecast_days = st.sidebar.slider(
    "Forecast Days",
    min_value=MIN_FORECAST_DAYS,
    max_value=MAX_FORECAST_DAYS,
    value=DEFAULT_FORECAST_DAYS,
    help="Number of days to forecast into the future"
)

validation_days = st.sidebar.slider(
    "Validation Days",
    min_value=MIN_VALIDATION_DAYS,
    max_value=MAX_VALIDATION_DAYS,
    value=DEFAULT_VALIDATION_DAYS,
    help="Historical days used for model validation"
)

# Advanced options
with st.sidebar.expander("⚙️ Advanced Options"):
    use_cache = st.checkbox(
        "Use Caching",
        value=True,
        help="Cache models and data for faster loads"
    )
    
    enable_reports = st.checkbox(
        "Enable Report Export",
        value=True,
        help="Generate Excel and text reports"
    )

run_button = st.sidebar.button(
    "🚀 Run Forecast",
    use_container_width=True,
    type="primary"
)


# ============ HEADER ============
st.title("📈 Hybrid NSE Stock Predictor")
st.write(
    "**Live NSE Data** + **XGBoost** + **Prophet** Hybrid Forecast"
)

# Add info box
st.info(
    "Select stock settings from sidebar and press **Run Forecast** to start.",
    icon="ℹ️"
)


# ============ RUN MODEL ============
if run_button:
    try:
        with st.spinner("🔄 Running Hybrid Model..."):
            results = run_hybrid_model(
                symbol=stock,
                forecast_days=forecast_days,
                validation_days=validation_days,
                use_cache=use_cache
            )

        # -------- METRICS --------
        st.subheader("📊 Model Performance Metrics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Latest Price",
                value=f"₹{results['latest_price']:.2f}",
                help="Current stock closing price"
            )

        with col2:
            st.metric(
                label="RMSE",
                value=f"{results['rmse']}",
                help="Root Mean Squared Error (lower is better)"
            )

        with col3:
            st.metric(
                label="MAPE",
                value=f"{results['mape']}%",
                help="Mean Absolute Percentage Error"
            )

        with col4:
            st.metric(
                label="Direction Accuracy",
                value=f"{results['direction_acc']}%",
                help="% of correct price direction predictions"
            )

        # -------- MODEL WEIGHTS --------
        st.subheader("⚖️ Hybrid Model Weights")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="XGBoost Weight",
                value=f"{results['xgb_weight']:.2%}",
                delta=f"RMSE: {results['xgb_rmse']}"
            )

        with col2:
            st.metric(
                label="Prophet Weight",
                value=f"{results['prophet_weight']:.2%}",
                delta=f"RMSE: {results['prophet_rmse']}"
            )

        # -------- INTERACTIVE CHART --------
        st.subheader("📉 Historical + Forecast Chart")

        fig = go.Figure()

        # Historical data
        fig.add_trace(
            go.Scatter(
                x=results["historical_dates"],
                y=results["historical_prices"],
                mode="lines",
                name="Historical Price",
                line=dict(color="blue", width=2),
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Price: ₹%{y:.2f}<extra></extra>"
            )
        )

        # Hybrid forecast
        fig.add_trace(
            go.Scatter(
                x=results["future_dates"],
                y=results["hybrid_forecast"],
                mode="lines",
                name="Hybrid Forecast",
                line=dict(color="green", width=2.5, dash="dash"),
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Forecast: ₹%{y:.2f}<extra></extra>"
            )
        )

        # Confidence interval for future (shaded area)
        fig.add_trace(
            go.Scatter(
                x=results["future_dates"],
                y=results["future_upper_ci"],
                fill=None,
                mode="lines",
                line_color="rgba(0,255,0,0)",
                showlegend=False,
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Upper CI: ₹%{y:.2f}<extra></extra>"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=results["future_dates"],
                y=results["future_lower_ci"],
                fill="tonexty",
                mode="lines",
                line_color="rgba(0,255,0,0)",
                name="95% Confidence Interval",
                fillcolor="rgba(0,255,0,0.2)",
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Lower CI: ₹%{y:.2f}<extra></extra>"
            )
        )

        # XGBoost forecast
        fig.add_trace(
            go.Scatter(
                x=results["future_dates"],
                y=results["xgb_forecast"],
                mode="lines",
                name="XGBoost Only",
                line=dict(color="orange", width=1.5, dash="dot"),
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>XGBoost: ₹%{y:.2f}<extra></extra>"
            )
        )

        # Prophet forecast
        fig.add_trace(
            go.Scatter(
                x=results["future_dates"],
                y=results["prophet_forecast"],
                mode="lines",
                name="Prophet Only",
                line=dict(color="red", width=1.5, dash="dot"),
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Prophet: ₹%{y:.2f}<extra></extra>"
            )
        )

        fig.update_layout(
            title=f"{stock} Price Prediction - {forecast_days} Days",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            hovermode="x unified",
            template="plotly_white",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # -------- TABS FOR DIFFERENT VIEWS --------
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            ["🔮 Predictions", "📊 Technical Indicators", "💹 Backtest Results", 
             "📈 Advanced Metrics", "🔍 Feature Analysis", "⚠️ Anomalies & Correlations"]
        )

        # TAB 1: Future Predictions
        with tab1:
            st.subheader("Future Predictions Table")
            pred_df = pd.DataFrame({
                "Date": results["future_dates"],
                "Hybrid Forecast (₹)": results["hybrid_forecast"].round(2),
                "Lower CI (₹)": results["future_lower_ci"].round(2),
                "Upper CI (₹)": results["future_upper_ci"].round(2),
                "XGBoost (₹)": results["xgb_forecast"],
                "Prophet (₹)": results["prophet_forecast"].round(2),
            })

            st.dataframe(
                pred_df,
                use_container_width=True,
                height=min(400, len(pred_df) * 35 + 40),
                hide_index=True
            )

            csv = pred_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name=f"{stock}_predictions.csv",
                mime="text/csv"
            )

        # TAB 2: Technical Indicators
        with tab2:
            col1, col2 = st.columns(2)

            # RSI Chart
            with col1:
                st.write("**Relative Strength Index (RSI)**")
                rsi_dates = results["historical_dates"].tail(len(results["rsi"]))
                fig_rsi = go.Figure()
                fig_rsi.add_trace(
                    go.Scatter(
                        x=rsi_dates,
                        y=results["rsi"],
                        mode="lines",
                        name="RSI",
                        line=dict(color="purple", width=2)
                    )
                )
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
                fig_rsi.update_layout(height=300, hovermode="x unified", template="plotly_white")
                st.plotly_chart(fig_rsi, use_container_width=True)

            # MACD Chart
            with col2:
                st.write("**MACD (Moving Average Convergence Divergence)**")
                macd_dates = results["historical_dates"].tail(len(results["macd"]))
                fig_macd = go.Figure()
                fig_macd.add_trace(
                    go.Scatter(
                        x=macd_dates,
                        y=results["macd"],
                        mode="lines",
                        name="MACD",
                        line=dict(color="blue", width=2)
                    )
                )
                fig_macd.add_trace(
                    go.Scatter(
                        x=macd_dates,
                        y=results["macd_signal"],
                        mode="lines",
                        name="Signal",
                        line=dict(color="red", width=1)
                    )
                )
                fig_macd.add_trace(
                    go.Bar(
                        x=macd_dates,
                        y=results["macd_histogram"],
                        name="Histogram",
                        marker=dict(color="gray"),
                        opacity=0.3
                    )
                )
                fig_macd.update_layout(height=300, hovermode="x unified", template="plotly_white")
                st.plotly_chart(fig_macd, use_container_width=True)

            # Bollinger Bands Chart
            st.write("**Bollinger Bands**")
            bb_dates = results["historical_dates"].tail(len(results["bb_middle"]))
            fig_bb = go.Figure()
            fig_bb.add_trace(
                go.Scatter(
                    x=bb_dates,
                    y=results["historical_prices"].tail(len(results["bb_middle"])),
                    mode="lines",
                    name="Price",
                    line=dict(color="blue", width=2)
                )
            )
            fig_bb.add_trace(
                go.Scatter(
                    x=bb_dates,
                    y=results["bb_upper"],
                    mode="lines",
                    name="Upper Band",
                    line=dict(color="red", width=1, dash="dash")
                )
            )
            fig_bb.add_trace(
                go.Scatter(
                    x=bb_dates,
                    y=results["bb_middle"],
                    mode="lines",
                    name="Middle Band (SMA)",
                    line=dict(color="gray", width=1)
                )
            )
            fig_bb.add_trace(
                go.Scatter(
                    x=bb_dates,
                    y=results["bb_lower"],
                    mode="lines",
                    name="Lower Band",
                    line=dict(color="green", width=1, dash="dash"),
                    fill="tonexty"
                )
            )
            fig_bb.update_layout(height=400, hovermode="x unified", template="plotly_white")
            st.plotly_chart(fig_bb, use_container_width=True)

        # TAB 3: Backtest Results
        with tab3:
            if results["backtest"]:
                st.write("**Strategy Performance (Based on Hybrid Predictions)**")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Model Return",
                        f"{results['backtest']['total_return']:.2f}%",
                        help="Total return using model predictions"
                    )

                with col2:
                    st.metric(
                        "Buy & Hold Return",
                        f"{results['backtest']['buy_hold_return']:.2f}%",
                        help="Buy and hold strategy return"
                    )

                with col3:
                    st.metric(
                        "Sharpe Ratio",
                        f"{results['backtest']['sharpe_ratio']:.2f}",
                        help="Risk-adjusted return (higher is better)"
                    )

                with col4:
                    st.metric(
                        "Max Drawdown",
                        f"{results['backtest']['max_drawdown']:.2f}%",
                        help="Maximum peak-to-trough decline"
                    )

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Total Trades",
                        results['backtest']['trades'],
                        help="Number of buy/sell signals"
                    )

                with col2:
                    st.metric(
                        "Final Capital",
                        f"₹{results['backtest']['final_capital']:,.2f}",
                        help="Final account value"
                    )

                # Portfolio value chart
                st.write("**Portfolio Value Over Time**")
                backtest_dates = results["historical_dates"].tail(len(results["backtest"]["portfolio_values"]))
                fig_backtest = go.Figure()
                fig_backtest.add_trace(
                    go.Scatter(
                        x=backtest_dates,
                        y=results["backtest"]["portfolio_values"],
                        mode="lines",
                        name="Model Strategy",
                        line=dict(color="green", width=2),
                        fill="tozeroy"
                    )
                )
                fig_backtest.update_layout(
                    title="Portfolio Value Growth",
                    xaxis_title="Date",
                    yaxis_title="Value (₹)",
                    height=400,
                    template="plotly_white"
                )
                st.plotly_chart(fig_backtest, use_container_width=True)

                st.info(
                    "⚠️ Backtest results are based on historical data only. "
                    "Past performance does not guarantee future results.",
                    icon="ℹ️"
                )
            else:
                st.warning("Backtest data not available", icon="⚠️")

        # TAB 4: Advanced Metrics
        with tab4:
            st.write("**Model Validation Confidence Intervals**")
            
            val_df = pd.DataFrame({
                "Index": range(len(results["validation_lower_ci"])),
                "Lower CI (₹)": results["validation_lower_ci"].round(2),
                "Upper CI (₹)": results["validation_upper_ci"].round(2),
                "Range (₹)": (results["validation_upper_ci"] - results["validation_lower_ci"]).round(2),
            })

            st.dataframe(val_df, use_container_width=True, height=200, hide_index=False)

            st.write("**Indicator Summary Statistics**")

            stats_data = {
                "Metric": ["RSI Mean", "RSI Std Dev", "MACD Mean", "MACD Signal Mean"],
                "Value": [
                    f"{results['rsi'].mean():.2f}",
                    f"{results['rsi'].std():.2f}",
                    f"{results['macd'].mean():.4f}",
                    f"{results['macd_signal'].mean():.4f}",
                ]
            }
            st.dataframe(
                pd.DataFrame(stats_data),
                use_container_width=True,
                hide_index=True
            )

            # RSI Interpretation
            st.write("**Technical Analysis Insights**")
            
            latest_rsi = results["rsi"].iloc[-1]
            if latest_rsi > 70:
                st.warning(f"🔴 RSI is {latest_rsi:.2f} - Stock appears **overbought**")
            elif latest_rsi < 30:
                st.success(f"🟢 RSI is {latest_rsi:.2f} - Stock appears **oversold**")
            else:
                st.info(f"🟡 RSI is {latest_rsi:.2f} - Stock is in **neutral** territory")

            # MACD Interpretation
            latest_macd = results["macd"].iloc[-1]
            latest_signal = results["macd_signal"].iloc[-1]
            if latest_macd > latest_signal:
                st.success("🟢 MACD is above signal line - **Bullish** signal")
            else:
                st.error("🔴 MACD is below signal line - **Bearish** signal")

        # TAB 5: Feature Importance and Analysis
        with tab5:
            st.write("**Feature Importance**")

            if results.get("feature_importance"):
                # Display feature importance as bar chart
                importance_dict = results["feature_importance"]
                importance_df = pd.DataFrame(
                    list(importance_dict.items()),
                    columns=["Feature", "Importance"]
                ).sort_values("Importance", ascending=False)

                fig_importance = go.Figure(
                    data=[
                        go.Bar(
                            x=importance_df["Importance"],
                            y=importance_df["Feature"],
                            orientation="h",
                            marker=dict(color="steelblue"),
                        )
                    ]
                )
                fig_importance.update_layout(
                    title="XGBoost Feature Importance",
                    xaxis_title="Importance Score",
                    yaxis_title="Feature",
                    height=400,
                    template="plotly_white"
                )
                st.plotly_chart(fig_importance, use_container_width=True)

                st.write("**Top 5 Important Features:**")
                top_5 = importance_df.head(5)
                st.dataframe(top_5, use_container_width=True, hide_index=True)

            # Rolling Performance
            if results.get("rolling_performance"):
                st.write("**Rolling Performance Metrics**")
                rolling_df = pd.DataFrame({
                    "Window": results["rolling_performance"]["dates"],
                    "RMSE": results["rolling_performance"]["rmse"],
                    "MAPE": results["rolling_performance"]["mape"]
                })

                fig_rolling = go.Figure()
                fig_rolling.add_trace(
                    go.Scatter(
                        x=rolling_df["Window"],
                        y=rolling_df["RMSE"],
                        mode="lines",
                        name="RMSE",
                        line=dict(color="red")
                    )
                )
                fig_rolling.update_layout(
                    title="Rolling Window RMSE (30 days)",
                    xaxis_title="Window",
                    yaxis_title="RMSE",
                    height=300,
                    template="plotly_white"
                )
                st.plotly_chart(fig_rolling, use_container_width=True)

        # TAB 6: Anomalies and Correlations
        with tab6:
            col1, col2 = st.columns(2)

            # Anomalies section
            with col1:
                st.write("**Anomaly Detection**")
                st.metric(
                    "Anomalies Detected",
                    results.get("anomaly_count", 0),
                    help="Number of unusual price movements detected"
                )

                if results.get("anomalies"):
                    st.warning(
                        f"Found {len(results['anomalies'])} anomalies in historical data. "
                        "These may indicate volatility or market events.",
                        icon="⚠️"
                    )

            # Correlations section
            with col2:
                st.write("**High Correlations**")
                if results.get("correlations", {}).get("high_correlations"):
                    corr_list = results["correlations"]["high_correlations"]
                    st.info(f"Found {len(corr_list)} feature pairs with |correlation| > 0.8", icon="ℹ️")

                    for corr in corr_list[:5]:  # Show top 5
                        st.write(
                            f"• **{corr['feature1']}** ↔ **{corr['feature2']}**: "
                            f"{corr['correlation']:.3f}"
                        )
                else:
                    st.success("No high correlations found (all < 0.8)", icon="✅")

            # Correlation Heatmap
            st.write("**Correlation Matrix Heatmap**")
            if results.get("correlations", {}).get("matrix"):
                corr_matrix = pd.DataFrame(results["correlations"]["matrix"])

                fig_corr = go.Figure(
                    data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale="RdBu",
                        zmid=0,
                        zmin=-1,
                        zmax=1
                    )
                )
                fig_corr.update_layout(
                    title="Feature Correlation Matrix",
                    height=500,
                    width=600
                )
                st.plotly_chart(fig_corr, use_container_width=True)

            # Export report
            if enable_reports:
                st.write("**Export Results**")
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("📊 Generate Excel Report"):
                        try:
                            from reporting import generate_prediction_report
                            report_path = generate_prediction_report(results, stock)
                            st.success(f"Report generated: {report_path}", icon="✅")
                        except Exception as e:
                            st.error(f"Error generating report: {e}", icon="❌")

                with col2:
                    if st.button("📋 Generate Text Summary"):
                        try:
                            from reporting import generate_text_summary
                            summary = generate_text_summary(results, stock)
                            st.code(summary, language="text")
                        except Exception as e:
                            st.error(f"Error generating summary: {e}", icon="❌")

        # Success message
        st.success(
            f"✅ Forecast completed successfully for {stock}!",
            icon="✅"
        )

    except ValueError as e:
        st.error(
            f"⚠️ Input Error: {str(e)}",
            icon="❌"
        )
        logger.error(f"Input validation error: {e}")

    except Exception as e:
        st.error(
            f"❌ Error running model: {str(e)}",
            icon="🚨"
        )
        logger.error(f"Model execution error: {e}")

        # Show debugging info in expander
        with st.expander("📋 Error Details"):
            st.code(str(e), language="text")
