# 📈 Hybrid NSE Stock Predictor

A Streamlit web app that forecasts NSE (National Stock Exchange, India) stock prices using a **hybrid model** combining **XGBoost** and **Facebook Prophet**, with live data pulled from Yahoo Finance.

## ✨ Features
- 📊 **Live stock data** fetched via `yfinance`
- 🔧 **Advanced feature engineering**: lag features, rolling stats, EMA, returns, calendar features
- 🤖 **Dual-model forecasting**: XGBoost (ML) + Prophet (time-series)
- ⚖️ **Automatic hybrid weighting** based on validation RMSE
- 📉 **Interactive charts** (Plotly) comparing historical, hybrid, XGBoost, and Prophet forecasts
- 📋 **Comprehensive metrics**: RMSE, MAPE, and directional accuracy
- 📥 **Export predictions** as CSV
- ✅ **Robust error handling** with detailed logging
- 🚀 **Production-ready code** with type hints and docstrings

## 🔧 Tech Stack
- **Frontend/App:** Streamlit
- **Modeling:** XGBoost, Prophet, scikit-learn
- **Data:** yfinance (NSE data)
- **Visualization:** Plotly
- **Logging:** Python logging module

## 📋 Prerequisites
- Python 3.8+
- pip or conda

## 🚀 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app01.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure
```
├── app01.py               # Streamlit UI with enhanced UX
├── model.py               # Data loading, feature engineering, hybrid model logic
├── requirements.txt       # Python dependencies with pinned versions
├── LICENSE                # MIT License
├── README.md              # This file
└── *.csv                  # Sample stock data files
```

## 🎯 How It Works

1. **Data Loading**: Downloads 5 years of historical NSE stock data
2. **Feature Engineering**: Creates 9 engineered features (lags, rolling stats, EMA, calendar)
3. **Model Training**: 
   - Trains XGBoost on ML features
   - Trains Prophet on time-series data
4. **Hybrid Weighting**: Automatically weights models based on validation RMSE
5. **Forecasting**: Generates predictions using the weighted hybrid model

### Model Performance Metrics
- **RMSE**: Root Mean Squared Error (lower = better)
- **MAPE**: Mean Absolute Percentage Error
- **Direction Accuracy**: Percentage of correct price direction predictions

## 🎮 Usage

1. **Select Stock**: Choose from the bundled stock datasets: RELIANCE, ADANIPORTS, ASIANPAINT, AXISBANK, or HCLTECH
2. **Set Forecast Days**: Choose 7-60 days to forecast (default: 30)
3. **Set Validation Days**: Choose 30-180 days for model validation (default: 120)
4. **Click "Run Forecast"**: Wait for the model to complete
5. **View Results**: 
   - Check performance metrics
   - View interactive forecast chart
   - Download predictions as CSV

## 📊 Supported Stocks
- RELIANCE
- ADANIPORTS
- ASIANPAINT
- AXISBANK
- HCLTECH

(Easily add more by updating the `STOCKS` list in `app01.py`)

## 🔍 Code Quality Improvements
- ✅ Type hints for all functions
- ✅ Comprehensive docstrings
- ✅ Error handling with detailed logging
- ✅ Clean code formatting following PEP 8
- ✅ Constants defined at module level
- ✅ Version-pinned dependencies
- ✅ Input validation
- ✅ Better UX with icons and descriptions

## ⚠️ Disclaimer
This model is for educational purposes only. Stock market predictions are inherently uncertain and past performance does not guarantee future results. Always do your own research and consult a financial advisor before making investment decisions.

## 👨‍💼 Author
**Himanshu Rajpoot**
BCA (AI & Machine Learning), Galgotias University
GitHub: [Shambhavi2306](https://github.com/Shambhavi2306)

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing
Contributions are welcome! Feel free to:
- Report bugs
- Suggest improvements
- Submit pull requests

## 📞 Support
For issues or questions, please open an issue on GitHub.

