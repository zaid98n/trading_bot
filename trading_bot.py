from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yfinance as yf
from nsetools import Nse
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import requests

app = FastAPI()

# Request Body Schema
class StockRequest(BaseModel):
    stock_symbol: str

# Data Fetching Functions
def fetch_google_finance_data(stock):
    if not stock.endswith('.NS'):
        stock = stock + '.NS'
    try:
        data = yf.download(tickers=stock, period="1d", interval="1m", auto_adjust=True)
        return data
    except:
        return None

def fetch_nse_data(stock):
    nse = Nse()
    try:
        return nse.get_quote(stock)
    except:
        return None

def fetch_investing_data(stock):
    base_url = "https://api.investing.com/api/search/v2/search"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(f"{base_url}?q={stock}", headers=headers)
        if response.status_code != 200:
            return None
        data = response.json()
        quotes = data.get("quotes", [])
        if quotes:
            first_quote = quotes[0]
            quote_url = first_quote.get("url", None)
            if quote_url:
                return {"url": "https://in.investing.com" + quote_url}
        return None
    except:
        return None

# Analysis Functions
def compute_RSI(prices, period=14):
    delta = prices.diff().dropna()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(window=period).mean()
    roll_down = down.rolling(window=period).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

def perform_technical_analysis(data):
    try:
        close = data['Close']
        short_sma = close.rolling(window=5).mean().iloc[-1]
        long_sma = close.rolling(window=20).mean().iloc[-1]
        rsi = compute_RSI(close, period=14).iloc[-1]
        tech_signal = 1 if short_sma > long_sma and rsi < 70 else -1 if short_sma < long_sma and rsi > 30 else 0
        return tech_signal, short_sma, long_sma, rsi
    except:
        return 0, None, None, None

def perform_fundamental_analysis(stock):
    try:
        ticker = yf.Ticker(stock)
        pe_ratio = ticker.info.get("trailingPE", None)
        signal = 1 if pe_ratio and pe_ratio < 20 else -1 if pe_ratio and pe_ratio > 25 else 0
        return signal, pe_ratio
    except:
        return 0, None

def perform_sentiment_analysis(stock):
    try:
        ticker = yf.Ticker(stock)
        news_items = ticker.news
        analyzer = SentimentIntensityAnalyzer()
        sentiment_scores = [analyzer.polarity_scores(item['title'])['compound'] for item in news_items]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        signal = 1 if avg_sentiment > 0.05 else -1 if avg_sentiment < -0.05 else 0
        return signal, avg_sentiment
    except:
        return 0, 0

def final_recommendation(tech, fundamental, sentiment):
    total = tech + fundamental + sentiment
    if total >= 1:
        return "Buy"
    elif total <= -1:
        return "Sell"
    else:
        return "Hold"

def compute_trade_levels(last_price, recommendation):
    if recommendation == "Buy":
        return last_price, last_price * 1.01, last_price * 0.995
    elif recommendation == "Sell":
        return last_price, last_price * 0.99, last_price * 1.005
    else:
        return last_price, last_price, last_price

# POST API Endpoint
@app.post("/analyze_stock/")
def analyze_stock(request: StockRequest):
    stock = request.stock_symbol.strip().upper()
    google_data = fetch_google_finance_data(stock)
    if google_data is None or google_data.empty:
        raise HTTPException(status_code=404, detail="Failed to fetch stock data.")

    last_price = google_data['Close'].iloc[-1]
    nse_data = fetch_nse_data(stock)
    investing_data = fetch_investing_data(stock)

    tech_signal, short_sma, long_sma, rsi = perform_technical_analysis(google_data)
    fundamental_signal, pe_ratio = perform_fundamental_analysis(stock)
    sentiment_signal, avg_sentiment = perform_sentiment_analysis(stock)

    recommendation = final_recommendation(tech_signal, fundamental_signal, sentiment_signal)
    entry_price, target_price, stop_loss = compute_trade_levels(last_price, recommendation)

    return {
        "stock": stock,
        "last_price": last_price,
        "nse_data": nse_data,
        "investing_url": investing_data,
        "technical_analysis": {
            "short_sma": short_sma,
            "long_sma": long_sma,
            "rsi": rsi,
            "signal": tech_signal
        },
        "fundamental_analysis": {
            "pe_ratio": pe_ratio,
            "signal": fundamental_signal
        },
        "sentiment_analysis": {
            "avg_sentiment": avg_sentiment,
            "signal": sentiment_signal
        },
        "recommendation": recommendation,
        "trade_levels": {
            "entry_price": entry_price,
            "target_price": target_price,
            "stop_loss": stop_loss
        }
    }

if __name__ == "__main__":
    print("App is loaded:", app)
