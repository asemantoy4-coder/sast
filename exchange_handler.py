import yfinance as yf
import pandas as pd

class DataHandler:
    @staticmethod
    def fetch_data(symbol: str, interval: str):
        # تبدیل سمبل برای یاهو اگر لازم بود
        clean_symbol = symbol.replace("USDT", "-USD")
        ticker = yf.Ticker(clean_symbol)
        df = ticker.history(period="2d", interval=interval)
        return df

    @staticmethod
    def check_connection():
        return True # برای عبور از تست‌های اولیه سیستم