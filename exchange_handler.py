import pandas as pd
import requests
import time

class DataHandler:
    """کلاس مدیریت دریافت داده از صرافی (نسخه پایدار با User-Agent)"""

    @staticmethod
    def fetch_data(symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """دریافت کندل‌های OHLCV برای تحلیل فنی"""
        try:
            binance_symbol = symbol.replace("/", "")
            url = f"https://api.binance.com/api/v3/klines"
            params = {"symbol": binance_symbol, "interval": timeframe, "limit": limit}
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            response = requests.get(url, params=params, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data, columns=[
                    'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'Close time', 'Quote asset volume', 'Number of trades',
                    'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
                ])
                numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                df[numeric_cols] = df[numeric_cols].astype(float)
                return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ Error in fetch_data: {e}")
            return pd.DataFrame()

    @staticmethod
    def fetch_ticker(symbol: str):
        """دریافت قیمت لحظه‌ای برای اعلام سود (Profit Alert)"""
        try:
            binance_symbol = symbol.replace("/", "")
            url = f"https://api.binance.com/api/v3/ticker/price"
            params = {"symbol": binance_symbol}
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            response = requests.get(url, params=params, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {'symbol': symbol, 'last': float(data['price'])}
            return None
        except Exception as e:
            print(f"❌ Error in fetch_ticker: {e}")
            return None
