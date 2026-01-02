import pandas as pd
import requests
import time

class DataHandler:
    """کلاس مدیریت دریافت داده از صرافی (Binance Public API)"""

    @staticmethod
    def fetch_data(symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """
        دریافت کندل‌های OHLCV از API بیننس (بدون نیاز به کلید خصوصی)
        """
        try:
            # تبدیل نماد به فرمت بیننس (مثال BTC/USDT -> BTCUSDT)
            binance_symbol = symbol.replace("/", "")
            
            url = f"https://api.binance.com/api/v3/klines"
            params = {
                "symbol": binance_symbol,
                "interval": timeframe,
                "limit": limit
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                print(f"⚠️ API Error {response.status_code}: {response.text}")
                return pd.DataFrame()
            
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            # ساخت دیتافریم
            df = pd.DataFrame(data, columns=[
                'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close time', 'Quote asset volume', 'Number of trades',
                'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
            ])
            
            # تبدیل ستون‌های اصلی به عدد
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            
            # حذف ستون‌های اضافه و نگهداری ستون‌های لازم برای utils.py
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            return df
            
        except Exception as e:
            print(f"❌ Critical Error fetching data: {e}")
            return pd.DataFrame()
