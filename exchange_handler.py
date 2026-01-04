import pandas as pd
import requests
import time
import logging

# تنظیم لاگ‌گذاری برای هماهنگی با utils
logger = logging.getLogger(__name__)

class DataHandler:
    """کلاس مدیریت دریافت داده از صرافی (نسخه هماهنگ با استراتژی اسکالپ)"""

    @staticmethod
    def fetch_data(symbol: str, timeframe: str = '5m', limit: int = 100) -> pd.DataFrame:
        """دریافت کندل‌های OHLCV و آماده‌سازی برای تحلیل utils"""
        try:
            # تبدیل BTC/USDT به BTCUSDT
            binance_symbol = symbol.replace("/", "").upper()
            url = f"https://api.binance.com/api/v3/klines"
            params = {"symbol": binance_symbol, "interval": timeframe, "limit": limit}
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            response = requests.get(url, params=params, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                # نام ستون‌ها باید دقیقاً مطابق انتظار utils باشد
                df = pd.DataFrame(data, columns=[
                    'Open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'Close_time', 'Quote_vol', 'Trades', 'Taker_base', 'Taker_quote', 'Ignore'
                ])
                
                # تبدیل به عدد (بسیار مهم برای محاسبات ریاضی)
                numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
                
                # بازگرداندن ستون‌های مورد نیاز utils
                return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            logger.error(f"Binance API Error: Status {response.status_code}")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Exception in fetch_data for {symbol}: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def fetch_ticker(symbol: str):
        """دریافت قیمت لحظه‌ای برای مانیتورینگ تارگت‌ها در main.py"""
        try:
            binance_symbol = symbol.replace("/", "").upper()
            url = f"https://api.binance.com/api/v3/ticker/price"
            params = {"symbol": binance_symbol}
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {'symbol': symbol, 'last': float(data['price'])}
            return None
        except Exception as e:
            logger.error(f"Error in fetch_ticker: {e}")
            return None

# برای هماهنگی با فراخوانی‌های مستقیم در main.py (اختیاری اما توصیه شده)
fetch_data = DataHandler.fetch_data
fetch_ticker = DataHandler.fetch_ticker
