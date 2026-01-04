import pandas as pd
import requests
import time
import logging

# تنظیم لاگ‌گذاری
logger = logging.getLogger(__name__)

class DataHandler:
    """کلاس مدیریت دریافت داده با قابلیت دور زدن محدودیت IP (Error 451)"""

    # لیست دامنه‌های جایگزین بایننس برای عبور از فیلتر دیتاسنترها
    BASE_URLS = [
        "https://api1.binance.com",
        "https://api2.binance.com",
        "https://api3.binance.com",
        "https://data-api.binance.vision" # دامنه مخصوص تست و داده‌های عمومی
    ]

    @staticmethod
    def fetch_data(symbol: str, timeframe: str = '5m', limit: int = 100) -> pd.DataFrame:
        binance_symbol = symbol.replace("/", "").upper()
        
        # تست کردن تک‌تک آدرس‌ها تا رسیدن به پاسخ
        for base_url in DataHandler.BASE_URLS:
            try:
                url = f"{base_url}/api/v3/klines"
                params = {"symbol": binance_symbol, "interval": timeframe, "limit": limit}
                
                # استفاده از هدرهای واقعی‌تر برای شبیه‌سازی مرورگر
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/json'
                }
                
                response = requests.get(url, params=params, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    df = pd.DataFrame(data, columns=[
                        'Open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
                        'Close_time', 'Quote_vol', 'Trades', 'Taker_base', 'Taker_quote', 'Ignore'
                    ])
                    
                    # نام ستون‌ها دقیقاً هماهنگ با Utils (حروف بزرگ اول)
                    df['Open'] = pd.to_numeric(df['Open'])
                    df['High'] = pd.to_numeric(df['High'])
                    df['Low'] = pd.to_numeric(df['Low'])
                    df['Close'] = pd.to_numeric(df['Close'])
                    df['Volume'] = pd.to_numeric(df['Volume'])
                    
                    return df[['Open', 'High', 'Low', 'Close', 'Volume']]
                
                elif response.status_code == 451:
                    logger.warning(f"⚠️ Limit 451 on {base_url}, trying next mirror...")
                    continue
                    
            except Exception as e:
                logger.error(f"Connect error to {base_url}: {e}")
                continue
        
        logger.error(f"❌ All Binance mirrors failed for {symbol}")
        return pd.DataFrame()

    @staticmethod
    def fetch_ticker(symbol: str):
        binance_symbol = symbol.replace("/", "").upper()
        for base_url in DataHandler.BASE_URLS:
            try:
                url = f"{base_url}/api/v3/ticker/price"
                params = {"symbol": binance_symbol}
                response = requests.get(url, params=params, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    return {'symbol': symbol, 'last': float(data['price'])}
            except:
                continue
        return None

# متدهای مستقیم برای فراخوانی در main.py
fetch_data = DataHandler.fetch_data
fetch_ticker = DataHandler.fetch_ticker
