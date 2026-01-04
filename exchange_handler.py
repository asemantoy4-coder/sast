import pandas as pd
import requests
import time
import logging

# تنظیم لاگ‌گذاری
logger = logging.getLogger(__name__)

class DataHandler:
    """کلاس هوشمند مدیریت داده با قابلیت سوئیچ بین صرافی‌ها در صورت مسدودی"""

    @staticmethod
    def fetch_data(symbol: str, timeframe: str = '5m', limit: int = 100) -> pd.DataFrame:
        """تلاش برای دریافت دیتا از بایننس و در صورت خطا، سوئیچ به MEXC"""
        
        # ۱. تلاش برای دریافت از Binance (با استفاده از Mirror ها)
        binance_symbol = symbol.replace("/", "").upper()
        binance_mirrors = [
            "https://api1.binance.com/api/v3/klines",
            "https://data-api.binance.vision/api/v3/klines"
        ]

        for url in binance_mirrors:
            try:
                params = {"symbol": binance_symbol, "interval": timeframe, "limit": limit}
                resp = requests.get(url, params=params, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    df = pd.DataFrame(data, columns=[
                        'Open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
                        'Close_time', 'Quote_vol', 'Trades', 'Taker_base', 'Taker_quote', 'Ignore'
                    ])
                    return DataHandler._format_df(df)
                elif resp.status_code == 451:
                    logger.warning(f"Binance 451 error on {url}. Trying alternatives...")
            except:
                continue

        # ۲. در صورت شکست بایننس، دریافت از MEXC (بدون محدودیت IP)
        logger.info(f"Switching to MEXC for {symbol} due to Binance 451 error.")
        try:
            mexc_symbol = symbol.replace("/", "_").upper() if "/" in symbol else f"{symbol[:-4]}_{symbol[-4:]}"
            url = "https://api.mexc.com/api/v3/klines"
            params = {"symbol": mexc_symbol.replace("_", ""), "interval": timeframe, "limit": limit}
            
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                df = pd.DataFrame(data, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close_time', 'Quote_vol'])
                return DataHandler._format_df(df)
        except Exception as e:
            logger.error(f"MEXC Fetch Error: {e}")

        return pd.DataFrame()

    @staticmethod
    def _format_df(df: pd.DataFrame) -> pd.DataFrame:
        """استانداردسازی اعداد برای تحلیل تکنیکال"""
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df[numeric_cols]

    @staticmethod
    def fetch_ticker(symbol: str):
        """دریافت قیمت لحظه‌ای با اولویت بایننس و جایگزینی MEXC"""
        clean_symbol = symbol.replace("/", "").upper()
        # تلاش برای بایننس
        try:
            resp = requests.get(f"https://api1.binance.com/api/v3/ticker/price?symbol={clean_symbol}", timeout=5)
            if resp.status_code == 200:
                return {'symbol': symbol, 'last': float(resp.json()['price'])}
        except:
            pass
        
        # تلاش برای MEXC (در صورت خطای بایننس)
        try:
            resp = requests.get(f"https://api.mexc.com/api/v3/ticker/price?symbol={clean_symbol}", timeout=5)
            if resp.status_code == 200:
                return {'symbol': symbol, 'last': float(resp.json()['price'])}
        except:
            return None

# متدهای مستقیم
fetch_data = DataHandler.fetch_data
fetch_ticker = DataHandler.fetch_ticker
