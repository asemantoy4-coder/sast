import pandas as pd
import requests
import logging

logger = logging.getLogger(__name__)

class DataHandler:
    @staticmethod
    def fetch_data(symbol: str, timeframe: str = '5m', limit: int = 10) -> pd.DataFrame:
        clean_symbol = symbol.replace("/", "").upper()
        
        # لیست صرافی‌ها برای تست زنده
        test_targets = [
            {"name": "Binance", "url": f"https://api.binance.com/api/v3/klines?symbol={clean_symbol}&interval={timeframe}&limit={limit}"},
            {"name": "MEXC", "url": f"https://api.mexc.com/api/v3/klines?symbol={clean_symbol}&interval={timeframe}&limit={limit}"},
            {"name": "KuCoin", "url": f"https://api.kucoin.com/api/v1/market/candles?symbol={symbol.replace('/', '-')}&type={timeframe.replace('m', 'min')}"}
        ]

        for target in test_targets:
            try:
                print(f"📡 Testing connection to {target['name']}...")
                resp = requests.get(target['url'], timeout=7)
                print(f"✅ {target['name']} Response: {resp.status_code}")
                
                if resp.status_code == 200:
                    data = resp.json()
                    # اگر بایننس یا مکسی بود
                    raw_data = data if target['name'] != "KuCoin" else data.get('data', [])
                    df = pd.DataFrame(raw_data)
                    
                    # نام‌گذاری ستون‌ها برای سازگاری با بقیه کدها
                    if not df.empty:
                        df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume'] + list(df.columns[6:])
                        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                            df[col] = pd.to_numeric(df[col])
                        print(f"💰 Successfully fetched from {target['name']}")
                        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            except Exception as e:
                print(f"❌ {target['name']} Connection Failed: {e}")
        
        return pd.DataFrame()

    @staticmethod
    def fetch_ticker(symbol: str):
        # برای تست، قیمت فیکس برمی‌گردانیم تا مانیتورینگ متوقف نشود
        return {'symbol': symbol, 'last': 50000.0}

fetch_data = DataHandler.fetch_data
fetch_ticker = DataHandler.fetch_ticker
