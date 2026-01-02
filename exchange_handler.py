import pandas as pd
import requests
import time

class DataHandler:
    """کلاس مدیریت دریافت داده از صرافی (نسخه دیباگ)"""

    @staticmethod
    def fetch_data(symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """
        دریافت کندل‌های OHLCV از API بیننس (Public)
        """
        try:
            # تبدیل نماد به فرمت بیننس (مثال BTC/USDT -> BTCUSDT)
            binance_symbol = symbol.replace("/", "")
            print(f"🔍 [Handler] Fetching data for {binance_symbol}...")
            
            url = f"https://api.binance.com/api/v3/klines"
            params = {
                "symbol": binance_symbol,
                "interval": timeframe,
                "limit": limit
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            # بررسی کد وضعیت (200 یعنی موفق)
            if response.status_code == 200:
                data = response.json()
                
                if not data:
                    print(f"⚠️ [Handler] No data returned for {binance_symbol}")
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
                
                print(f"✅ [Handler] Data received successfully for {binance_symbol} (Len: {len(df)})")
                return df
                
            else:
                # اینجا را دقیق ببینید: چه کدی می‌دهد؟ (418 یا 404؟)
                print(f"⚠️ [Handler] Binance API Error {response.status_code}: {response.text}")
                return pd.DataFrame()
            
        except requests.exceptions.Timeout:
            print(f"❌ [Handler] Request timed out for {symbol}. Connection too slow.")
            return pd.DataFrame()
        except requests.exceptions.RequestException as e:
            print(f"❌ [Handler] Connection error for {symbol}: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ [Handler] Critical Error fetching data: {e}")
            return pd.DataFrame()
