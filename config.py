import os
from dotenv import load_dotenv

load_dotenv()

# تنظیمات بازار
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL = "5m"
LEVERAGE = 10

# متغیرهای محیطی
API_KEY = os.getenv("BINANCE_API_KEY", "your_api_key")
API_SECRET = os.getenv("BINANCE_API_SECRET", "your_api_secret")