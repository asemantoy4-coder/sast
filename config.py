import os
from dotenv import load_dotenv

# بارگذاری متغیرها از فایل .env (محلی) یا Environment Variables (در Render)
load_dotenv()

# --- تنظیمات بازار ---
SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
INTERVAL = "5m"

# --- تنظیمات ریسک ---
RISK_PERCENT = os.getenv("RISK_PERCENT", "1.0")  # درصد ریسک
LEVERAGE = os.getenv("LEVERAGE", "10")          # اهرم

# --- تنظیمات تلگرام (امن) ---
# توجه: مقدار پیش‌فرض را خالی گذاشتم تا کد رندر از پنل بخواند
# توکن واقعی را در پنل Render وارد کنید، نه اینجا.
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# --- تنظیمات فنی ربات (برای حالت API این‌ها بیشتر در سمت جاوا مدیریت می‌شوند) ---
SCALP_INTERVAL = int(os.getenv("SCALP_INTERVAL", "2"))
SIGNAL_COOLDOWN = int(os.getenv("SIGNAL_COOLDOWN", "300"))
MAX_SIGNALS_PER_HOUR = int(os.getenv("MAX_SIGNALS_PER_HOUR", "12"))

# --- فیلترهای سیگنال (بسیار مهم - اگر نباشند کد خطا می‌دهد یا ساده کار می‌کند) ---
MIN_SIGNAL_QUALITY = float(os.getenv("MIN_SIGNAL_QUALITY", "6.5"))
ENABLE_MULTI_TF_FILTER = os.getenv("ENABLE_MULTI_TF_FILTER", "True").lower() in ["true", "1"]
ENABLE_MARKET_REGIME_FILTER = os.getenv("ENABLE_MARKET_REGIME_FILTER", "True").lower() in ["true", "1"]

# --- کلیدهای صرافی (اختیاری - در حالت فقط سیگنال‌دهی به کار نمی‌روند) ---
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
