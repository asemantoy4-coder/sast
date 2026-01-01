import os
from dotenv import load_dotenv

# بارگذاری متغیرها از فایل .env (اگر وجود داشته باشد)
load_dotenv()

# --- تنظیمات بازار ---
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL = "5m"
RISK_PERCENT = 1.0  # درصد ریسک در هر معامله
LEVERAGE = 10       # اهرم (در صورت استفاده از معاملات فیوچرز)

# --- تنظیمات تلگرام ---
# توکن دقیقاً بر اساس تصویری که فرستادید (فاصله اضافی بعد از AAF حذف شد)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8066443971:AAFBvYtLTdQIrLe07CJ-X18UyaPi3Dpb5zo")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "@AsemanSignals")

# --- تنظیمات فنی ربات ---
SCALP_INTERVAL = 2  # فرکانس اسکن بازار (ثانیه)
SIGNAL_COOLDOWN = 300  # فاصله بین دو سیگنال متوالی برای یک ارز (ثانیه)

# --- متغیرهای صرافی (اختیاری برای حالت فقط سیگنال) ---
API_KEY = os.getenv("BINANCE_API_KEY", "your_api_key")
API_SECRET = os.getenv("BINANCE_API_SECRET", "your_api_secret")
# حداکثر تعداد سیگنال مجاز در هر ساعت برای جلوگیری از اسپم
MAX_SIGNALS_PER_HOUR = 12
