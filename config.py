import os
from dotenv import load_dotenv

# بارگذاری متغیرها (در سیستم محلی از .env و در سرور از Environment Variables)
load_dotenv()

# --- تنظیمات تلگرام ---
# توکن ربات شما (از BotFather)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8066443971:AAFBvYtLTdQIrLe07CJ-X18UyaPi3Dpb5zo")

# آیدی کانال شما (حتماً با @ شروع شود)
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "@AsemanSignals")

# --- تنظیمات واچ‌لیست (ارزهای مورد نظر برای تحلیل) ---
WATCHLIST_STR = os.getenv("WATCHLIST", "ETHUSDT,ENAUSDT,1INCHUSDT,UNIUSDT,XRPUSDT")
WATCHLIST = [s.strip() for s in WATCHLIST_STR.split(",")]

# --- تنظیمات استراتژی و فیلترها ---
# حد نصاب امتیاز برای ارسال سیگنال (بین 2 تا 4 تنظیم شود)
MIN_SIGNAL_QUALITY = float(os.getenv("MIN_SIGNAL_QUALITY", "2.0"))

# غیرفعال کردن فیلترهای سخت‌گیرانه برای اطمینان از ارسال اولین سیگنال‌ها
ENABLE_MULTI_TF_FILTER = False
ENABLE_MARKET_REGIME_FILTER = False

# --- کلیدهای صرافی (اختیاری برای حالت سیگنال‌دهی) ---
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# --- تنظیمات زمانی ---
TIMEZONE = "Asia/Tehran"
