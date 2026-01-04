import os
from dotenv import load_dotenv

load_dotenv()

# --- تنظیمات بازار ---
WATCHLIST_STR = os.getenv("WATCHLIST", "ETHUSDT,ENAUSDT,1INCHUSDT,UNIUSDT,XRPUSDT")
WATCHLIST = [s.strip() for s in WATCHLIST_STR.split(",")]

# --- تلگرام ---
# حتما در پنل Render این دو مورد را تعریف کنید
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# --- تنظیمات استراتژی (هماهنگ با main.py) ---
# عدد 2 باعث می‌شود در بازار فعلی هم سیگنال‌های مطمئن صادر شود
MIN_SIGNAL_QUALITY = 2.0 

# فعلا فیلترهای اضافی را غیرفعال می‌کنیم تا از سلامت ارسال پیام مطمئن شوید
ENABLE_MULTI_TF_FILTER = False
ENABLE_MARKET_REGIME_FILTER = False

# --- تنظیمات صرافی (عمومی) ---
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")
