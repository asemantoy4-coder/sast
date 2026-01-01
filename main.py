from exchange_handler import DataHandler
from utils import calculate_volume_profile, get_ichimoku
import config
import time

def run_scalper():
    print(f"🚀 Starting Scalp Bot for {config.SYMBOL}...")
    
    while True:
        try:
            # ۱. دریافت دیتا
            df = DataHandler.fetch_data(config.SYMBOL, config.INTERVAL)
            
            # ۲. محاسبات
            ichimoku = get_ichimoku(df)
            vp = calculate_volume_profile(df)
            
            # ۳. منطق ورود (تلاقی قیمت با POC و ایچیموکو)
            if ichimoku['current_price'] > vp['poc'] and ichimoku['current_price'] > ichimoku['tenkan']:
                print(f"🟢 Signal BUY | Price: {ichimoku['current_price']} | POC: {vp['poc']}")
            
            time.sleep(60) # صبر برای آپدیت بعدی
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    run_scalper()