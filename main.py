import os
import time
import threading
import schedule
from flask import Flask, jsonify
from datetime import datetime
import pytz
import exchange_handler
import utils
import config

# ۱. راه اندازی اپلیکیشن فلاسگ
app = Flask(__name__)
port = int(os.environ.get("PORT", 5000))

# واچ‌لیست و حافظه سیگنال‌ها
WATCHLIST = config.WATCHLIST
ACTIVE_SIGNALS = {}

def get_iran_time():
    """محاسبه زمان فعلی تهران"""
    return datetime.now(pytz.timezone('Asia/Tehran'))

# ۲. بدنه اصلی تحلیل و ارسال پیام
def analyze_and_broadcast(symbol):
    try:
        # دریافت داده از صرافی
        df = exchange_handler.DataHandler.fetch_data(symbol, '5m', limit=100)
        if df.empty:
            print(f"⚠️ داده‌ای برای {symbol} دریافت نشد.")
            return
        
        # تحلیل تکنیکال
        analysis = utils.generate_scalp_signals(df)
        score = analysis.get('score', 0)
        
        # --- بخش تست: شرط روی 0 تنظیم شده تا پیام حتما ارسال شود ---
        if abs(score) >= 0:
            side = "BUY" if score >= 0 else "SELL"
            current_price = analysis['price']
            
            # محاسبه حد ضرر و تارگت
            sl = current_price * 0.995 if side == "BUY" else current_price * 1.005
            exits = utils.get_exit_levels(current_price, sl, direction=side)
            
            # ذخیره برای پایش تارگت
            ACTIVE_SIGNALS[symbol] = {
                'side': side, 
                'tp2': exits['tp2'], 
                'sl': sl,
                'tp2_pct': abs(exits['tp2']-current_price)/current_price*100
            }
            
            # ساخت پیام تلگرام با استفاده از آیدی کانال در config
            msg = (
                f"🚀 *NEW SIGNAL: {symbol}* 🚀\n"
                f"📶 Side: {'🟢 BUY' if side == 'BUY' else '🔴 SELL'}\n"
                f"💵 Entry: {current_price:.4f}\n"
                f"🎯 Target 2: {exits['tp2']:.4f}\n"
                f"🛑 SL: {sl:.4f}\n"
                f"📡 {config.TELEGRAM_CHAT_ID}"
            )
            
            # ارسال به تلگرام
            utils.send_telegram_notification(msg, side)
            print(f"✅ تلاش برای ارسال سیگنال {symbol} به تلگرام انجام شد.")
        else:
            print(f"ℹ️ امتیاز {symbol} برابر {score} است (کمتر از حد نصاب).")

    except Exception as e:
        print(f"❌ خطا در تحلیل {symbol}: {str(e)}")

# ۳. پایش لحظه‌ای قیمت‌ها برای تارگت و استاپ
def check_targets():
    while True:
        try:
            for symbol in list(ACTIVE_SIGNALS.keys()):
                sig = ACTIVE_SIGNALS[symbol]
                ticker = exchange_handler.DataHandler.fetch_ticker(symbol)
                if not ticker: continue
                
                price = ticker['last']
                
                # چک کردن تارگت
                if (sig['side'] == "BUY" and price >= sig['tp2']) or \
                   (sig['side'] == "SELL" and price <= sig['tp2']):
                    utils.send_telegram_notification(f"✅ TARGET HIT: {symbol}\n💰 Profit Achieved!", "INFO")
                    del ACTIVE_SIGNALS[symbol]
                
                # چک کردن استاپ
                elif (sig['side'] == "BUY" and price <= sig['sl']) or \
                     (sig['side'] == "SELL" and price >= sig['sl']):
                    utils.send_telegram_notification(f"🛑 STOP LOSS HIT: {symbol}", "ERROR")
                    del ACTIVE_SIGNALS[symbol]
            
            time.sleep(20)
        except Exception as e:
            print(f"❌ خطا در مانیتورینگ: {e}")
            time.sleep(30)

# ۴. زمان‌بندی (ساعتی)
def hourly_job():
    now = get_iran_time()
    # فقط بین ساعت ۱۰ صبح تا ۷ شب تهران اجرا شود
    if 10 <= now.hour <= 19:
        print(f"⏰ شروع تحلیل خودکار ساعت {now.hour}:00")
        for symbol in WATCHLIST:
            analyze_and_broadcast(symbol)
            time.sleep(2)

def run_scheduler():
    schedule.every().hour.at(":00").do(hourly_job)
    while True:
        schedule.run_pending()
        time.sleep(30)

# ۵. مسیرهای وب (Routes)
@app.route('/')
def home():
    return jsonify({
        "status": "online",
        "iran_time": get_iran_time().strftime('%H:%M:%S'),
        "monitored_pairs": list(ACTIVE_SIGNALS.keys()),
        "channel": config.TELEGRAM_CHAT_ID
    })

@app.route('/force_analyze')
def force_analyze():
    """تست دستی با لاگ‌گذاری کامل"""
    print("⚡ Manual Trigger: Starting analysis...")
    results = []
    
    # اطمینان از اینکه واچ‌لیست خالی نیست
    test_watchlist = ['BTC/USDT', 'ETH/USDT'] 
    
    for symbol in test_watchlist:
        try:
            print(f"🔍 Checking {symbol}...")
            # ۱. دریافت دیتا
            df = exchange_handler.fetch_data(symbol, '5m', limit=100)
            
            if df is None or df.empty:
                print(f"❌ No data for {symbol}")
                continue
                
            # ۲. تحلیل با استفاده از utils (حالت تست فعال)
            analysis = utils.generate_scalp_signals(df, test_mode=True)
            
            # ۳. ساخت پیام
            msg = f"🧪 *TEST SIGNAL*\n🪙 Symbol: {symbol}\n💰 Price: {analysis['price']}\n📊 Signal: {analysis['signal']}"
            
            # ۴. ارسال به تلگرام
            success = utils.send_telegram_notification(msg, analysis['signal'])
            
            results.append({"symbol": symbol, "sent": success, "signal": analysis['signal']})
            
        except Exception as e:
            print(f"🔥 Error analyzing {symbol}: {str(e)}")
            
    return jsonify({
        "status": "Analysis complete",
        "results": results,
        "time": datetime.now().strftime("%H:%M:%S")
    })

# ۶. شروع برنامه
if __name__ == "__main__":
    # اجرای ترد زمان‌بندی
    threading.Thread(target=run_scheduler, daemon=True).start()
    # اجرای ترد پایش قیمت
    threading.Thread(target=check_targets, daemon=True).start()
    
    print(f"🚀 Server is starting on port {port}...")
    app.run(host='0.0.0.0', port=port)
