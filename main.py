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

# ==================== INITIALIZATION ====================
app = Flask(__name__)
port = int(os.environ.get("PORT", 5000))

# لیست ارزهای مورد نظر
WATCHLIST = ["ETHUSDT", "ENAUSDT", "1INCHUSDT", "UNIUSDT", "XRPUSDT"]
# حافظه برای پایش لحظه‌ای تارگت‌ها
ACTIVE_SIGNALS = {}

def get_iran_time():
    """دریافت زمان فعلی به وقت ایران"""
    return datetime.now(pytz.timezone('Asia/Tehran'))

# ==================== CORE LOGIC ====================

def analyze_and_broadcast(symbol):
    """تحلیل فنی و ارسال به تلگرام در صورت وجود سیگنال"""
    try:
        # دریافت داده
        df = exchange_handler.DataHandler.fetch_data(symbol, '5m', limit=100)
        if df.empty:
            print(f"⚠️ No data for {symbol}")
            return

        # تحلیل توسط موتور utils
        analysis = utils.generate_scalp_signals(df)
        score = analysis.get('score', 0)
        
        # حد نصاب امتیاز برای ارسال سیگنال (قابل تغییر در config یا اینجا)
        if abs(score) >= 4:
            side = "BUY" if score > 0 else "SELL"
            current_price = analysis['price']
            
            # محاسبه حد ضرر و تارگت‌ها
            sl = current_price * 0.995 if score > 0 else current_price * 1.005
            exits = utils.get_exit_levels(current_price, sl, direction=side)
            
            # ذخیره در حافظه برای چک کردن تارگت در لحظه
            ACTIVE_SIGNALS[symbol] = {
                'side': side,
                'tp2': exits['tp2'],
                'sl': sl,
                'tp2_pct': abs(exits['tp2']-current_price)/current_price*100
            }
            
            # ساخت پیام تلگرام
            msg = f"🚀 *NEW SIGNAL: {symbol}* 🚀\n📶 Side: {'🟢 BUY' if side == 'BUY' else '🔴 SELL'}\n💵 Entry: {current_price:.4f}\n🎯 Target 2: {exits['tp2']:.4f}\n🛑 SL: {sl:.4f}\n📡 @AsemanSignals"
            utils.send_telegram_notification(msg, side)
            print(f"✅ Signal sent for {symbol}")
        else:
            print(f"ℹ️ {symbol} score is {score}, not enough for signal.")

    except Exception as e:
        print(f"❌ Error analyzing {symbol}: {e}")

def check_targets():
    """پایش لحظه‌ای قیمت برای اعلام سود یا ضرر"""
    while True:
        try:
            for symbol in list(ACTIVE_SIGNALS.keys()):
                signal = ACTIVE_SIGNALS[symbol]
                ticker = exchange_handler.DataHandler.fetch_ticker(symbol)
                if not ticker: continue
                
                current_price = ticker['last']
                
                # چک کردن تارگت ۲
                if (signal['side'] == "BUY" and current_price >= signal['tp2']) or \
                   (signal['side'] == "SELL" and current_price <= signal['tp2']):
                    msg = f"✅ *PROFIT TARGET 2 HIT!* ✅\n💰 {symbol}\n📈 Profit: {signal['tp2_pct']:.2f}%\n✨ مبارک است!"
                    utils.send_telegram_notification(msg, "INFO")
                    del ACTIVE_SIGNALS[symbol]
                
                # چک کردن استاپ لاس
                elif (signal['side'] == "BUY" and current_price <= signal['sl']) or \
                     (signal['side'] == "SELL" and current_price >= signal['sl']):
                    msg = f"🛑 *STOP LOSS HIT* 🛑\n📉 {symbol}\n⚠️ مدیریت ریسک رعایت شود."
                    utils.send_telegram_notification(msg, "ERROR")
                    del ACTIVE_SIGNALS[symbol]
            
            time.sleep(15)
        except Exception as e:
            print(f"❌ Monitor error: {e}")
            time.sleep(30)

def hourly_job():
    """وظیفه ساعتی در بازه ۱۰ صبح تا ۷ شب ایران"""
    now = get_iran_time()
    if 10 <= now.hour <= 19:
        print(f"⏰ Starting scheduled analysis at {now.hour}:00")
        for symbol in WATCHLIST:
            analyze_and_broadcast(symbol)
            time.sleep(2)

def run_scheduler():
    """اجرای زمان‌بند در پس‌زمینه"""
    schedule.every().hour.at(":00").do(hourly_job)
    while True:
        schedule.run_pending()
        time.sleep(30)

# ==================== FLASK ROUTES ====================

@app.route('/')
def home():
    """نمایش وضعیت ربات در آدرس اصلی"""
    return jsonify({
        "status": "active",
        "iran_time": get_iran_time().strftime('%H:%M:%S'),
        "monitored_pairs": list(ACTIVE_SIGNALS.keys())
    })

@app.route('/force_analyze')
def force_analyze():
    """اجرای دستی تحلیل برای تست سریع"""
    now = get_iran_time()
    for symbol in WATCHLIST:
        analyze_and_broadcast(symbol)
    return jsonify({
        "message": "Manual trigger executed",
        "time": now.strftime('%H:%M:%S')
    })

# ==================== START SERVER ====================

if __name__ == "__main__":
    # شروع تردها
    threading.Thread(target=run_scheduler, daemon=True).start()
    threading.Thread(target=check_targets, daemon=True).start()
    
    # اجرای سرور فلاسگ
    print(f"🚀 Aseman Server started on port {port}")
    app.run(host='0.0.0.0', port=port)
