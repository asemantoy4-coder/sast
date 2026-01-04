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

# ۱. تعریف اپلیکیشن (باید حتماً اینجا باشد تا خطا ندهد)
app = Flask(__name__)
port = int(os.environ.get("PORT", 5000))

WATCHLIST = ["ETHUSDT", "ENAUSDT", "1INCHUSDT", "UNIUSDT", "XRPUSDT"]
ACTIVE_SIGNALS = {}

def get_iran_time():
    return datetime.now(pytz.timezone('Asia/Tehran'))

# ۲. توابع منطق اصلی ربات
def analyze_and_broadcast(symbol):
    try:
        df = exchange_handler.DataHandler.fetch_data(symbol, '5m', limit=100)
        if df.empty: return
        
        analysis = utils.generate_scalp_signals(df)
        score = analysis.get('score', 0)
        
        # اگر امتیاز کافی بود پیام بفرست
        if abs(score) >= 4:
            side = "BUY" if score > 0 else "SELL"
            current_price = analysis['price']
            sl = current_price * 0.995 if score > 0 else current_price * 1.005
            exits = utils.get_exit_levels(current_price, sl, direction=side)
            
            ACTIVE_SIGNALS[symbol] = {
                'side': side, 'tp2': exits['tp2'], 'sl': sl,
                'tp2_pct': abs(exits['tp2']-current_price)/current_price*100
            }
            
            msg = f"🚀 *NEW SIGNAL: {symbol}* 🚀\n📶 Side: {'🟢 BUY' if side == 'BUY' else '🔴 SELL'}\n💵 Entry: {current_price:.4f}\n🎯 Target 2: {exits['tp2']:.4f}\n🛑 SL: {sl:.4f}\n📡 @AsemanSignals"
            utils.send_telegram_notification(msg, side)
            print(f"✅ Signal sent for {symbol}")
        else:
            print(f"ℹ️ {symbol} score: {score} (No action)")
    except Exception as e:
        print(f"❌ Error in analysis: {e}")

def check_targets():
    while True:
        try:
            for symbol in list(ACTIVE_SIGNALS.keys()):
                sig = ACTIVE_SIGNALS[symbol]
                ticker = exchange_handler.DataHandler.fetch_ticker(symbol)
                if not ticker: continue
                price = ticker['last']
                if (sig['side'] == "BUY" and price >= sig['tp2']) or (sig['side'] == "SELL" and price <= sig['tp2']):
                    utils.send_telegram_notification(f"✅ TARGET HIT: {symbol}", "INFO")
                    del ACTIVE_SIGNALS[symbol]
            time.sleep(20)
        except: time.sleep(30)

def hourly_job():
    now = get_iran_time()
    if 10 <= now.hour <= 19:
        for symbol in WATCHLIST:
            analyze_and_broadcast(symbol)
            time.sleep(2)

def run_scheduler():
    schedule.every().hour.at(":00").do(hourly_job)
    while True:
        schedule.run_pending()
        time.sleep(30)

# ۳. مسیرهای وب (Routes) - همه بعد از تعریف app
@app.route('/')
def home():
    return jsonify({"status": "active", "iran_time": get_iran_time().strftime('%H:%M:%S')})

@app.route('/force_analyze')
def force_analyze():
    """تست دستی"""
    for symbol in WATCHLIST:
        analyze_and_broadcast(symbol)
    return jsonify({"message": "Manual analysis triggered"})

# ۴. اجرای نهایی
if __name__ == "__main__":
    threading.Thread(target=run_scheduler, daemon=True).start()
    threading.Thread(target=check_targets, daemon=True).start()
    app.run(host='0.0.0.0', port=port)
