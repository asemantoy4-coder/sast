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

@app.route('/force_analyze')
def force_analyze():
    """اجرای دستی تحلیل برای تمام ارزهای واچ‌لیست"""
    now = get_iran_time()
    results = []
    
    # اجرای تحلیل برای تک تک ارزها بدون توجه به ساعت
    for symbol in WATCHLIST:
        try:
            analyze_and_broadcast(symbol)
            results.append(f"Analyzed {symbol}")
        except Exception as e:
            results.append(f"Error {symbol}: {str(e)}")
            
    return jsonify({
        "message": "Manual analysis triggered",
        "time_iran": now.strftime('%H:%M:%S'),
        "results": results
    })
    
app = Flask(__name__)
port = int(os.environ.get("PORT", 5000))

# لیست ارزهای درخواستی شما
WATCHLIST = ["ETHUSDT", "ENAUSDT", "1INCHUSDT", "UNIUSDT", "XRPUSDT"]
# حافظه موقت برای ذخیره سیگنال‌های باز جهت اعلام سود
ACTIVE_SIGNALS = {} 

# --- توابع مدیریت زمان و تحلیل ---
def get_iran_time():
    return datetime.now(pytz.timezone('Asia/Tehran'))

def check_targets():
    """این بخش هر ۱۰ ثانیه قیمت‌ها را برای اعلام سود چک می‌کند"""
    while True:
        try:
            for symbol in list(ACTIVE_SIGNALS.keys()):
                signal = ACTIVE_SIGNALS[symbol]
                ticker = exchange_handler.DataHandler.fetch_ticker(symbol)
                if not ticker: continue
                
                current_price = ticker['last']
                
                # بررسی تارگت ۲
                if (signal['side'] == "BUY" and current_price >= signal['tp2']) or \
                   (signal['side'] == "SELL" and current_price <= signal['tp2']):
                    
                    msg = f"✅ *PROFIT TARGET 2 HIT!* ✅\n\n💰 {symbol}\n📈 Profit: {signal['tp2_pct']:.2f}%\n💵 Price: {current_price:.4f}\n\n✨ تبریک! تارگت اصلی محقق شد."
                    utils.send_telegram_notification(msg, "INFO")
                    del ACTIVE_SIGNALS[symbol]
                
                # بررسی استاپ لاس
                elif (signal['side'] == "BUY" and current_price <= signal['sl']) or \
                     (signal['side'] == "SELL" and current_price >= signal['sl']):
                    
                    msg = f"🛑 *STOP LOSS HIT* 🛑\n\n📉 {symbol}\n💵 Price: {current_price:.4f}\n\n⚠️ معامله با رعایت مدیریت ریسک بسته شد."
                    utils.send_telegram_notification(msg, "ERROR")
                    del ACTIVE_SIGNALS[symbol]
            
            time.sleep(10)
        except Exception as e:
            print(f"❌ Error in monitor: {e}")
            time.sleep(30)

def hourly_job():
    """تحلیل ساعتی ارزها"""
    now = get_iran_time()
    if 10 <= now.hour <= 19:
        for symbol in WATCHLIST:
            analyze_and_broadcast(symbol)
            time.sleep(2)

def analyze_and_broadcast(symbol):
    """تولید سیگنال و ذخیره برای پایش"""
    try:
        df = exchange_handler.DataHandler.fetch_data(symbol, '5m', limit=100)
        if df.empty: return
        analysis = utils.generate_scalp_signals(df)
        score = analysis.get('score', 0)
        
        if abs(score) >= 4:
            side = "BUY" if score > 0 else "SELL"
            current_price = analysis['price']
            sl = current_price * 0.995 if score > 0 else current_price * 1.005
            exits = utils.get_exit_levels(current_price, sl, direction=side)
            
            ACTIVE_SIGNALS[symbol] = {
                'side': side, 'tp2': exits['tp2'], 'sl': sl,
                'tp2_pct': abs(exits['tp2']-current_price)/current_price*100
            }
            
            msg = f"🚀 *NEW SIGNAL: {symbol}* 🚀\n📶 Side: {side}\n💵 Entry: {current_price:.4f}\n🎯 Target 2: {exits['tp2']:.4f}\n🛑 SL: {sl:.4f}\n📡 @AsemanSignals"
            utils.send_telegram_notification(msg, side)
    except: pass

# --- بخش اجرای سرور و زمان‌بند ---
def run_scheduler():
    schedule.every().hour.at(":00").do(hourly_job)
    while True:
        schedule.run_pending()
        time.sleep(30)

@app.route('/')
def home():
    return jsonify({"status": "active", "iran_time": get_iran_time().strftime('%H:%M:%S')})

if __name__ == "__main__":
    threading.Thread(target=run_scheduler, daemon=True).start()
    threading.Thread(target=check_targets, daemon=True).start()
    app.run(host='0.0.0.0', port=port)
