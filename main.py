import os
import time
import threading
import schedule
from flask import Flask, jsonify, request
from datetime import datetime, timedelta
import pytz
import exchange_handler
import utils
import config
import json
from typing import Dict, List, Optional, Any

# ۱. راه‌اندازی اپلیکیشن Flask
app = Flask(__name__)
port = int(os.environ.get("PORT", 5000))

# واچ‌لیست و حافظه سیگنال‌ها
WATCHLIST = config.WATCHLIST if hasattr(config, 'WATCHLIST') else ["BTCUSDT", "ETHUSDT"]
ACTIVE_SIGNALS: Dict[str, Dict] = {}
SIGNAL_HISTORY: List[Dict] = []
SYSTEM_START_TIME = datetime.now(pytz.timezone('Asia/Tehran'))

# تنظیمات سیستم
class SystemConfig:
    CHECK_INTERVAL = 20  # ثانیه
    MIN_SCORE = 3  # حداقل امتیاز برای سیگنال
    TRADING_HOURS = (10, 19)  # ساعت ۱۰ تا ۱۹
    MAX_HISTORY = 100  # حداکثر تاریخچه ذخیره‌شده
    RISK_FREE_ENABLED = True  # فعال‌سازی حالت ریسک‌فری

def get_iran_time() -> datetime:
    """محاسبه زمان فعلی تهران"""
    return datetime.now(pytz.timezone('Asia/Tehran'))

def load_signal_history():
    """بارگذاری تاریخچه سیگنال‌ها از فایل"""
    global SIGNAL_HISTORY
    try:
        if os.path.exists('signal_history.json'):
            with open('signal_history.json', 'r') as f:
                SIGNAL_HISTORY = json.load(f)
                print(f"✅ تاریخچه {len(SIGNAL_HISTORY)} سیگنال بارگذاری شد")
    except Exception as e:
        print(f"❌ خطا در بارگذاری تاریخچه: {e}")

def save_signal_history():
    """ذخیره تاریخچه سیگنال‌ها در فایل"""
    try:
        with open('signal_history.json', 'w') as f:
            json.dump(SIGNAL_HISTORY[-SystemConfig.MAX_HISTORY:], f, indent=2)
    except Exception as e:
        print(f"❌ خطا در ذخیره تاریخچه: {e}")

# ۲. بدنه اصلی تحلیل و ارسال پیام
def analyze_and_broadcast(symbol: str, force: bool = False) -> Dict[str, Any]:
    """
    تحلیل نماد و ارسال سیگنال در صورت وجود شرایط
    """
    try:
        # بررسی زمان معاملاتی
        iran_time = get_iran_time()
        if not force and not (SystemConfig.TRADING_HOURS[0] <= iran_time.hour <= SystemConfig.TRADING_HOURS[1]):
            print(f"⏰ خارج از ساعت معاملاتی ({iran_time.hour}:{iran_time.minute})")
            return {"status": "outside_trading_hours"}
        
        # تمیز کردن نام نماد
        clean_symbol = symbol.replace("/", "").upper()
        
        # دریافت داده از صرافی
        df = exchange_handler.DataHandler.fetch_data(clean_symbol, '5m', limit=100)
        if df is None or df.empty:
            print(f"⚠️ داده‌ای برای {clean_symbol} دریافت نشد.")
            return {"status": "no_data", "symbol": clean_symbol}
        
        # تحلیل تکنیکال
        analysis = utils.generate_scalp_signals(df)
        score = analysis.get('score', 0)
        current_price = analysis.get('price', 0)
        
        print(f"📊 تحلیل {clean_symbol}: امتیاز={score}, قیمت={current_price}")
        
        # بررسی شرایط سیگنال
        if abs(score) >= SystemConfig.MIN_SCORE or force:
            side = "BUY" if score >= 0 else "SELL"
            
            # محاسبه حد ضرر و تارگت‌ها
            if side == "BUY":
                sl = current_price * 0.995
                risk = current_price - sl
                tp1 = current_price + (risk * 1.5)
                tp2 = current_price + (risk * 3)
            else:  # SELL
                sl = current_price * 1.005
                risk = sl - current_price
                tp1 = current_price - (risk * 1.5)
                tp2 = current_price - (risk * 3)
            
            # ذخیره اطلاعات سیگنال
            signal_data = {
                'symbol': clean_symbol,
                'side': side,
                'entry': current_price,
                'score': abs(score),
                'exit_levels': {
                    'tp1': tp1,
                    'tp2': tp2,
                    'stop_loss': sl,
                    'direction': side,
                    'risk_percentage': 0.5 if side == 'BUY' else 0.5
                },
                'timestamp': iran_time.isoformat(),
                'status': 'ACTIVE',
                'notifications_sent': {
                    'tp1': False,
                    'tp2': False,
                    'sl': False
                },
                'force': force
            }
            
            # بررسی وجود سیگنال فعال برای این نماد
            if clean_symbol in ACTIVE_SIGNALS:
                old_status = ACTIVE_SIGNALS[clean_symbol].get('status', 'UNKNOWN')
                print(f"⚠️ سیگنال فعال قبلی برای {clean_symbol} با وضعیت {old_status}")
                
                # اگر سیگنال قبلی هنوز فعال است، ارسال نکن
                if old_status == 'ACTIVE':
                    return {
                        "status": "active_signal_exists",
                        "symbol": clean_symbol,
                        "message": "سیگنال فعال قبلی هنوز باز است"
                    }
            
            # ذخیره در حافظه فعال
            ACTIVE_SIGNALS[clean_symbol] = signal_data
            
            # اضافه به تاریخچه
            SIGNAL_HISTORY.append(signal_data.copy())
            if len(SIGNAL_HISTORY) > SystemConfig.MAX_HISTORY:
                SIGNAL_HISTORY.pop(0)
            
            # ساخت پیام تلگرام
            emoji = "🟢" if side == "BUY" else "🔴"
            signal_type = "🔧 FORCE" if force else "🚀 AUTO"
            
            msg = (
                f"{signal_type} *SIGNAL: {clean_symbol}* {emoji}\n"
                f"📶 Direction: {side}\n"
                f"📊 Score: {abs(score)}/10\n"
                f"💵 Entry Price: {current_price:.4f}\n"
                f"🎯 Take Profit 1: {tp1:.4f}\n"
                f"🎯 Take Profit 2: {tp2:.4f}\n"
                f"🛑 Stop Loss: {sl:.4f}\n"
                f"📈 Risk/Reward: 1:3\n"
                f"⏰ Time: {iran_time.strftime('%H:%M:%S')}\n"
                f"📡 Channel: {config.TELEGRAM_CHAT_ID if hasattr(config, 'TELEGRAM_CHAT_ID') else 'N/A'}\n"
                f"#{clean_symbol.replace('USDT', '')} #{side}"
            )
            
            # ارسال به تلگرام
            success = utils.send_telegram_notification(msg, side)
            
            if success:
                print(f"✅ سیگنال {clean_symbol} ارسال شد. وضعیت: ACTIVE")
                return {
                    "status": "success",
                    "symbol": clean_symbol,
                    "side": side,
                    "entry": current_price,
                    "tp1": tp1,
                    "tp2": tp2,
                    "sl": sl
                }
            else:
                print(f"❌ ارسال سیگنال {clean_symbol} ناموفق بود")
                # اگر ارسال ناموفق بود، سیگنال را حذف کن
                if clean_symbol in ACTIVE_SIGNALS:
                    del ACTIVE_SIGNALS[clean_symbol]
                return {"status": "telegram_error", "symbol": clean_symbol}
        
        else:
            print(f"ℹ️ امتیاز {clean_symbol}: {score} (کمتر از حد نصاب {SystemConfig.MIN_SCORE})")
            return {
                "status": "low_score",
                "symbol": clean_symbol,
                "score": score,
                "min_required": SystemConfig.MIN_SCORE
            }
            
    except Exception as e:
        error_msg = f"❌ خطا در تحلیل {symbol}: {str(e)}"
        print(error_msg)
        return {"status": "error", "symbol": symbol, "error": str(e)}

# ۳. منطق بررسی تارگت‌ها و استاپ‌لاس
def check_active_signals(symbol: str, current_price: float, signal_data: Dict) -> str:
    """
    بررسی اینکه آیا قیمت به تارگت‌ها یا استاپ‌لاس رسیده است
    """
    if symbol not in ACTIVE_SIGNALS:
        return "NOT_FOUND"
    
    levels = signal_data.get('exit_levels')
    if not levels:
        return "NO_LEVELS"
    
    side = levels.get('direction', 'BUY')
    status = "ACTIVE"
    
    # محاسبه سود/ضرر
    if side == 'BUY':
        profit_pct = ((current_price - signal_data['entry']) / signal_data['entry']) * 100
    else:  # SELL
        profit_pct = ((signal_data['entry'] - current_price) / signal_data['entry']) * 100
    
    # بررسی شرایط
    if side == 'BUY':
        # بررسی TP2
        if not signal_data['notifications_sent']['tp2'] and current_price >= levels['tp2']:
            send_target_notification(symbol, current_price, signal_data, "TP2", profit_pct)
            signal_data['notifications_sent']['tp2'] = True
            signal_data['status'] = "CLOSED_TP2"
            status = "CLOSED"
            
        # بررسی TP1
        elif not signal_data['notifications_sent']['tp1'] and current_price >= levels['tp1']:
            send_target_notification(symbol, current_price, signal_data, "TP1", profit_pct)
            signal_data['notifications_sent']['tp1'] = True
            
            # فعال‌سازی ریسک‌فری
            if SystemConfig.RISK_FREE_ENABLED:
                signal_data['exit_levels']['stop_loss'] = signal_data['entry']
                print(f"🛡️ ریسک‌فری فعال شد برای {symbol} - استاپ به نقطه ورود منتقل شد")
            
        # بررسی Stop Loss
        elif not signal_data['notifications_sent']['sl'] and current_price <= levels['stop_loss']:
            send_stop_loss_notification(symbol, current_price, signal_data, profit_pct)
            signal_data['notifications_sent']['sl'] = True
            signal_data['status'] = "CLOSED_SL"
            status = "CLOSED"
            
    elif side == 'SELL':
        # بررسی TP2
        if not signal_data['notifications_sent']['tp2'] and current_price <= levels['tp2']:
            send_target_notification(symbol, current_price, signal_data, "TP2", profit_pct)
            signal_data['notifications_sent']['tp2'] = True
            signal_data['status'] = "CLOSED_TP2"
            status = "CLOSED"
            
        # بررسی TP1
        elif not signal_data['notifications_sent']['tp1'] and current_price <= levels['tp1']:
            send_target_notification(symbol, current_price, signal_data, "TP1", profit_pct)
            signal_data['notifications_sent']['tp1'] = True
            
            # فعال‌سازی ریسک‌فری
            if SystemConfig.RISK_FREE_ENABLED:
                signal_data['exit_levels']['stop_loss'] = signal_data['entry']
                print(f"🛡️ ریسک‌فری فعال شد برای {symbol} - استاپ به نقطه ورود منتقل شد")
            
        # بررسی Stop Loss
        elif not signal_data['notifications_sent']['sl'] and current_price >= levels['stop_loss']:
            send_stop_loss_notification(symbol, current_price, signal_data, profit_pct)
            signal_data['notifications_sent']['sl'] = True
            signal_data['status'] = "CLOSED_SL"
            status = "CLOSED"
    
    # اگر وضعیت بسته شد، از لیست فعال حذف کن
    if status == "CLOSED":
        close_signal(symbol, current_price, signal_data, profit_pct)
        save_signal_history()  # ذخیره تاریخچه
    
    return status

def send_target_notification(symbol: str, price: float, signal_data: Dict, target_level: str, profit_pct: float):
    """ارسال اعلان رسیدن به تارگت"""
    emoji = "💰" if target_level == "TP2" else "✅"
    title = "FINAL TARGET HIT! 🔥" if target_level == "TP2" else "FIRST TARGET REACHED"
    
    msg = (
        f"{emoji} *{symbol} - {title}*\n"
        f"🎯 {target_level}: {signal_data['exit_levels'][target_level.lower()]:.4f}\n"
        f"💵 Current: {price:.4f}\n"
        f"📈 Profit: {profit_pct:.2f}%\n"
        f"📊 Entry: {signal_data['entry']:.4f}\n"
        f"🕒 Duration: {calculate_duration(signal_data['timestamp'])}\n"
    )
    
    if target_level == "TP1" and SystemConfig.RISK_FREE_ENABLED:
        msg += f"\n🛡️ *RISK-FREE ACTIVATED*\nStop Loss moved to entry point"
    
    utils.send_telegram_notification(msg, "TARGET" if target_level == "TP2" else "INFO")

def send_stop_loss_notification(symbol: str, price: float, signal_data: Dict, profit_pct: float):
    """ارسال اعلان رسیدن به استاپ‌لاس"""
    msg = (
        f"🛑 *{symbol} - STOP LOSS HIT!*\n"
        f"📉 SL: {signal_data['exit_levels']['stop_loss']:.4f}\n"
        f"💵 Current: {price:.4f}\n"
        f"📊 Entry: {signal_data['entry']:.4f}\n"
        f"📉 Loss: {profit_pct:.2f}%\n"
        f"🕒 Duration: {calculate_duration(signal_data['timestamp'])}\n"
        f"❌ Position CLOSED"
    )
    utils.send_telegram_notification(msg, "STOP")

def close_signal(symbol: str, close_price: float, signal_data: Dict, profit_pct: float):
    """بستن سیگنال و ذخیره اطلاعات نهایی"""
    signal_data['closed_at'] = close_price
    signal_data['closed_time'] = get_iran_time().isoformat()
    signal_data['final_profit_pct'] = profit_pct
    signal_data['duration'] = calculate_duration(signal_data['timestamp'])
    
    print(f"📋 سیگنال {symbol} بسته شد. سود: {profit_pct:.2f}%")
    
    # حذف از لیست فعال
    if symbol in ACTIVE_SIGNALS:
        del ACTIVE_SIGNALS[symbol]

def calculate_duration(timestamp: str) -> str:
    """محاسبه مدت زمان از ایجاد سیگنال"""
    try:
        start = datetime.fromisoformat(timestamp)
        now = get_iran_time()
        duration = now - start
        
        if duration.days > 0:
            return f"{duration.days}d {duration.seconds//3600}h"
        elif duration.seconds >= 3600:
            return f"{duration.seconds//3600}h {(duration.seconds%3600)//60}m"
        else:
            return f"{duration.seconds//60}m"
    except:
        return "N/A"

# ۴. پایش لحظه‌ای قیمت‌ها
def check_targets():
    """مانیتورینگ لحظه‌ای قیمت برای سیگنال‌های فعال"""
    last_status_log = time.time()
    
    while True:
        try:
            symbols_to_check = list(ACTIVE_SIGNALS.keys())
            
            if not symbols_to_check:
                # لاگ وضعیت هر 5 دقیقه
                if time.time() - last_status_log > 300:
                    print(f"📊 سیستم فعال - هیچ سیگنال فعالی وجود ندارد. زمان: {get_iran_time().strftime('%H:%M:%S')}")
                    last_status_log = time.time()
                time.sleep(SystemConfig.CHECK_INTERVAL)
                continue
            
            print(f"🔍 مانیتورینگ {len(symbols_to_check)} سیگنال فعال...")
            
            for symbol in symbols_to_check:
                if symbol not in ACTIVE_SIGNALS:
                    continue
                
                # دریافت قیمت لحظه‌ای
                ticker = exchange_handler.DataHandler.fetch_ticker(symbol)
                if not ticker:
                    print(f"⚠️ دریافت قیمت برای {symbol} ناموفق بود")
                    continue
                
                price = ticker.get('last', 0)
                if price == 0:
                    continue
                
                signal_data = ACTIVE_SIGNALS[symbol]
                
                # بررسی وضعیت سیگنال
                status = check_active_signals(symbol, price, signal_data)
                
                # نمایش وضعیت لحظه‌ای
                if status == "ACTIVE" and time.time() - last_status_log > 300:
                    levels = signal_data['exit_levels']
                    print(f"📊 {symbol}: {price:.4f} | TP1: {levels['tp1']:.4f} | TP2: {levels['tp2']:.4f} | SL: {levels['stop_loss']:.4f}")
            
            if time.time() - last_status_log > 300:
                last_status_log = time.time()
            
            time.sleep(SystemConfig.CHECK_INTERVAL)
            
        except Exception as e:
            print(f"❌ خطا در مانیتورینگ: {e}")
            time.sleep(30)

# ۵. زمان‌بندی (ساعتی)
def hourly_job():
    """اجرای تحلیل ساعتی"""
    now = get_iran_time()
    
    # فقط در ساعات معاملاتی
    if SystemConfig.TRADING_HOURS[0] <= now.hour <= SystemConfig.TRADING_HOURS[1]:
        print(f"⏰ شروع تحلیل ساعتی ساعت {now.hour}:{now.minute:02d}")
        
        for symbol in WATCHLIST:
            analyze_and_broadcast(symbol, force=False)
            time.sleep(2)  # تاخیر بین تحلیل نمادها
    
    else:
        print(f"⏰ خارج از ساعت معاملاتی ({now.hour}:{now.minute:02d}) - تحلیل انجام نمی‌شود")

def run_scheduler():
    """اجرای زمان‌بند"""
    # اجرای هر ساعت در دقیقه ۰
    schedule.every().hour.at(":00").do(hourly_job)
    
    # اجرای تست هر ۱۵ دقیقه (برای توسعه)
    # schedule.every(15).minutes.do(lambda: print(f"🧪 تست زمان‌بند - {get_iran_time().strftime('%H:%M:%S')}"))
    
    print("⏰ زمان‌بند راه‌اندازی شد")
    
    while True:
        schedule.run_pending()
        time.sleep(30)

# ۶. مسیرهای وب (Routes)
@app.route('/')
def home():
    """صفحه اصلی"""
    return jsonify({
        "status": "online",
        "name": "Crypto Trading Bot",
        "version": "2.0",
        "iran_time": get_iran_time().strftime('%Y-%m-%d %H:%M:%S'),
        "active_signals": len(ACTIVE_SIGNALS),
        "trading_hours": f"{SystemConfig.TRADING_HOURS[0]}:00 - {SystemConfig.TRADING_HOURS[1]}:00",
        "uptime": str(datetime.now(pytz.timezone('Asia/Tehran')) - SYSTEM_START_TIME),
        "endpoints": {
            "/": "این صفحه",
            "/signals": "وضعیت سیگنال‌ها",
            "/analyze/<symbol>": "تحلیل نماد",
            "/force_analyze": "تحلیل اجباری واچ‌لیست",
            "/check/<symbol>": "بررسی نماد",
            "/stats": "آمار سیستم"
        }
    })

@app.route('/signals')
def signals_status():
    """نمایش وضعیت سیگنال‌های فعال و تاریخچه"""
    active_signals = []
    
    for symbol, data in ACTIVE_SIGNALS.items():
        # دریافت قیمت لحظه‌ای
        ticker = exchange_handler.DataHandler.fetch_ticker(symbol)
        current_price = ticker.get('last', 0) if ticker else 0
        
        # محاسبه سود/ضرر
        if data['side'] == 'BUY':
            profit_pct = ((current_price - data['entry']) / data['entry'] * 100) if current_price > 0 else 0
        else:
            profit_pct = ((data['entry'] - current_price) / data['entry'] * 100) if current_price > 0 else 0
        
        active_signals.append({
            'symbol': symbol,
            'side': data['side'],
            'entry': data['entry'],
            'current_price': current_price,
            'profit_pct': round(profit_pct, 2),
            'tp1': data['exit_levels']['tp1'],
            'tp2': data['exit_levels']['tp2'],
            'sl': data['exit_levels']['stop_loss'],
            'status': data['status'],
            'score': data.get('score', 0),
            'timestamp': data['timestamp'],
            'duration': calculate_duration(data['timestamp'])
        })
    
    # تاریخچه سیگنال‌های اخیر
    recent_history = SIGNAL_HISTORY[-20:] if len(SIGNAL_HISTORY) > 20 else SIGNAL_HISTORY
    
    return jsonify({
        "active_signals": active_signals,
        "active_count": len(active_signals),
        "recent_history": recent_history,
        "total_history": len(SIGNAL_HISTORY),
        "system_time": get_iran_time().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/analyze/<symbol>')
def analyze_symbol(symbol: str):
    """تحلیل دستی یک نماد"""
    force = request.args.get('force', 'false').lower() == 'true'
    result = analyze_and_broadcast(symbol, force=force)
    return jsonify(result)

@app.route('/force_analyze')
def force_analyze():
    """تحلیل اجباری کل واچ‌لیست"""
    results = []
    
    # استفاده از واچ‌لیست کانفیگ
    watchlist = WATCHLIST
    
    print(f"🚀 شروع تحلیل اجباری {len(watchlist)} نماد")
    
    for symbol in watchlist:
        try:
            result = analyze_and_broadcast(symbol, force=True)
            results.append(result)
            time.sleep(1)  # تاخیر برای جلوگیری از محدودیت API
            
        except Exception as e:
            results.append({
                "symbol": symbol,
                "status": "error",
                "error": str(e)
            })
    
    return jsonify({
        "status": "completed",
        "total": len(watchlist),
        "successful": len([r for r in results if r.get('status') == 'success']),
        "results": results
    })

@app.route('/check/<symbol>')
def check_symbol(symbol: str):
    """بررسی وضعیت یک نماد"""
    try:
        clean_symbol = symbol.replace("/", "").upper()
        ticker = exchange_handler.DataHandler.fetch_ticker(clean_symbol)
        
        if not ticker:
            return jsonify({"error": "No ticker data available"}), 404
        
        price = ticker.get('last', 0)
        
        if clean_symbol in ACTIVE_SIGNALS:
            status = check_active_signals(clean_symbol, price, ACTIVE_SIGNALS[clean_symbol])
            return jsonify({
                "symbol": clean_symbol,
                "price": price,
                "status": status,
                "signal_data": ACTIVE_SIGNALS.get(clean_symbol),
                "has_active_signal": True
            })
        else:
            # بررسی تاریخچه
            history_for_symbol = [s for s in SIGNAL_HISTORY if s.get('symbol') == clean_symbol]
            recent_history = history_for_symbol[-5:] if len(history_for_symbol) > 5 else history_for_symbol
            
            return jsonify({
                "symbol": clean_symbol,
                "price": price,
                "status": "NO_ACTIVE_SIGNAL",
                "recent_history": recent_history,
                "has_active_signal": False
            })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stats')
def system_stats():
    """آمار و گزارش عملکرد سیستم"""
    total_signals = len(SIGNAL_HISTORY)
    successful_signals = len([s for s in SIGNAL_HISTORY if s.get('status', '').startswith('CLOSED_TP')])
    stop_loss_signals = len([s for s in SIGNAL_HISTORY if s.get('status') == 'CLOSED_SL'])
    active_signals = len(ACTIVE_SIGNALS)
    
    # محاسبه میانگین سود
    closed_signals = [s for s in SIGNAL_HISTORY if 'final_profit_pct' in s]
    avg_profit = sum(s['final_profit_pct'] for s in closed_signals) / len(closed_signals) if closed_signals else 0
    
    return jsonify({
        "system": {
            "start_time": SYSTEM_START_TIME.strftime('%Y-%m-%d %H:%M:%S'),
            "uptime": str(datetime.now(pytz.timezone('Asia/Tehran')) - SYSTEM_START_TIME),
            "iran_time": get_iran_time().strftime('%Y-%m-%d %H:%M:%S')
        },
        "performance": {
            "total_signals": total_signals,
            "active_signals": active_signals,
            "successful_closed": successful_signals,
            "stop_loss_closed": stop_loss_signals,
            "win_rate": f"{(successful_signals/(successful_signals+stop_loss_signals)*100 if (successful_signals+stop_loss_signals) > 0 else 0):.1f}%",
            "average_profit": f"{avg_profit:.2f}%"
        },
        "config": {
            "trading_hours": SystemConfig.TRADING_HOURS,
            "check_interval": SystemConfig.CHECK_INTERVAL,
            "min_score": SystemConfig.MIN_SCORE,
            "risk_free_enabled": SystemConfig.RISK_FREE_ENABLED
        },
        "watchlist": WATCHLIST
    })

@app.route('/settings', methods=['GET', 'POST'])
def system_settings():
    """مدیریت تنظیمات سیستم"""
    if request.method == 'GET':
        return jsonify({
            "trading_hours": SystemConfig.TRADING_HOURS,
            "check_interval": SystemConfig.CHECK_INTERVAL,
            "min_score": SystemConfig.MIN_SCORE,
            "risk_free_enabled": SystemConfig.RISK_FREE_ENABLED,
            "max_history": SystemConfig.MAX_HISTORY
        })
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            
            if 'trading_hours' in data:
                SystemConfig.TRADING_HOURS = tuple(data['trading_hours'])
            
            if 'check_interval' in data:
                SystemConfig.CHECK_INTERVAL = int(data['check_interval'])
            
            if 'min_score' in data:
                SystemConfig.MIN_SCORE = int(data['min_score'])
            
            if 'risk_free_enabled' in data:
                SystemConfig.RISK_FREE_ENABLED = bool(data['risk_free_enabled'])
            
            return jsonify({
                "status": "success",
                "message": "تنظیمات به‌روز شد",
                "new_settings": {
                    "trading_hours": SystemConfig.TRADING_HOURS,
                    "check_interval": SystemConfig.CHECK_INTERVAL,
                    "min_score": SystemConfig.MIN_SCORE,
                    "risk_free_enabled": SystemConfig.RISK_FREE_ENABLED
                }
            })
        except Exception as e:
            return jsonify({"status": "error", "error": str(e)}), 400

# ۷. شروع برنامه
if __name__ == "__main__":
    # بارگذاری تاریخچه
    load_signal_history()
    
    # اجرای ترد زمان‌بندی
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    
    # اجرای ترد پایش قیمت
    monitor_thread = threading.Thread(target=check_targets, daemon=True)
    monitor_thread.start()
    
    # اطلاعات راه‌اندازی
    print("\n" + "="*50)
    print("🚀 Crypto Trading Bot v2.0")
    print("="*50)
    print(f"📅 تاریخ: {get_iran_time().strftime('%Y-%m-%d')}")
    print(f"⏰ ساعت: {get_iran_time().strftime('%H:%M:%S')}")
    print(f"📊 واچ‌لیست: {', '.join(WATCHLIST)}")
    print(f"⚙️ ساعت معاملاتی: {SystemConfig.TRADING_HOURS[0]}:00 - {SystemConfig.TRADING_HOURS[1]}:00")
    print(f"📈 حداقل امتیاز سیگنال: {SystemConfig.MIN_SCORE}")
    print(f"🔄 فاصله بررسی: هر {SystemConfig.CHECK_INTERVAL} ثانیه")
    print("="*50)
    
    # ذخیره خودکار تاریخچه هنگام خروج
    import atexit
    atexit.register(save_signal_history)
    atexit.register(lambda: print("\n👋 سیستم در حال خاموش شدن..."))
    
    print(f"🌐 سرور در حال راه‌اندازی روی پورت {port}...")
    print(f"📊 API در دسترس: http://localhost:{port}")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=False)
