import time
import json
from datetime import datetime
from exchange_handler import DataHandler
import utils
import config
import pandas as pd
from collections import deque
import hashlib

class AsemanSignalBot:
    def __init__(self):
        self.symbol = getattr(config, 'SYMBOL', 'BTC/USDT')
        self.signals_log = deque(maxlen=50)
        self.signal_cooldown = getattr(config, 'SIGNAL_COOLDOWN', 300)
        self.last_signal_time = 0
        
        self.signal_stats = {
            'total_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'today_signals': 0
        }
        
        self.signal_fingerprints = set()
        
        self.signal_burst_protection = {
            'count': 0,
            'window_start': time.time(),
            'max_per_hour': getattr(config, 'MAX_SIGNALS_PER_HOUR', 12)
        }

    def calculate_signal_quality_score(self, analysis):
        """
        محاسبه کیفیت سیگنال با امتیازدهی چند بعدی (نسخه اصلاح شده)
        این تابع عمق تحلیل را بررسی می‌کند تا سیگنال‌های ضعیف فیلتر شوند.
        """
        score_weights = {
            'base_score': 0.4,
            'volume_confirmation': 0.2,
            'multi_timeframe_alignment': 0.15,
            'risk_reward_ratio': 0.15,
            'market_context': 0.1
        }
        
        quality_score = analysis.get('score', 0) * score_weights['base_score']
        
        # --- دسترسی ایمن به تحلیل‌ها ---
        inner = analysis.get('analysis', {})
        volume_profile = inner.get('volume_profile', {})
        
        # 1. تأیید حجم (اصلاح شده: استفاده از کلیدهای صحیح utils.py)
        # در نسخه قبلی شما کلید volume_confirmation وجود نداشت، ما از in_value_area استفاده می‌کنیم
        if volume_profile.get('in_value_area', False):
            quality_score += 2 * score_weights['volume_confirmation']
        
        # 2. هماهنگی چند تایم‌فریم (بسیار مهم برای اسکالپ)
        if getattr(config, 'ENABLE_MULTI_TF_FILTER', True):
            if self.check_multi_timeframe_alignment():
                quality_score += 1.5 * score_weights['multi_timeframe_alignment']
        
        # 3. نسبت ریسک به ریوارد (تخمینی بر اساس نوسان)
        current_price = analysis.get('price', 0)
        if current_price > 0:
            market_regime = inner.get('market_regime', {})
            atr = market_regime.get('atr_percent', 1.0)
            
            # اگر ATR مناسب برای اسکالپ باشد (نه خیلی کم که در کارمزار بخورد، نه خیلی زیاد)
            if 0.3 <= atr <= 1.5: 
                quality_score += 2 * score_weights['risk_reward_ratio']
            elif atr > 2.0: # نوسان خیلی زیاد خطرناک است
                quality_score -= 1
        
        # 4. بررسی شرایط بازار
        market_regime = inner.get('market_regime', {})
        if market_regime.get('scalp_safe', False):
            quality_score += 1 * score_weights['market_context']
        
        return min(10, max(0, quality_score))
    
    def check_multi_timeframe_alignment(self):
        """
        بررسی هماهنگی سیگنال در تایم‌فریم‌های مختلف (5m, 15m, 1h)
        توجه: این بخش باعث ایجاد 2 درخواست API اضافه می‌شود.
        """
        timeframes = ['15m', '1h'] # 5m در حلقه اصلی بررسی می‌شود، نیازی به تکرار نیست
        aligned_count = 0
        
        # برای جلوگیری از کند شدن زیاد، فقط وقتی امتیاز پایین باشد چک نکنیم؟ 
        # خیر، برای کیفیت باید همیشه چک شود.
        
        for tf in timeframes:
            try:
                # دریافت دیتا برای تایم‌فریم بالاتر
                df_tf = DataHandler.fetch_data(self.symbol, tf, limit=50)
                if not df_tf.empty and len(df_tf) > 20:
                    analysis_tf = utils.generate_scalp_signals(df_tf)
                    # اگر روند در تایم بالاتر هم جهت اصلی باشد، امتیاز بده
                    score_tf = analysis_tf.get('score', 0)
                    # اگر جهت پوزیشن ما با جهت تایم بالاتر یکی باشد
                    # فرض: این تابع وقتی صدا زده می‌شود که ما یک پتانسیل سیگنال داریم
                    if abs(score_tf) >= 2.0: 
                        aligned_count += 1
            except Exception as e:
                print(f"⚠️ Multi TF check failed for {tf}: {e}")
                continue
        
        # اگر حداقل 1 تایم‌فریم بالاتر همسو بود (یا خنثی نبود)
        return aligned_count >= 1
    
    def create_signal_fingerprint(self, analysis, side):
        """ایجاد اثر انگشت منحصربه‌فرد برای جلوگیری از ارسال سیگنال‌های تکراری"""
        signal_data = f"{side}_{analysis.get('price', 0):.4f}_{analysis.get('score', 0):.1f}"
        
        # استفاده از کلیدهای صحیح nested dictionary
        inner = analysis.get('analysis', {})
        key_features = [
            inner.get('market_regime', {}).get('regime', ''),
            inner.get('volume_profile', {}).get('current_zone', ''),
            datetime.now().strftime('%Y%m%d%H')
        ]
        signal_data += '_'.join(key_features)
        
        return hashlib.md5(signal_data.encode()).hexdigest()
    
    def check_burst_protection(self):
        """جلوگیری از ارسال سیگنال‌های پی در پی (Spam)"""
        current_time = time.time()
        
        if current_time - self.signal_burst_protection['window_start'] > 3600:
            self.signal_burst_protection = {
                'count': 0,
                'window_start': current_time,
                'max_per_hour': getattr(config, 'MAX_SIGNALS_PER_HOUR', 12)
            }
        
        if self.signal_burst_protection['count'] >= self.signal_burst_protection['max_per_hour']:
            wait_time = 3600 - (current_time - self.signal_burst_protection['window_start'])
            print(f"⏳ Burst protection active. Wait {wait_time/60:.0f} min.")
            return False
        
        self.signal_burst_protection['count'] += 1
        return True
    
    def should_send_signal(self, analysis, side):
        """بررسی جامع شرایط ارسال سیگنال"""
        current_time = time.time()
        
        # ۱. بررسی کول‌داون
        if current_time - self.last_signal_time < self.signal_cooldown:
            return False
        
        # ۲. بررسی محافظت در برابر تراکم سیگنال
        if not self.check_burst_protection():
            return False
        
        # ۳. بررسی تکراری نبودن سیگنال
        fingerprint = self.create_signal_fingerprint(analysis, side)
        if fingerprint in self.signal_fingerprints:
            return False
        
        # ۴. بررسی کیفیت سیگنال
        quality_score = self.calculate_signal_quality_score(analysis)
        min_quality = getattr(config, 'MIN_SIGNAL_QUALITY', 7.0)
        
        if quality_score < min_quality:
            # print(f"❌ Quality Low: {quality_score:.1f} < {min_quality}") # برای دیباگ
            return False
        
        # ۵. بررسی شرایط ویژه بازار
        inner = analysis.get('analysis', {})
        market_regime = inner.get('market_regime', {})
        
        if getattr(config, 'ENABLE_MARKET_REGIME_FILTER', True):
            # اگر رژیم بازار خطرناک باشد، سیگنال نده
            if market_regime.get('regime') in ['DANGEROUS', 'HIGH_VOLATILITY', 'DEAD']:
                return False
        
        # ۶. بررسی نوسان (Volatility)
        volatility = market_regime.get('volatility', 0)
        if volatility > 0.05: # نوسان لحظه‌ای بالای 5 درصد خطرناک است
            return False
            
        return True
    
    def send_signal(self, analysis, side):
        """ارسال سیگنال با فرمت حرفه‌ای"""
        current_price = analysis.get('price', 0)
        if current_price == 0:
            return
        
        # استخراج آبجکت‌های تحلیل به صورت ایمن
        inner = analysis.get('analysis', {})
        volume_profile = inner.get('volume_profile', {})
        market_regime = inner.get('market_regime', {})
        ichimoku = inner.get('ichimoku', {})
        
        # تعیین جهت تمیز (بدون ایموجی)
        clean_side = "BUY" if "BUY" in side else "SELL"
        
        # محاسبه سطوح به صورت داینامیک
        if clean_side == "BUY":
            stop_loss = min(
                current_price * 0.995, # حد ضرر استاندارد
                volume_profile.get('val', current_price * 0.99) # حد ضرر بر اساس حجم (ارزش پایین)
            )
        else:
            stop_loss = max(
                current_price * 1.005, # حد ضرر استاندارد
                volume_profile.get('vah', current_price * 1.01) # حد ضرر بر اساس حجم (ارزش بالا)
            )
        
        stop_loss_pct = ((stop_loss - current_price) / current_price) * 100
        if clean_side == "BUY": stop_loss_pct *= -1 # نمایش درصد منفی برای خرید
        
        # دریافت سطوح خروج (اصلاح شده: ارسال جهت)
        try:
            exits = utils.get_exit_levels(current_price, stop_loss, direction=clean_side)
        except Exception as e:
            print(f"Error calculating exits: {e}")
            exits = {
                'tp1': current_price * (1.01 if clean_side == "BUY" else 0.99),
                'tp2': current_price * (1.02 if clean_side == "BUY" else 0.98)
            }
        
        # محاسبه نسبت ریسک به ریوارد
        if clean_side == "BUY":
            rr_ratio = (exits.get('tp2', current_price * 1.02) - current_price) / (current_price - stop_loss)
        else:
            rr_ratio = (current_price - exits.get('tp2', current_price * 0.98)) / (stop_loss - current_price)
        
        signal_id = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # محاسبه کیفیت نهایی برای نمایش
        quality_score = self.calculate_signal_quality_score(analysis)
        if quality_score >= 9:
            emoji = "🔥"
            strength = "STRONG"
        elif quality_score >= 7:
            emoji = "⚡"
            strength = "MEDIUM"
        else:
            emoji = "📊"
            strength = "WEAK"
        
        # ساخت پیام حرفه‌ای (تمام کلیدها اصلاح شده‌اند)
        msg = f"""
{emoji} *ASEMAN SIGNAL #{signal_id}* {emoji}
─────────────────────────────
🎯 *PAIR:* {self.symbol}
⏰ *TIME:* {datetime.now().strftime('%H:%M:%S')}
📶 *STRENGTH:* {strength} ({quality_score:.1f}/10)
─────────────────────────────
📊 *SIGNAL TYPE:* {side}
💰 *ENTRY PRICE:* {current_price:.4f}
─────────────────────────────
🎯 *TAKE PROFIT LEVELS:*
TP¹: `{exits.get('tp1', 0):.4f}` (+{abs(exits.get('tp1', current_price)-current_price)/current_price*100:.2f}%)
TP²: `{exits.get('tp2', 0):.4f}` (+{abs(exits.get('tp2', current_price)-current_price)/current_price*100:.2f}%)
─────────────────────────────
🛑 *STOP LOSS:* `{stop_loss:.4f}` ({stop_loss_pct:.2f}%)
📈 *RISK/REWARD:* 1:{max(1.0, rr_ratio):.1f}
─────────────────────────────
📊 *MARKET ANALYSIS:*
• Regime: {market_regime.get('regime', 'N/A')}
• Trend: {market_regime.get('direction', 'N/A')}
• VP Zone: {volume_profile.get('current_zone', 'N/A')}
• Ichimoku: {ichimoku.get('trend', 'N/A')}
─────────────────────────────
🔍 *KEY REASONS:*
{chr(10).join(['• ' + reason for reason in analysis.get('reasons', ['No reasons provided'])[:3]])}
─────────────────────────────
📡 *Signal Provider:* @AsemanSignals
⚠️ *Disclaimer:* Trading involves risk. Use proper risk management.
"""
        
        try:
            if hasattr(utils, 'send_telegram_notification'):
                utils.send_telegram_notification(msg, signal_type=clean_side)
        except Exception as e:
            print(f"⚠️ Failed to send Telegram: {e}")
        
        # ذخیره در فایل برای بک‌آپ
        self.save_signal_to_file({
            'id': signal_id,
            'timestamp': datetime.now().isoformat(),
            'symbol': self.symbol,
            'side': side,
            'entry': current_price,
            'tp1': exits.get('tp1'),
            'tp2': exits.get('tp2'),
            'sl': stop_loss,
            'quality_score': quality_score,
            'reasons': analysis.get('reasons', []),
            'market_regime': market_regime
        })
        
        # آپدیت آمار
        self.update_stats(side, analysis)
        
        # ثبت در تاریخچه
        fingerprint = self.create_signal_fingerprint(analysis, side)
        self.signal_fingerprints.add(fingerprint)
        self.last_signal_time = time.time()
        
        print(f"\n✅ Signal #{signal_id} sent | Quality: {quality_score:.1f}/10 | RR: 1:{max(1.0, rr_ratio):.1f}")
    
    def save_signal_to_file(self, signal_data):
        """ذخیره سیگنال در فایل JSON"""
        try:
            filename = f"signals_{datetime.now().strftime('%Y%m%d')}.json"
            try:
                with open(filename, 'r') as f:
                    signals = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                signals = []
            
            signals.append(signal_data)
            
            with open(filename, 'w') as f:
                json.dump(signals, f, indent=2, default=str)
                
        except Exception as e:
            print(f"⚠️ File Save Error: {e}")
    
    def update_stats(self, side, analysis):
        """آپدیت آمار سیگنال‌ها"""
        self.signal_stats['total_signals'] += 1
        self.signal_stats['today_signals'] += 1
        
        if "BUY" in side:
            self.signal_stats['buy_signals'] += 1
        else:
            self.signal_stats['sell_signals'] += 1
    
    def display_dashboard(self, analysis):
        """نمایش داشبورد زیبا در کنسول"""
        quality_score = self.calculate_signal_quality_score(analysis)
        
        # دسترسی صحیح به داده‌های تو در تو
        inner = analysis.get('analysis', {})
        market_regime = inner.get('market_regime', {})
        current_score = analysis.get('score', 0)
        
        # رنگ‌بندی بر اساس کیفیت
        if quality_score >= 8:
            score_color = "\033[92m"  # سبز
        elif quality_score >= 6:
            score_color = "\033[93m"  # زرد
        else:
            score_color = "\033[91m"  # قرمز
        
        # حذف خط قبلی
        print("\033[K", end="")
        
        interval = getattr(config, 'INTERVAL', '5m')
        dashboard = f"""
╔══════════════════════════════════════════════════════════════╗
║ 🚀 ASEMAN SIGNAL BOT v2.0                   {datetime.now().strftime('%H:%M:%S')} ║
╠══════════════════════════════════════════════════════════════╣
║ 📊 SYMBOL: {self.symbol:<12} TF: {interval:<5} PRICE: {analysis.get('price', 0):<10.4f} ║
║ 🎯 SCORE: {score_color}{current_score:<5.1f}\033[0m | QUALITY: {score_color}{quality_score:<5.1f}/10\033[0m | REGIME: {market_regime.get('regime', 'N/A'):<10} ║
║ 📈 TREND: {market_regime.get('direction', 'N/A'):<8} | SAFE: {'✅' if market_regime.get('scalp_safe') else '❌':<3} | VP: {inner.get('volume_profile', {}).get('current_zone', 'N/A'):<8} ║
║ 📡 SIGNALS Today: {self.signal_stats['today_signals']:<3} | Total: {self.signal_stats['total_signals']:<4} | B:{self.signal_stats['buy_signals']}/S:{self.signal_stats['sell_signals']} ║
╚══════════════════════════════════════════════════════════════╝
"""
        print(dashboard, end="\r")
    
    def run(self):
        """حلقه اصلی اجرای ربات"""
        interval = getattr(config, 'INTERVAL', '5m')
        print(f"""
╔══════════════════════════════════════════════════╗
║        🚀 ASEMAN SIGNAL BOT v2.0                ║
║        📡 Scanning: {self.symbol:<15}         ║
║        ⚡ Interval: {interval:<5}                    ║
╚══════════════════════════════════════════════════╝
        """)
        
        try:
            while True:
                try:
                    # ۱. دریافت داده‌ها
                    df = DataHandler.fetch_data(self.symbol, getattr(config, 'INTERVAL', '5m'), limit=100)
                    
                    if df.empty or len(df) < 20:
                        time.sleep(getattr(config, 'SCALP_INTERVAL', 10))
                        continue
                    
                    # ۲. تحلیل بازار
                    analysis = utils.generate_scalp_signals(df)
                    
                    # ۳. نمایش داشبورد
                    self.display_dashboard(analysis)
                    
                    # ۴. بررسی و ارسال سیگنال خرید
                    if analysis.get('score', 0) >= 3.5:
                        if self.should_send_signal(analysis, "🟢 BUY"):
                            self.send_signal(analysis, "🟢 BUY")
                    
                    # ۵. بررسی و ارسال سیگنال فروش
                    elif analysis.get('score', 0) <= -3.5:
                        if self.should_send_signal(analysis, "🔴 SELL"):
                            self.send_signal(analysis, "🔴 SELL")
                    
                    # ۶. خواب کنترل‌شده
                    sleep_time = getattr(config, 'SCALP_INTERVAL', 10)
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    print(f"\n⚠️ Loop Error: {e}")
                    time.sleep(5)
                    
        except KeyboardInterrupt:
            print("\n\n🛑 Bot stopped by user")
            self.display_final_stats()
    
    def display_final_stats(self):
        """نمایش آمار نهایی"""
        print(f"""
{'='*60}
📊 FINAL STATISTICS
{'='*60}
Total Signals Generated: {self.signal_stats['total_signals']}
Buy Signals: {self.signal_stats['buy_signals']}
Sell Signals: {self.signal_stats['sell_signals']}
Today's Signals: {self.signal_stats['today_signals']}
{'='*60}
📡 @AsemanSignals
        """)

if __name__ == "__main__":
    bot = AsemanSignalBot()
    bot.run()
```