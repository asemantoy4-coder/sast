import time
import json
from datetime import datetime, timedelta
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
            'today_signals': 0,
            'last_24h_profit_potential': 0
        }
        
        # تشخیص سیگنال‌های تکراری
        self.signal_fingerprints = set()
        
        # مدیریت فرکانس سیگنال با getattr برای جلوگیری از AttributeError
        self.signal_burst_protection = {
            'count': 0,
            'window_start': time.time(),
            'max_per_hour': getattr(config, 'MAX_SIGNALS_PER_HOUR', 12)
        }
    
    def calculate_signal_quality_score(self, analysis):
        """محاسبه کیفیت سیگنال با امتیازدهی چند بعدی"""
        score_weights = {
            'base_score': 0.4,
            'volume_confirmation': 0.2,
            'multi_timeframe_alignment': 0.15,
            'risk_reward_ratio': 0.15,
            'market_context': 0.1
        }
        
        quality_score = analysis.get('score', 0) * score_weights['base_score']
        
        # تأیید حجم
        if analysis.get('volume_profile', {}).get('volume_confirmation', False):
            quality_score += 2 * score_weights['volume_confirmation']
        
        # هماهنگی چند تایم‌فریم
        if getattr(config, 'ENABLE_MULTI_TF_FILTER', True) and self.check_multi_timeframe_alignment():
            quality_score += 1.5 * score_weights['multi_timeframe_alignment']
        
        # نسبت ریسک به ریوارد
        current_price = analysis.get('price', 0)
        if current_price > 0:
            stop_loss = current_price * 0.995 if analysis.get('score', 0) >= 3.5 else current_price * 1.005
            rr_ratio = abs(current_price - stop_loss) / current_price * 100
            
            if rr_ratio < 1.0:
                quality_score += 2 * score_weights['risk_reward_ratio']
        
        # بررسی شرایط بازار
        market_context = analysis.get('market_regime', {})
        if market_context.get('scalp_safe', False) and market_context.get('volatility', 'NORMAL') == 'NORMAL':
            quality_score += 1 * score_weights['market_context']
        
        return min(10, max(0, quality_score))
    
    def check_multi_timeframe_alignment(self):
        """بررسی هماهنگی سیگنال در تایم‌فریم‌های مختلف"""
        timeframes = ['5m', '15m', '1h']
        aligned_count = 0
        
        for tf in timeframes:
            try:
                df_tf = DataHandler.fetch_data(self.symbol, tf, limit=50)
                if not df_tf.empty and len(df_tf) > 20:
                    analysis_tf = utils.generate_scalp_signals(df_tf)
                    if abs(analysis_tf.get('score', 0)) >= 2:
                        aligned_count += 1
            except Exception as e:
                print(f"⚠️ Multi TF check failed for {tf}: {e}")
                continue
        
        return aligned_count >= 2
    
    def create_signal_fingerprint(self, analysis, side):
        """ایجاد اثر انگشت منحصربه‌فرد برای جلوگیری از ارسال سیگنال‌های تکراری"""
        signal_data = f"{side}_{analysis.get('price', 0):.4f}_{analysis.get('score', 0):.1f}"
        
        key_features = [
            analysis.get('market_regime', {}).get('regime', ''),
            analysis.get('volume_profile', {}).get('current_zone', ''),
            datetime.now().strftime('%Y%m%d%H')
        ]
        signal_data += '_'.join(key_features)
        
        return hashlib.md5(signal_data.encode()).hexdigest()
    
    def check_burst_protection(self):
        """جلوگیری از ارسال سیگنال‌های پی در پی"""
        current_time = time.time()
        
        # بازنشانی شمارنده هر ساعت
        if current_time - self.signal_burst_protection['window_start'] > 3600:
            self.signal_burst_protection = {
                'count': 0,
                'window_start': current_time,
                'max_per_hour': getattr(config, 'MAX_SIGNALS_PER_HOUR', 12)
            }
        
        # بررسی حد مجاز
        if self.signal_burst_protection['count'] >= self.signal_burst_protection['max_per_hour']:
            wait_time = 3600 - (current_time - self.signal_burst_protection['window_start'])
            print(f"⏳ Burst protection: Max signals per hour reached. Waiting {wait_time/60:.0f} minutes.")
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
            return False
        
        # ۵. بررسی شرایط ویژه بازار
        market_regime = analysis.get('market_regime', {})
        if getattr(config, 'ENABLE_MARKET_REGIME_FILTER', True):
            if market_regime.get('regime') == 'DANGEROUS' or not market_regime.get('scalp_safe', False):
                return False
        
        # ۶. بررسی نوسان بیش از حد
        if analysis.get('volatility', 0) > 5:
            return False
        
        return True
    
    def send_signal(self, analysis, side):
        """ارسال سیگنال با فرمت حرفه‌ای"""
        current_price = analysis.get('price', 0)
        if current_price == 0:
            print("⚠️ Invalid price for signal")
            return
        
        # محاسبه سطوح به صورت داینامیک
        if "BUY" in side:
            stop_loss = min(
                current_price * 0.995,
                analysis.get('volume_profile', {}).get('val', current_price * 0.99)
            )
            stop_loss_pct = ((current_price - stop_loss) / current_price) * 100
        else:
            stop_loss = max(
                current_price * 1.005,
                analysis.get('volume_profile', {}).get('vah', current_price * 1.01)
            )
            stop_loss_pct = ((stop_loss - current_price) / current_price) * 100
        
        # دریافت سطوح خروج
        try:
            exits = utils.get_exit_levels(current_price, stop_loss)
        except:
            exits = {
                'tp1': current_price * (1.01 if "BUY" in side else 0.99),
                'tp2': current_price * (1.02 if "BUY" in side else 0.98)
            }
        
        # محاسبه نسبت ریسک به ریوارد
        if "BUY" in side:
            rr_ratio = (exits.get('tp2', current_price * 1.02) - current_price) / (current_price - stop_loss)
        else:
            rr_ratio = (current_price - exits.get('tp2', current_price * 0.98)) / (stop_loss - current_price)
        
        # ایجاد امضای سیگنال
        signal_id = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # انتخاب ایموجی بر اساس کیفیت
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
        
        # ساخت پیام حرفه‌ای
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
• Regime: {analysis.get('market_regime', {}).get('regime', 'N/A')}
• Trend: {analysis.get('market_regime', {}).get('direction', 'N/A')}
• VP Zone: {analysis.get('volume_profile', {}).get('current_zone', 'N/A')}
• Ichimoku: {analysis.get('ichimoku', {}).get('trend', 'N/A')}
─────────────────────────────
🔍 *KEY REASONS:*
{chr(10).join(['• ' + reason for reason in analysis.get('reasons', ['No reasons provided'])[:3]])}
─────────────────────────────
📡 *Signal Provider:* @AsemanSignals
⚠️ *Disclaimer:* Trading involves risk. Use proper risk management.
"""
        
        try:
            # ارسال به تلگرام
            if hasattr(utils, 'send_telegram_notification'):
                utils.send_telegram_notification(msg)
            else:
                print("⚠️ Telegram notification function not found")
        except Exception as e:
            print(f"⚠️ Failed to send Telegram notification: {e}")
        
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
            'market_regime': analysis.get('market_regime', {})
        })
        
        # آپدیت آمار
        self.update_stats(side, analysis)
        
        # ثبت در تاریخچه
        fingerprint = self.create_signal_fingerprint(analysis, side)
        self.signal_fingerprints.add(fingerprint)
        self.last_signal_time = time.time()
        
        print(f"\n✅ Signal #{signal_id} sent to @AsemanSignals")
        print(f"   Quality: {quality_score:.1f}/10 | RR: 1:{max(1.0, rr_ratio):.1f}")
    
    def save_signal_to_file(self, signal_data):
        """ذخیره سیگنال در فایل JSON برای رکورد"""
        try:
            filename = f"signals_{datetime.now().strftime('%Y%m%d')}.json"
            
            # بارگذاری سیگنال‌های قبلی
            try:
                with open(filename, 'r') as f:
                    signals = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                signals = []
            
            # اضافه کردن سیگنال جدید
            signals.append(signal_data)
            
            # ذخیره
            with open(filename, 'w') as f:
                json.dump(signals, f, indent=2, default=str)
                
        except Exception as e:
            print(f"⚠️ Could not save signal to file: {e}")
    
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
        market_regime = analysis.get('market_regime', {})
        current_score = analysis.get('score', 0)
        
        # رنگ‌بندی بر اساس کیفیت
        if quality_score >= 8:
            score_color = "\033[92m"  # سبز
        elif quality_score >= 6:
            score_color = "\033[93m"  # زرد
        else:
            score_color = "\033[91m"  # قرمز
        
        # پاک کردن خط قبلی
        print("\033[K", end="")
        
        # نمایش داشبورد
        interval = getattr(config, 'INTERVAL', '5m')
        dashboard = f"""
╔══════════════════════════════════════════════════════════════╗
║ 🚀 ASEMAN SIGNAL BOT v2.0                   {datetime.now().strftime('%H:%M:%S')} ║
╠══════════════════════════════════════════════════════════════╣
║ 📊 SYMBOL: {self.symbol:<12} TF: {interval:<5} PRICE: {analysis.get('price', 0):<10.4f} ║
║ 🎯 SCORE: {score_color}{current_score:<5.1f}\033[0m | QUALITY: {score_color}{quality_score:<5.1f}/10\033[0m | REGIME: {market_regime.get('regime', 'N/A'):<10} ║
║ 📈 TREND: {market_regime.get('direction', 'N/A'):<8} | SAFE: {'✅' if market_regime.get('scalp_safe') else '❌':<3} | VP: {analysis.get('volume_profile', {}).get('current_zone', 'N/A'):<8} ║
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
                    print(f"\n⚠️ Analysis Error: {e}")
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