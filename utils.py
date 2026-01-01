import pandas as pd
import numpy as np
import requests
import config
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ==================== TELEGRAM NOTIFICATION ====================
def send_telegram_notification(message: str, signal_type: str = "INFO") -> bool:
    """
    ارسال سیگنال به کانال تلگرام با فرمت حرفه‌ای
    
    Parameters:
    -----------
    message : str
        متن پیام
    signal_type : str
        نوع سیگنال: "BUY", "SELL", "ALERT", "INFO", "ERROR"
    
    Returns:
    --------
    bool: موفقیت ارسال
    """
    try:
        # فرمت‌بندی پیام بر اساس نوع سیگنال
        emoji_map = {
            "BUY": "🟢",
            "SELL": "🔴", 
            "ALERT": "⚠️",
            "INFO": "ℹ️",
            "ERROR": "❌"
        }
        
        emoji = emoji_map.get(signal_type, "📊")
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        formatted_message = f"{emoji} *{signal_type}* [{timestamp}]\n{message}"
        
        url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": config.TELEGRAM_CHAT_ID,
            "text": formatted_message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }
        
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
        
    except Exception as e:
        print(f"❌ Telegram Error: {e}")
        return False

# ==================== VOLUME PROFILE ADVANCED ====================
def get_pro_volume_profile(df: pd.DataFrame, bins: int = 100) -> Dict:
    """
    محاسبه پیشرفته Volume Profile برای اسکالپ
    
    Returns:
    --------
    Dict شامل:
        - poc: نقطه کنترل اصلی
        - vah: سقف ناحیه ارزش (Value Area High)
        - val: کف ناحیه ارزش (Value Area Low)
        - current_zone: NEUTRAL/CHEAP/EXPENSIVE
        - volume_density: توزیع حجم
        - high_volume_nodes: گره‌های پرحجم
    """
    try:
        if len(df) < bins:
            return {"poc": 0, "vah": 0, "val": 0, "current_zone": "NEUTRAL"}
        
        # 1. ایجاد سطوح قیمتی
        price_min = df['Low'].min()
        price_max = df['High'].max()
        
        if price_max <= price_min:
            return {"poc": 0, "vah": 0, "val": 0, "current_zone": "NEUTRAL"}
        
        price_levels = np.linspace(price_min, price_max, bins)
        bin_width = (price_max - price_min) / (bins - 1)
        
        # 2. تخصیص حجم به هر سطح
        volumes = np.zeros(bins - 1)
        price_vol_map = []
        
        for idx in range(len(df)):
            close_price = df['Close'].iloc[idx]
            volume = df['Volume'].iloc[idx] if 'Volume' in df.columns else 0
            
            # پیدا کردن بین مناسب
            bin_idx = int((close_price - price_min) // bin_width)
            bin_idx = max(0, min(bin_idx, len(volumes) - 1))
            
            volumes[bin_idx] += volume
            price_vol_map.append((close_price, volume))
        
        # 3. پیدا کردن POC
        poc_idx = np.argmax(volumes)
        poc_price = price_levels[poc_idx] + (bin_width / 2)
        
        # 4. محاسبه Value Area (70%)
        total_volume = np.sum(volumes)
        target_va_volume = total_volume * 0.70
        
        # گسترش از POC به بیرون
        low_idx, high_idx = poc_idx, poc_idx
        current_va_volume = volumes[poc_idx]
        
        while current_va_volume < target_va_volume and (low_idx > 0 or high_idx < len(volumes) - 1):
            # اولویت به سمت با حجم‌تر
            left_vol = volumes[low_idx - 1] if low_idx > 0 else 0
            right_vol = volumes[high_idx + 1] if high_idx < len(volumes) - 1 else 0
            
            if left_vol >= right_vol and low_idx > 0:
                low_idx -= 1
                current_va_volume += volumes[low_idx]
            elif high_idx < len(volumes) - 1:
                high_idx += 1
                current_va_volume += volumes[high_idx]
            else:
                break
        
        vah_price = price_levels[high_idx] + bin_width
        val_price = price_levels[low_idx]
        
        # 5. تعیین ناحیه فعلی
        current_price = df['Close'].iloc[-1]
        current_zone = "NEUTRAL"
        
        if current_price < val_price:
            current_zone = "CHEAP"
        elif current_price > vah_price:
            current_zone = "EXPENSIVE"
        
        # 6. تشخیص گره‌های پرحجم
        volume_threshold = np.percentile(volumes[volumes > 0], 75) if len(volumes[volumes > 0]) > 0 else 0
        high_volume_nodes = []
        
        for i, vol in enumerate(volumes):
            if vol > volume_threshold:
                node_price = price_levels[i] + (bin_width / 2)
                high_volume_nodes.append({
                    "price": float(node_price),
                    "volume": float(vol),
                    "strength": float(vol / total_volume * 100)
                })
        
        return {
            "poc": float(poc_price),
            "vah": float(vah_price),
            "val": float(val_price),
            "current_zone": current_zone,
            "current_price": float(current_price),
            "value_area_range": float(vah_price - val_price),
            "volume_distribution": volumes.tolist(),
            "high_volume_nodes": sorted(high_volume_nodes, key=lambda x: x["strength"], reverse=True)[:5],
            "poc_strength": float(volumes[poc_idx] / total_volume * 100),
            "in_value_area": val_price <= current_price <= vah_price
        }
        
    except Exception as e:
        print(f"❌ Volume Profile Error: {e}")
        return {"poc": 0, "vah": 0, "val": 0, "current_zone": "NEUTRAL", "error": str(e)}

# ==================== MARKET REGIME DETECTION ====================
def detect_market_regime(df: pd.DataFrame, window: int = 50) -> Dict:
    """
    تشخیص رژیم بازار با فیلترهای اسکالپ
    """
    try:
        if len(df) < window:
            return {"regime": "INSUFFICIENT_DATA", "scalp_safe": False, "direction": "NEUTRAL"}
        
        # 1. محاسبه نوسان
        returns = df['Close'].pct_change().dropna()
        volatility = returns.rolling(window=window).std()
        current_volatility = volatility.iloc[-1]
        
        # 2. محاسبه ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        atr_percent = (atr / df['Close']) * 100
        current_atr_pct = atr_percent.iloc[-1]
        
        # 3. تشخیص روند با ADX ساده‌شده
        # اگر قیمت بالاتر از SMA50 باشد، روند صعودی
        sma_50 = df['Close'].rolling(window=50).mean()
        sma_20 = df['Close'].rolling(window=20).mean()
        
        current_price = df['Close'].iloc[-1]
        
        # تعیین جهت
        if current_price > sma_50.iloc[-1] and sma_20.iloc[-1] > sma_50.iloc[-1]:
            direction = "BULLISH"
        elif current_price < sma_50.iloc[-1] and sma_20.iloc[-1] < sma_50.iloc[-1]:
            direction = "BEARISH"
        else:
            direction = "SIDEWAYS"
        
        # 4. شرایط اسکالپ امن
        scalp_safe = True
        
        # فیلتر نوسان: برای اسکالپ نیاز به نوسان متوسط داریم
        if current_volatility < 0.001:  # نوسان خیلی کم
            scalp_safe = False
            regime = "DEAD_MARKET"
        elif current_volatility > 0.02:  # نوسان خیلی زیاد
            scalp_safe = False
            regime = "VOLATILE"
        elif current_atr_pct > 2.0:  # ATR بیش از 2%
            scalp_safe = False
            regime = "HIGH_VOLATILITY"
        elif direction == "SIDEWAYS":
            regime = "RANGING"
            # در رنج هم می‌توان اسکالپ کرد
        else:
            regime = "TRENDING"
        
        # 5. فیلتر حجم
        if 'Volume' in df.columns:
            avg_volume = df['Volume'].rolling(window=20).mean().iloc[-1]
            current_volume = df['Volume'].iloc[-1]
            
            if current_volume < avg_volume * 0.5:  # حجم کمتر از نصف میانگین
                scalp_safe = False
                regime = "LOW_LIQUIDITY"
        
        return {
            "regime": regime,
            "scalp_safe": scalp_safe,
            "direction": direction,
            "volatility": float(current_volatility),
            "atr_percent": float(current_atr_pct),
            "price_vs_sma50": float((current_price / sma_50.iloc[-1] - 1) * 100),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Market Regime Error: {e}")
        return {"regime": "ERROR", "scalp_safe": False, "direction": "NEUTRAL"}

# ==================== ICHIMOKU ANALYSIS ====================
def get_ichimoku(df: pd.DataFrame) -> Dict:
    """
    تحلیل ایچیموکو بهینه‌شده برای اسکالپ
    """
    try:
        if len(df) < 52:
            return {"trend": "NEUTRAL", "price_above_cloud": False, "signal": "NO_DATA"}
        
        # محاسبه اجزای اصلی
        high_9 = df['High'].rolling(window=9, min_periods=1).max()
        low_9 = df['Low'].rolling(window=9, min_periods=1).min()
        tenkan = (high_9 + low_9) / 2
        
        high_26 = df['High'].rolling(window=26, min_periods=1).max()
        low_26 = df['Low'].rolling(window=26, min_periods=1).min()
        kijun = (high_26 + low_26) / 2
        
        high_52 = df['High'].rolling(window=52, min_periods=1).max()
        low_52 = df['Low'].rolling(window=52, min_periods=1).min()
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        senkou_b = ((high_52 + low_52) / 2).shift(26)
        
        current_price = df['Close'].iloc[-1]
        
        # تحلیل وضعیت
        price_above_cloud = current_price > max(senkou_a.iloc[-1], senkou_b.iloc[-1])
        price_below_cloud = current_price < min(senkou_a.iloc[-1], senkou_b.iloc[-1])
        
        # تعیین روند
        if tenkan.iloc[-1] > kijun.iloc[-1] and price_above_cloud:
            trend = "STRONG_BULLISH"
            signal = "BUY"
        elif tenkan.iloc[-1] > kijun.iloc[-1]:
            trend = "BULLISH"
            signal = "BUY"
        elif tenkan.iloc[-1] < kijun.iloc[-1] and price_below_cloud:
            trend = "STRONG_BEARISH"
            signal = "SELL"
        elif tenkan.iloc[-1] < kijun.iloc[-1]:
            trend = "BEARISH"
            signal = "SELL"
        else:
            trend = "NEUTRAL"
            signal = "HOLD"
        
        # محاسبه شیب برای قدرت روند
        if len(kijun) >= 5:
            kijun_slope = (kijun.iloc[-1] - kijun.iloc[-5]) / kijun.iloc[-5] * 100
        else:
            kijun_slope = 0
        
        return {
            "trend": trend,
            "signal": signal,
            "price_above_cloud": price_above_cloud,
            "price_below_cloud": price_below_cloud,
            "tenkan": float(tenkan.iloc[-1]),
            "kijun": float(kijun.iloc[-1]),
            "kijun_slope_pct": float(kijun_slope),
            "cloud_top": float(max(senkou_a.iloc[-1], senkou_b.iloc[-1])),
            "cloud_bottom": float(min(senkou_a.iloc[-1], senkou_b.iloc[-1])),
            "cloud_width_pct": float((senkou_a.iloc[-1] - senkou_b.iloc[-1]) / senkou_b.iloc[-1] * 100),
            "price_vs_tenkan": float((current_price / tenkan.iloc[-1] - 1) * 100),
            "price_vs_kijun": float((current_price / kijun.iloc[-1] - 1) * 100)
        }
        
    except Exception as e:
        print(f"❌ Ichimoku Error: {e}")
        return {"trend": "NEUTRAL", "price_above_cloud": False, "signal": "ERROR"}

# ==================== SCALP SIGNAL GENERATOR ====================
def generate_scalp_signals(df: pd.DataFrame) -> Dict:
    """
    تولید سیگنال اسکالپ نهایی با امتیازدهی هوشمند
    
    Returns:
    --------
    Dict شامل تمام تحلیل‌ها و امتیاز نهایی
    """
    try:
        if len(df) < 100:
            return {"score": 0, "signal": "INSUFFICIENT_DATA", "reasons": []}
        
        current_price = df['Close'].iloc[-1]
        
        # 1. جمع‌آوری تمام تحلیل‌ها
        volume_profile = get_pro_volume_profile(df)
        market_regime = detect_market_regime(df)
        ichimoku = get_ichimoku(df)
        
        # 2. محاسبه امتیاز پایه
        score = 0
        reasons = []
        
        # 2.1 امتیاز Volume Profile (حداکثر ۳ امتیاز)
        if volume_profile['current_zone'] == "CHEAP":
            score += 3
            reasons.append("قیمت در ناحیه ارزان حجمی")
        elif volume_profile['current_zone'] == "EXPENSIVE":
            score -= 3
            reasons.append("قیمت در ناحیه گران حجمی")
        
        if volume_profile.get('in_value_area', False):
            score += 1
            reasons.append("قیمت در Value Area")
        
        # 2.2 امتیاز Market Regime (حداکثر ۳ امتیاز)
        if market_regime['scalp_safe']:
            score += 2
            reasons.append("بازار برای اسکالپ امن است")
        
        if market_regime['direction'] == "BULLISH":
            score += 1
            reasons.append("روند صعودی")
        elif market_regime['direction'] == "BEARISH":
            score -= 1
            reasons.append("روند نزولی")
        
        # 2.3 امتیاز Ichimoku (حداکثر ۴ امتیاز)
        if ichimoku['signal'] == "BUY":
            score += 2
            reasons.append(f"سیگنال ایچیموکو: {ichimoku['trend']}")
        
        if ichimoku['price_above_cloud']:
            score += 2
            reasons.append("قیمت بالای ابر کومو")
        elif ichimoku['price_below_cloud']:
            score -= 2
            reasons.append("قیمت زیر ابر کومو")
        
        # 2.4 امتیاز حجم لحظه‌ای
        if 'Volume' in df.columns:
            avg_volume = df['Volume'].rolling(window=20).mean().iloc[-1]
            current_volume = df['Volume'].iloc[-1]
            
            if current_volume > avg_volume * 1.5:
                score += 2
                reasons.append("حجم معاملات بالا")
            elif current_volume < avg_volume * 0.5:
                score -= 1
                reasons.append("حجم معاملات پایین")
        
        # 2.5 امتیاز نوسان مناسب برای اسکالپ
        atr_percent = market_regime.get('atr_percent', 0)
        if 0.5 <= atr_percent <= 1.5:  # نوسان ایده‌آل برای اسکالپ
            score += 2
            reasons.append(f"نوسان مناسب ({atr_percent:.2f}%)")
        
        # 3. تعیین سیگنال نهایی
        signal = "HOLD"
        confidence = 0
        
        if score >= 7:
            signal = "STRONG_BUY"
            confidence = min(score / 10, 0.95)
        elif score >= 4:
            signal = "BUY"
            confidence = min(score / 10, 0.8)
        elif score <= -7:
            signal = "STRONG_SELL"
            confidence = min(abs(score) / 10, 0.95)
        elif score <= -4:
            signal = "SELL"
            confidence = min(abs(score) / 10, 0.8)
        else:
            signal = "HOLD"
            confidence = 0.5
        
        # 4. جمع‌بندی
        result = {
            "price": float(current_price),
            "score": float(score),
            "signal": signal,
            "confidence": float(confidence),
            "reasons": reasons,
            "timestamp": datetime.now().isoformat(),
            "analysis": {
                "volume_profile": volume_profile,
                "market_regime": market_regime,
                "ichimoku": ichimoku
            }
        }
        
        # 5. ارسال نوتیفیکیشن برای سیگنال‌های قوی
        if signal in ["STRONG_BUY", "STRONG_SELL"] and confidence > 0.8:
            message = f"""
🔔 *سیگنال {signal}* 🔔

💰 قیمت: {current_price:.2f}
📊 امتیاز: {score:.1f}/10
🎯 اطمینان: {confidence*100:.0f}%

📈 تحلیل:
{chr(10).join(reasons)}

⏰ زمان: {datetime.now().strftime('%H:%M:%S')}
"""
            send_telegram_notification(message, signal.split('_')[-1])
        
        return result
        
    except Exception as e:
        print(f"❌ Signal Generation Error: {e}")
        return {"score": 0, "signal": "ERROR", "reasons": [f"Error: {str(e)}"]}

# ==================== EXIT LEVELS CALCULATOR ====================
def get_exit_levels(price: float, stop_loss: float, 
                   direction: str = "BUY", 
                   scalping_mode: bool = True) -> Dict:
    """
    محاسبه سطوح خروج هوشمند برای اسکالپ
    """
    try:
        risk = abs(price - stop_loss)
        
        if direction.upper() == "BUY":
            multiplier = 1
        elif direction.upper() == "SELL":
            multiplier = -1
        else:
            multiplier = 1 if price > stop_loss else -1
        
        # تنظیمات بر اساس حالت اسکالپ
        if scalping_mode:
            tp1_ratio = 0.7   # 70% ریسک
            tp2_ratio = 1.5   # 150% ریسک
            trailing_activation_ratio = 0.5  # فعال‌سازی تریلینگ در 50% حرکت
            partial_exit_pct = 0.4  # 40% پوزیشن در TP1
        else:
            tp1_ratio = 0.5
            tp2_ratio = 2.0
            trailing_activation_ratio = 0.3
            partial_exit_pct = 0.3
        
        tp1 = price + (risk * tp1_ratio * multiplier)
        tp2 = price + (risk * tp2_ratio * multiplier)
        
        # محاسبه نقطه فعال‌سازی تریلینگ استاپ
        trailing_activation = price + (risk * trailing_activation_ratio * multiplier)
        
        # Break-even نقطه
        breakeven_level = price + (risk * 0.25 * multiplier)
        
        # Risk/Reward Ratios
        rr_ratio_tp1 = tp1_ratio
        rr_ratio_tp2 = tp2_ratio
        
        return {
            "entry": float(price),
            "stop_loss": float(stop_loss),
            "risk_amount": float(risk),
            "risk_percent": float((risk / price) * 100),
            
            "tp1": float(tp1),
            "tp2": float(tp2),
            
            "tp1_distance_pct": float(abs(tp1 - price) / price * 100),
            "tp2_distance_pct": float(abs(tp2 - price) / price * 100),
            
            "rr_tp1": float(rr_ratio_tp1),
            "rr_tp2": float(rr_ratio_tp2),
            
            "trailing_activation": float(trailing_activation),
            "breakeven_level": float(breakeven_level),
            
            "partial_exit_pct": partial_exit_pct,
            "scalping_mode": scalping_mode,
            "direction": direction.upper(),
            
            "calculated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ Exit Levels Error: {e}")
        return {
            "entry": price,
            "stop_loss": stop_loss,
            "tp1": price * 1.01 if price > stop_loss else price * 0.99,
            "tp2": price * 1.02 if price > stop_loss else price * 0.98,
            "error": str(e)
        }

# ==================== MAIN UTILITY CLASS ====================
class ScalpUtils:
    """
    کلاس اصلی برای مدیریت تمام ابزارهای اسکالپ
    """
    
    def __init__(self, config_module):
        self.config = config_module
    
    def analyze_market(self, df: pd.DataFrame) -> Dict:
        """آنالیز کامل بازار"""
        return generate_scalp_signals(df)
    
    def calculate_exits(self, entry: float, stop_loss: float, 
                       direction: str = "BUY") -> Dict:
        """محاسبه سطوح خروج"""
        return get_exit_levels(entry, stop_loss, direction)
    
    def send_alert(self, message: str, alert_type: str = "INFO") -> bool:
        """ارسال هشدار"""
        return send_telegram_notification(message, alert_type)
    
    def get_market_health(self, df: pd.DataFrame) -> Dict:
        """بررسی سلامت بازار برای اسکالپ"""
        regime = detect_market_regime(df)
        vp = get_pro_volume_profile(df)
        
        return {
            "scalp_safe": regime["scalp_safe"],
            "regime": regime["regime"],
            "direction": regime["direction"],
            "volume_zone": vp["current_zone"],
            "volatility": regime.get("volatility", 0),
            "recommendation": "TRADE" if regime["scalp_safe"] else "WAIT"
        }