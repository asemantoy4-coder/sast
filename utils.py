import pandas as pd
import numpy as np
import requests
import config
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging

# تنظیمات لاگ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_bot.log')
    ]
)
logger = logging.getLogger(__name__)

# ==================== TELEGRAM NOTIFICATION ====================
def send_telegram_notification(message, signal_type="INFO"):
    # حذف هرگونه فاصله یا کاراکتر اضافی از ابتدا و انتهای توکن و آیدی
    token = str(config.TELEGRAM_BOT_TOKEN).strip().replace(" ", "")
    chat_id = str(config.TELEGRAM_CHAT_ID).strip().replace(" ", "")
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    try:
        # لاگ برای دیباگ (اختیاری)
        print(f"📡 Attempting to send message to {chat_id}...")
        
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            print("✅ Telegram: Message sent successfully!")
            return True
        else:
            print(f"❌ Telegram Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"🔥 Telegram Connection Exception: {e}")
        return False
        
        # تنظیم ایموجی
        emoji_map = {
            "BUY": "🟢", "SELL": "🔴", "STRONG_BUY": "🚀", "STRONG_SELL": "🔻",
            "ALERT": "⚠️", "INFO": "ℹ️", "ERROR": "❌", "TEST": "🧪",
            "HOLD": "⏸️", "CLOSE": "📉", "SCALP": "⚡"
        }
        
        emoji = emoji_map.get(signal_type, "📊")
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # فرمت پیام
        if signal_type in ["BUY", "SELL", "STRONG_BUY", "STRONG_SELL", "TEST"]:
            formatted_message = f"{emoji} *{signal_type}* [{timestamp}]\n{message}"
        else:
            formatted_message = f"{emoji} {signal_type} [{timestamp}]\n{message}"
        
        url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": config.TELEGRAM_CHAT_ID,
            "text": formatted_message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }
        
        # ارسال درخواست
        response = requests.post(url, json=payload, timeout=15)
        
        # عیب‌یابی دقیق
        if response.status_code != 200:
            error_data = response.json() if response.content else {}
            error_msg = error_data.get('description', 'Unknown error')
            logger.error(f"Telegram API Error (Status {response.status_code}): {error_msg}")
            logger.error(f"Chat ID: {config.TELEGRAM_CHAT_ID}")
            return False
        else:
            logger.info(f"Telegram notification sent successfully! (Type: {signal_type})")
            return True
            
    except requests.exceptions.Timeout:
        logger.error("Telegram Error: Request timeout (15 seconds)")
        return False
    except requests.exceptions.ConnectionError:
        logger.error("Telegram Error: Connection failed - check internet")
        return False
    except Exception as e:
        logger.error(f"Telegram Error: {type(e).__name__}: {str(e)}")
        return False

# ==================== VOLUME PROFILE ADVANCED ====================
def get_pro_volume_profile(df: pd.DataFrame, bins: int = 100) -> Dict[str, Any]:
    """
    محاسبه پیشرفته Volume Profile برای اسکالپ
    شامل نقاط POC، VAH/VAL و تشخیص گره‌های پرحجم (High Volume Nodes)
    """
    try:
        if len(df) < bins:
            return {
                "poc": 0, "vah": 0, "val": 0, 
                "current_zone": "NEUTRAL", 
                "in_value_area": False,
                "poc_strength": 0,
                "high_volume_nodes": [],
                "profile_valid": False
            }
        
        # 1. ایجاد سطوح قیمتی
        price_min = df['Low'].min()
        price_max = df['High'].max()
        
        if price_max <= price_min:
            return {
                "poc": 0, "vah": 0, "val": 0, 
                "current_zone": "NEUTRAL", 
                "in_value_area": False,
                "poc_strength": 0,
                "profile_valid": False
            }
        
        price_levels = np.linspace(price_min, price_max, bins)
        bin_width = (price_max - price_min) / (bins - 1)
        
        # 2. تخصیص حجم به هر سطح
        volumes = np.zeros(bins - 1)
        
        for idx in range(len(df)):
            close_price = df['Close'].iloc[idx]
            volume = df['Volume'].iloc[idx] if 'Volume' in df.columns else 0
            
            bin_idx = int((close_price - price_min) // bin_width)
            bin_idx = max(0, min(bin_idx, len(volumes) - 1))
            
            volumes[bin_idx] += volume
        
        # 3. پیدا کردن POC
        poc_idx = np.argmax(volumes)
        poc_price = price_levels[poc_idx] + (bin_width / 2)
        
        # 4. محاسبه Value Area (70%)
        total_volume = np.sum(volumes)
        if total_volume == 0: 
            return {
                "poc": 0, "vah": 0, "val": 0, 
                "current_zone": "NEUTRAL", 
                "in_value_area": False,
                "poc_strength": 0,
                "profile_valid": False
            }

        target_va_volume = total_volume * 0.70
        
        low_idx, high_idx = poc_idx, poc_idx
        current_va_volume = volumes[poc_idx]
        
        while current_va_volume < target_va_volume and (low_idx > 0 or high_idx < len(volumes) - 1):
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
        
        # 6. تشخیص گره‌های پرحجم (High Volume Nodes)
        volume_threshold = np.percentile(volumes[volumes > 0], 75) if len(volumes[volumes > 0]) > 0 else 0
        high_volume_nodes = []
        
        for i, vol in enumerate(volumes):
            if vol > volume_threshold:
                node_price = price_levels[i] + (bin_width / 2)
                strength = float(vol / total_volume * 100) if total_volume > 0 else 0
                high_volume_nodes.append({
                    "price": float(node_price),
                    "strength": strength,
                    "distance_pct": float(abs(node_price - current_price) / current_price * 100)
                })
        
        # 7. تشخیص ضعف/قوت پروفایل
        profile_valid = volumes[poc_idx] > (total_volume / len(volumes)) * 3
        
        return {
            "poc": float(poc_price),
            "vah": float(vah_price),
            "val": float(val_price),
            "current_zone": current_zone,
            "current_price": float(current_price),
            "value_area_range": float(vah_price - val_price),
            "in_value_area": val_price <= current_price <= vah_price,
            "poc_strength": float(volumes[poc_idx] / total_volume * 100) if total_volume > 0 else 0,
            "high_volume_nodes": sorted(high_volume_nodes, key=lambda x: x["strength"], reverse=True)[:5],
            "profile_valid": profile_valid,
            "total_volume": float(total_volume)
        }
        
    except Exception as e:
        logger.error(f"Volume Profile Error: {type(e).__name__}: {str(e)}")
        return {
            "poc": 0, "vah": 0, "val": 0, 
            "current_zone": "NEUTRAL", 
            "in_value_area": False,
            "poc_strength": 0,
            "high_volume_nodes": [],
            "profile_valid": False,
            "error": str(e)
        }

# ==================== MARKET REGIME DETECTION ====================
def detect_market_regime(df: pd.DataFrame, window: int = 50) -> Dict[str, Any]:
    """
    تشخیص رژیم بازار با فیلترهای اسکالپ
    شامل بررسی SMA، ATR و نوسان‌پذیری
    """
    try:
        if len(df) < window:
            return {
                "regime": "INSUFFICIENT_DATA", 
                "scalp_safe": False, 
                "direction": "NEUTRAL", 
                "volatility": 0, 
                "atr_percent": 0,
                "trend_strength": 0
            }
        
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
        
        # 3. تشخیص روند با قدرت
        sma_50 = df['Close'].rolling(window=50).mean()
        sma_20 = df['Close'].rolling(window=20).mean()
        current_price = df['Close'].iloc[-1]
        
        # قدرت روند
        price_vs_sma50 = ((current_price / sma_50.iloc[-1]) - 1) * 100 if sma_50.iloc[-1] > 0 else 0
        sma20_vs_sma50 = ((sma_20.iloc[-1] / sma_50.iloc[-1]) - 1) * 100 if sma_50.iloc[-1] > 0 else 0
        trend_strength = abs(price_vs_sma50) + abs(sma20_vs_sma50)
        
        if current_price > sma_50.iloc[-1] and sma_20.iloc[-1] > sma_50.iloc[-1]:
            direction = "BULLISH"
        elif current_price < sma_50.iloc[-1] and sma_20.iloc[-1] < sma_50.iloc[-1]:
            direction = "BEARISH"
        else:
            direction = "SIDEWAYS"
        
        # 4. شرایط اسکالپ امن
        scalp_safe = True
        regime = "RANGING"
        
        if current_volatility < 0.001:
            scalp_safe = False; regime = "DEAD_MARKET"
        elif current_volatility > 0.02:
            scalp_safe = False; regime = "VOLATILE"
        elif current_atr_pct > 2.0:
            scalp_safe = False; regime = "HIGH_VOLATILITY"
        elif direction == "SIDEWAYS" and trend_strength < 1.0:
            regime = "RANGING"
        else:
            regime = "TRENDING"
        
        # 5. فیلتر حجم
        volume_filter = "NORMAL"
        if 'Volume' in df.columns:
            avg_volume = df['Volume'].rolling(window=20).mean().iloc[-1]
            current_volume = df['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            if current_volume < avg_volume * 0.5:
                scalp_safe = False
                volume_filter = "LOW_VOLUME"
            elif volume_ratio > 2.0:
                volume_filter = "HIGH_VOLUME"
        
        return {
            "regime": regime,
            "scalp_safe": scalp_safe,
            "direction": direction,
            "volatility": float(current_volatility),
            "atr_percent": float(current_atr_pct),
            "price_vs_sma50": float(price_vs_sma50),
            "trend_strength": float(trend_strength),
            "volume_filter": volume_filter,
            "regime_score": calculate_regime_score(regime, scalp_safe, direction, current_atr_pct)
        }
        
    except Exception as e:
        logger.error(f"Market Regime Error: {type(e).__name__}: {str(e)}")
        return {
            "regime": "ERROR", 
            "scalp_safe": False, 
            "direction": "NEUTRAL",
            "volatility": 0,
            "atr_percent": 0,
            "trend_strength": 0
        }

def calculate_regime_score(regime: str, scalp_safe: bool, direction: str, atr_percent: float) -> float:
    """محاسبه امتیاز رژیم بازار"""
    score = 0.0
    
    # امتیاز رژیم
    regime_scores = {
        "TRENDING": 3.0,
        "RANGING": 2.0,
        "DEAD_MARKET": 0.0,
        "VOLATILE": -1.0,
        "HIGH_VOLATILITY": -2.0,
        "LOW_LIQUIDITY": -1.0
    }
    
    score += regime_scores.get(regime, 0.0)
    
    # امتیاز اسکالپ امن
    if scalp_safe:
        score += 2.0
    
    # امتیاز جهت روند
    if direction == "BULLISH":
        score += 1.5
    elif direction == "BEARISH":
        score += 1.0
    
    # امتیاز نوسان
    if 0.5 <= atr_percent <= 1.5:
        score += 2.0
    elif atr_percent < 0.3:
        score -= 1.0
    
    return float(score)

# ==================== ICHIMOKU ANALYSIS ====================
def get_ichimoku(df: pd.DataFrame) -> Dict[str, Any]:
    """تحلیل ایچیموکو بهینه‌شده برای اسکالپ"""
    try:
        if len(df) < 52:
            return {
                "trend": "NEUTRAL", 
                "price_above_cloud": False, 
                "signal": "NO_DATA",
                "ichimoku_score": 0
            }
        
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
        
        price_above_cloud = current_price > max(senkou_a.iloc[-1], senkou_b.iloc[-1])
        price_below_cloud = current_price < min(senkou_a.iloc[-1], senkou_b.iloc[-1])
        price_in_cloud = not (price_above_cloud or price_below_cloud)
        
        trend = "NEUTRAL"
        signal = "HOLD"
        
        # تحلیل سیگنال
        tenkan_kijun_diff = ((tenkan.iloc[-1] / kijun.iloc[-1]) - 1) * 100 if kijun.iloc[-1] > 0 else 0
        
        if tenkan.iloc[-1] > kijun.iloc[-1] and price_above_cloud:
            trend = "STRONG_BULLISH"; signal = "BUY"
        elif tenkan.iloc[-1] > kijun.iloc[-1]:
            trend = "BULLISH"; signal = "BUY"
        elif tenkan.iloc[-1] < kijun.iloc[-1] and price_below_cloud:
            trend = "STRONG_BEARISH"; signal = "SELL"
        elif tenkan.iloc[-1] < kijun.iloc[-1]:
            trend = "BEARISH"; signal = "SELL"
        
        # شیب کیجون
        kijun_slope = 0
        if len(kijun) >= 5:
            kijun_slope = ((kijun.iloc[-1] / kijun.iloc[-5]) - 1) * 100 if kijun.iloc[-5] > 0 else 0
        
        # ضخامت ابر
        cloud_thickness = abs(senkou_a.iloc[-1] - senkou_b.iloc[-1]) / current_price * 100 if current_price > 0 else 0
        
        # امتیاز ایچیموکو
        ichimoku_score = calculate_ichimoku_score(trend, signal, price_above_cloud, price_below_cloud, tenkan_kijun_diff)
        
        return {
            "trend": trend,
            "signal": signal,
            "price_above_cloud": price_above_cloud,
            "price_below_cloud": price_below_cloud,
            "price_in_cloud": price_in_cloud,
            "tenkan": float(tenkan.iloc[-1]),
            "kijun": float(kijun.iloc[-1]),
            "kijun_slope_pct": float(kijun_slope),
            "tenkan_kijun_diff_pct": float(tenkan_kijun_diff),
            "cloud_top": float(max(senkou_a.iloc[-1], senkou_b.iloc[-1])),
            "cloud_bottom": float(min(senkou_a.iloc[-1], senkou_b.iloc[-1])),
            "cloud_thickness_pct": float(cloud_thickness),
            "price_vs_kijun": float((current_price / kijun.iloc[-1] - 1) * 100),
            "ichimoku_score": float(ichimoku_score)
        }
        
    except Exception as e:
        logger.error(f"Ichimoku Error: {type(e).__name__}: {str(e)}")
        return {
            "trend": "NEUTRAL", 
            "price_above_cloud": False, 
            "signal": "ERROR",
            "ichimoku_score": 0
        }

def calculate_ichimoku_score(trend: str, signal: str, price_above_cloud: bool, 
                            price_below_cloud: bool, tenkan_kijun_diff: float) -> float:
    """محاسبه امتیاز ایچیموکو"""
    score = 0.0
    
    # امتیاز روند
    trend_scores = {
        "STRONG_BULLISH": 3.0,
        "BULLISH": 2.0,
        "STRONG_BEARISH": -3.0,
        "BEARISH": -2.0,
        "NEUTRAL": 0.0
    }
    
    score += trend_scores.get(trend, 0.0)
    
    # امتیاز سیگنال
    if signal == "BUY":
        score += 2.0
    elif signal == "SELL":
        score -= 2.0
    
    # امتیاز موقعیت نسبت به ابر
    if price_above_cloud:
        score += 2.0
    elif price_below_cloud:
        score -= 2.0
    
    # امتیاز اختلاف تنکان و کیجون
    if tenkan_kijun_diff > 0.5:
        score += 1.0
    elif tenkan_kijun_diff < -0.5:
        score -= 1.0
    
    return float(score)

# ==================== EXIT LEVELS CALCULATOR ====================
def get_exit_levels(price: float, stop_loss: float, 
                   direction: str = "BUY", 
                   scalping_mode: bool = True,
                   volatility_pct: float = 1.0) -> Dict[str, Any]:
    """
    محاسبه سطوح خروج هوشمند برای اسکالپ
    شامل Breakeven و Trailing Stop Logic
    """
    try:
        # اطمینان از جهت صحیح
        direction = direction.upper()
        if direction not in ["BUY", "SELL"]:
            direction = "BUY"
        
        # محاسبه ریسک
        risk = abs(price - stop_loss)
        if risk == 0: 
            risk = price * 0.01  # 1% ریسک پیش‌فرض
        
        # تنظیم ضریب بر اساس جهت
        multiplier = 1 if direction == "BUY" else -1
        
        # تنظیمات بر اساس حالت اسکالپ و نوسان
        if scalping_mode:
            # تنظیمات حساس‌تر برای اسکالپ
            tp1_ratio = 0.7 + (volatility_pct * 0.1)   # 70-80% ریسک
            tp2_ratio = 1.5 + (volatility_pct * 0.2)   # 150-170% ریسک
            trailing_activation_ratio = 0.5  # فعال‌سازی تریلینگ در 50% حرکت
            partial_exit_pct = 0.4  # 40% پوزیشن در TP1
            breakeven_ratio = 0.25  # Break-even در 25% حرکت
        else:
            # تنظیمات برای ترید معمولی
            tp1_ratio = 0.5 + (volatility_pct * 0.1)
            tp2_ratio = 2.0 + (volatility_pct * 0.3)
            trailing_activation_ratio = 0.3
            partial_exit_pct = 0.3
            breakeven_ratio = 0.15
        
        # محدود کردن نسبت‌ها
        tp1_ratio = max(0.3, min(tp1_ratio, 1.5))
        tp2_ratio = max(1.0, min(tp2_ratio, 3.0))
        
        # محاسبه سطوح
        tp1 = price + (risk * tp1_ratio * multiplier)
        tp2 = price + (risk * tp2_ratio * multiplier)
        
        # محاسبه نقطه فعال‌سازی تریلینگ استاپ
        trailing_activation = price + (risk * trailing_activation_ratio * multiplier)
        
        # Break-even نقطه
        breakeven_level = price + (risk * breakeven_ratio * multiplier)
        
        # محاسبه سود به ریسک
        risk_percent = (risk / price) * 100
        tp1_profit_percent = abs(tp1 - price) / price * 100
        tp2_profit_percent = abs(tp2 - price) / price * 100
        rr_tp1 = tp1_profit_percent / risk_percent if risk_percent > 0 else 0
        rr_tp2 = tp2_profit_percent / risk_percent if risk_percent > 0 else 0
        
        return {
            "entry": float(price),
            "stop_loss": float(stop_loss),
            "risk_amount": float(risk),
            "risk_percent": float(risk_percent),
            
            "tp1": float(tp1),
            "tp2": float(tp2),
            
            "tp1_distance_pct": float(abs(tp1 - price) / price * 100),
            "tp2_distance_pct": float(abs(tp2 - price) / price * 100),
            
            "tp1_profit_percent": float(tp1_profit_percent),
            "tp2_profit_percent": float(tp2_profit_percent),
            
            "rr_tp1": float(rr_tp1),
            "rr_tp2": float(rr_tp2),
            
            "trailing_activation": float(trailing_activation),
            "breakeven_level": float(breakeven_level),
            
            "partial_exit_pct": partial_exit_pct,
            "scalping_mode": scalping_mode,
            "direction": direction,
            "volatility_adjusted": volatility_pct,
            
            "calculated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Exit Levels Error: {type(e).__name__}: {str(e)}")
        return {
            "entry": price,
            "stop_loss": stop_loss,
            "tp1": price * 1.01 if direction.upper() == "BUY" else price * 0.99,
            "tp2": price * 1.02 if direction.upper() == "BUY" else price * 0.98,
            "error": str(e)
        }

# ==================== SCALP SIGNAL GENERATOR ====================
def generate_scalp_signals(df: pd.DataFrame, test_mode: bool = False, 
                          force_signal: Optional[str] = None) -> Dict[str, Any]:
    """
    تولید سیگنال اسکالپ نهایی با امتیازدهی هوشمند
    
    پارامترها:
    ----------
    df : pd.DataFrame
        داده‌های قیمت
    test_mode : bool, optional
        اگر True باشد، سیگنال‌های تست تولید می‌کند (پیش‌فرض: False)
    force_signal : str, optional
        اگر مشخص باشد، این سیگنال اجباری اعمال می‌شود (BUY/SELL)
    
    بازگشت:
    -------
    Dict: شامل تمام تحلیل‌ها و سیگنال نهایی
    """
    try:
        # اعتبارسنجی داده‌ها
        validation_result = validate_dataframe(df)
        if not validation_result["valid"]:
            return {
                "score": 0, 
                "signal": "INVALID_DATA", 
                "reasons": validation_result["errors"], 
                "analysis": {},
                "valid": False
            }
        
        current_price = df['Close'].iloc[-1]
        logger.info(f"Analyzing data: {len(df)} candles, Current Price: {current_price}")
        
        # 1. جمع‌آوری تمام تحلیل‌ها
        volume_profile = get_pro_volume_profile(df)
        market_regime = detect_market_regime(df)
        ichimoku = get_ichimoku(df)
        
        # 2. محاسبه امتیاز پایه
        score = 0.0
        reasons = []
        scoring_details = {}
        
        # 2.1 امتیاز Volume Profile
        vp_zone = volume_profile.get('current_zone')
        if vp_zone == "CHEAP":
            score += 3.0
            reasons.append("قیمت در ناحیه ارزان حجمی (CHEAP)")
            scoring_details["volume_zone"] = 3.0
        elif vp_zone == "EXPENSIVE":
            score -= 3.0
            reasons.append("قیمت در ناحیه گران حجمی (EXPENSIVE)")
            scoring_details["volume_zone"] = -3.0
        
        if volume_profile.get('in_value_area', False):
            score += 1.0
            reasons.append("قیمت در محدوده ارزش (Value Area)")
            scoring_details["in_value_area"] = 1.0
            
        # اگر POC قوی باشد
        poc_strength = volume_profile.get('poc_strength', 0)
        if poc_strength > 15:
            score += 1.0
            reasons.append(f"نقطه کنترل حجمی قوی (POC: {poc_strength:.1f}%)")
            scoring_details["poc_strength"] = 1.0
        
        # 2.2 امتیاز Market Regime
        if market_regime.get('scalp_safe', False):
            score += 2.0
            reasons.append("بازار برای اسکالپ امن است")
            scoring_details["scalp_safe"] = 2.0
        else:
            reasons.append(f"بازار برای اسکالپ مناسب نیست (رژیم: {market_regime.get('regime', 'UNKNOWN')})")
            scoring_details["scalp_safe"] = 0.0
        
        direction = market_regime.get('direction', 'NEUTRAL')
        if direction == "BULLISH":
            score += 1.5
            reasons.append("روند صعودی")
            scoring_details["direction"] = 1.5
        elif direction == "BEARISH":
            score += 1.0
            reasons.append("روند نزولی")
            scoring_details["direction"] = 1.0
        
        # اضافه کردن امتیاز رژیم
        regime_score = market_regime.get('regime_score', 0)
        score += regime_score
        scoring_details["regime_score"] = regime_score
        
        # 2.3 امتیاز Ichimoku
        ichimoku_signal = ichimoku.get('signal', 'HOLD')
        ichimoku_score = ichimoku.get('ichimoku_score', 0)
        
        score += ichimoku_score
        scoring_details["ichimoku_score"] = ichimoku_score
        
        if ichimoku_signal == "BUY":
            reasons.append(f"سیگنال ایچیموکو: {ichimoku.get('trend', 'NEUTRAL')}")
        elif ichimoku_signal == "SELL":
            reasons.append(f"سیگنال ایچیموکو: {ichimoku.get('trend', 'NEUTRAL')}")
        
        if ichimoku.get('price_above_cloud', False):
            reasons.append("قیمت بالای ابر کومو (مثبت)")
        elif ichimoku.get('price_below_cloud', False):
            reasons.append("قیمت زیر ابر کومو (منفی)")
        
        # 2.4 امتیاز حجم لحظه‌ای
        if 'Volume' in df.columns:
            avg_volume = df['Volume'].rolling(window=20).mean().iloc[-1]
            current_volume = df['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            if volume_ratio > 1.5:
                score += 2.0
                reasons.append(f"حجم معاملات بالا ({volume_ratio:.1f}x میانگین)")
                scoring_details["volume_ratio"] = 2.0
            elif volume_ratio < 0.5:
                score -= 1.0
                reasons.append(f"حجم معاملات پایین ({volume_ratio:.1f}x میانگین)")
                scoring_details["volume_ratio"] = -1.0
        
        # 2.5 امتیاز نوسان مناسب برای اسکالپ
        atr_percent = market_regime.get('atr_percent', 0)
        if 0.5 <= atr_percent <= 1.5:
            score += 2.0
            reasons.append(f"نوسان مناسب برای اسکالپ ({atr_percent:.2f}%)")
            scoring_details["atr_score"] = 2.0
        elif atr_percent > 2.0:
            score -= 1.0
            reasons.append(f"نوسان بسیار بالا ({atr_percent:.2f}%)")
            scoring_details["atr_score"] = -1.0
        
        # 3. تعیین سیگنال نهایی
        signal = "HOLD"
        confidence = 0.0
        
        # اولویت: سیگنال اجباری (اگر مشخص شده باشد)
        if force_signal and force_signal.upper() in ["BUY", "SELL"]:
            signal = force_signal.upper()
            confidence = 0.8
            reasons.append(f"سیگنال اجباری اعمال شد: {signal}")
            logger.info(f"Force signal applied: {signal}")
        
        # حالت تست: تولید سیگنال برای آزمایش
        elif test_mode:
            if score >= 0:
                signal = "BUY"
                reasons.append("🔬 حالت تست فعال - سیگنال خرید (امتیاز مثبت)")
            else:
                signal = "SELL"
                reasons.append("🔬 حالت تست فعال - سیگنال فروش (امتیاز منفی)")
            confidence = min(abs(score) / 10 + 0.3, 0.8)  # اعتماد متوسط در تست
        
        # حالت عادی
        else:
            # محاسبه امتیاز نرمال‌شده
            normalized_score = score / 15.0  # تقسیم بر حداکثر امتیاز ممکن
            
            if score >= 7.0:
                signal = "STRONG_BUY"
                confidence = min(normalized_score, 0.95)
                reasons.append("💪 سیگنال خرید قوی")
            elif score >= 4.0:
                signal = "BUY"
                confidence = min(normalized_score, 0.85)
                reasons.append("✅ سیگنال خرید")
            elif score <= -7.0:
                signal = "STRONG_SELL"
                confidence = min(abs(normalized_score), 0.95)
                reasons.append("💪 سیگنال فروش قوی")
            elif score <= -4.0:
                signal = "SELL"
                confidence = min(abs(normalized_score), 0.85)
                reasons.append("✅ سیگنال فروش")
            else:
                signal = "HOLD"
                confidence = 0.5
                reasons.append("⏸️ سیگنال مشخصی نیست (نگهداری)")
        
        # 4. محاسبه سطوح خروج
        exit_levels = None
        if signal in ["BUY", "SELL", "STRONG_BUY", "STRONG_SELL"]:
            # تعیین استاپ‌لاس بر اساس نوسان
            sl_distance = 0.5 if signal.startswith("STRONG") else 1.0
            sl_multiplier = 0.995 if signal in ["BUY", "STRONG_BUY"] else 1.005
            stop_loss = current_price * sl_multiplier
            
            exit_levels = get_exit_levels(
                price=current_price,
                stop_loss=stop_loss,
                direction="BUY" if signal in ["BUY", "STRONG_BUY"] else "SELL",
                scalping_mode=True,
                volatility_pct=atr_percent
            )
        
        # 5. جمع‌بندی
        result = {
            "price": float(current_price),
            "score": float(score),
            "signal": signal,
            "confidence": float(confidence),
            "reasons": reasons,
            "timestamp": datetime.now().isoformat(),
            "test_mode": test_mode,
            "force_signal": force_signal,
            "scoring_details": scoring_details,
            "exit_levels": exit_levels,
            "analysis": {
                "volume_profile": volume_profile,
                "market_regime": market_regime,
                "ichimoku": ichimoku
            },
            "valid": True
        }
        
        # لاگ اطلاعاتی برای دیباگ
        logger.info(f"Signal Generated: {signal} (Score: {score:.1f}, Confidence: {confidence:.1%})")
        logger.info(f"Price: {current_price:.2f}, Test Mode: {test_mode}")
        
        return result
        
    except Exception as e:
        error_msg = f"Signal Generation Error: {type(e).__name__}: {str(e)}"
        logger.error(error_msg)
        return {
            "score": 0, 
            "signal": "ERROR", 
            "reasons": [error_msg], 
            "analysis": {},
            "error": True,
            "valid": False
        }

# ==================== HELPER FUNCTIONS ====================
def format_price(price: float) -> str:
    """قالب‌بندی قیمت برای نمایش"""
    return f"{price:,.2f}"

def calculate_pivot_points(df: pd.DataFrame) -> Dict[str, float]:
    """محاسبه نقاط پیوت استاندارد"""
    try:
        if len(df) < 1:
            return {}
            
        high = df['High'].iloc[-1]
        low = df['Low'].iloc[-1]
        close = df['Close'].iloc[-1]
        
        pp = (high + low + close) / 3
        r1 = 2 * pp - low
        s1 = 2 * pp - high
        r2 = pp + (high - low)
        s2 = pp - (high - low)
        r3 = high + 2 * (pp - low)
        s3 = low - 2 * (high - pp)
        
        return {
            "pivot": float(pp),
            "r1": float(r1),
            "r2": float(r2),
            "r3": float(r3),
            "s1": float(s1),
            "s2": float(s2),
            "s3": float(s3)
        }
    except Exception as e:
        logger.error(f"Pivot Points Error: {e}")
        return {}

def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """اعتبارسنجی DataFrame ورودی"""
    errors = []
    
    # بررسی ستون‌های ضروری
    required_columns = ['Open', 'High', 'Low', 'Close']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    
    # بررسی حداقل داده
    if len(df) < 20:
        errors.append(f"Insufficient data: {len(df)} rows (minimum 20)")
    
    # بررسی مقادیر NaN
    if df.isnull().values.any():
        errors.append("DataFrame contains NaN values")
    
    # بررسی مقادیر غیرعادی
    for col in required_columns:
        if col in df.columns:
            if df[col].min() <= 0:
                errors.append(f"Column {col} contains non-positive values")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "row_count": len(df)
    }

# ==================== TEST FUNCTIONS ====================
def test_all_functions():
    """تابع تست برای بررسی عملکرد تمام توابع"""
    logger.info("🧪 Running tests...")
    
    try:
        # ایجاد داده تست
        dates = pd.date_range(start='2024-01-01', periods=200, freq='H')
        np.random.seed(42)
        base_price = 100
        prices = base_price + np.cumsum(np.random.randn(200) * 0.5)
        
        df = pd.DataFrame({
            'Open': prices * 0.999,
            'High': prices * 1.005,
            'Low': prices * 0.995,
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, 200)
        }, index=dates)
        
        logger.info(f"✅ Test DataFrame created: {len(df)} rows")
        
        # تست Volume Profile
        vp = get_pro_volume_profile(df)
        logger.info(f"✅ Volume Profile: POC={vp.get('poc', 0):.2f}, Zone={vp.get('current_zone', 'N/A')}, Valid={vp.get('profile_valid', False)}")
        
        # تست Market Regime
        regime = detect_market_regime(df)
        logger.info(f"✅ Market Regime: {regime.get('regime', 'N/A')}, Safe={regime.get('scalp_safe', False)}")
        
        # تست Ichimoku
        ichi = get_ichimoku(df)
        logger.info(f"✅ Ichimoku: Signal={ichi.get('signal', 'N/A')}, Score={ichi.get('ichimoku_score', 0):.1f}")
        
        # تست Exit Levels
        exit_levels = get_exit_levels(100, 98, "BUY", True)
        logger.info(f"✅ Exit Levels: TP1={exit_levels.get('tp1', 0):.2f}, RR={exit_levels.get('rr_tp1', 0):.1f}")
        
        # تست سیگنال (حالت عادی)
        signals = generate_scalp_signals(df, test_mode=False)
        logger.info(f"✅ Normal Signals: Signal={signals.get('signal', 'N/A')}, Score={signals.get('score', 0):.1f}")
        
        # تست سیگنال (حالت تست)
        test_signals = generate_scalp_signals(df, test_mode=True)
        logger.info(f"✅ Test Signals: Signal={test_signals.get('signal', 'N/A')}, Score={test_signals.get('score', 0):.1f}")
        
        # تست سیگنال اجباری
        forced_signals = generate_scalp_signals(df, test_mode=False, force_signal="BUY")
        logger.info(f"✅ Forced Signals: Signal={forced_signals.get('signal', 'N/A')}, Forced={forced_signals.get('force_signal', 'N/A')}")
        
        # تست تلگرام (حالت دیباگ)
        telegram_result = send_telegram_notification("Test message from utils", "TEST", debug_mode=True)
        logger.info(f"✅ Telegram Test (Debug): {'Success' if telegram_result else 'Failed'}")
        
        logger.info("✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {type(e).__name__}: {str(e)}")
        return False

# ==================== SIGNAL FORMATTER ====================
def format_signal_message(symbol: str, signal_data: Dict[str, Any]) -> str:
    """فرمت‌بندی پیام سیگنال برای تلگرام"""
    try:
        signal = signal_data.get('signal', 'HOLD')
        price = signal_data.get('price', 0)
        score = signal_data.get('score', 0)
        confidence = signal_data.get('confidence', 0)
        reasons = signal_data.get('reasons', [])
        
        # ایموجی بر اساس سیگنال
        emoji_map = {
            "STRONG_BUY": "🚀",
            "BUY": "🟢",
            "STRONG_SELL": "🔻",
            "SELL": "🔴",
            "HOLD": "⏸️",
            "TEST": "🧪"
        }
        
        emoji = emoji_map.get(signal, "📊")
        
        # ساخت پیام
        lines = [
            f"{emoji} *{signal}*",
            f"`{symbol}`",
            f"💰 قیمت: {format_price(price)}",
            f"📊 امتیاز: {score:.1f}",
            f"🎯 اعتماد: {confidence:.0%}",
            "",
            "*دلایل:*"
        ]
        
        # اضافه کردن دلایل (حداکثر 5 مورد)
        for i, reason in enumerate(reasons[:5]):
            lines.append(f"• {reason}")
        
        # اضافه کردن TP/SL اگر موجود باشد
        exit_levels = signal_data.get('exit_levels')
        if exit_levels and signal in ["BUY", "SELL", "STRONG_BUY", "STRONG_SELL"]:
            lines.extend([
                "",
                "*سطوح معاملاتی:*",
                f"🎯 TP1: {format_price(exit_levels.get('tp1', 0))}",
                f"🎯 TP2: {format_price(exit_levels.get('tp2', 0))}",
                f"⛔ SL: {format_price(exit_levels.get('stop_loss', 0))}"
            ])
        
        lines.append("")
        lines.append("📡 @AsemanSignals")
        
        return "\n".join(lines)
        
    except Exception as e:
        logger.error(f"Format Signal Message Error: {e}")
        return f"❌ Error formatting signal for {symbol}"

if __name__ == "__main__":
    # اجرای تست اگر فایل مستقیماً اجرا شود
    test_all_functions()
