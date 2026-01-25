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
def send_telegram_notification(message, signal_type="INFO", exit_levels=None):
    try:
        token = str(config.TELEGRAM_BOT_TOKEN).strip().replace(" ", "")
        chat_id = str(config.TELEGRAM_CHAT_ID).strip().replace(" ", "")
        
        emoji_map = {
            "BUY": "🟢", "SELL": "🔴", "STRONG_BUY": "🚀", "STRONG_SELL": "🔻",
            "TARGET": "🎯", "STOP": "🛑", "INFO": "ℹ️", "TEST": "🧪"
        }
        emoji = emoji_map.get(signal_type, "📊")
        
        full_message = f"{emoji} *{signal_type}*\n{message}"
        
        if exit_levels:
            full_message += (
                f"\n\n🎯 *Targets:*\n"
                f"🔹 Entry: {exit_levels['entry']:.4f}\n"
                f"✅ TP1: {exit_levels['tp1']:.4f}\n"
                f"✅ TP2: {exit_levels['tp2']:.4f}\n"
                f"🛑 SL: {exit_levels['stop_loss']:.4f}"
            )

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": full_message, "parse_mode": "Markdown"}
        
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"Telegram Error: {e}")
        return False
    except requests.exceptions.Timeout:
        logger.error("Telegram Error: Request timeout")
        return False
    except requests.exceptions.ConnectionError:
        logger.error("Telegram Error: Connection failed")
        return False
    except Exception as e:
        logger.error(f"Telegram Error: {type(e).__name__}: {str(e)}")
        return False

# ==================== DUAL CHIKOU FUTURE ANALYSIS ====================
def analyze_dual_chikou_future(current_price: float, price_26_periods_ago: float, 
                              price_52_periods_ago: float, tenkan_current: float = None, 
                              kijun_current: float = None, trend_direction: str = None) -> Dict[str, Any]:
    """
    تحلیل دو چیکوی آینده:
    1. چیکو 26 کندل جلوتر (قیمت 26 دوره قبل)
    2. چیکو 78 کندل جلوتر (قیمت 52 دوره قبل - چون 78-26=52)
    
    منطق: اگر هر دو چیکو بالای کندل باشند → سیگنال فروش قوی
          اگر هر دو چیکو زیر کندل باشند → سیگنال خرید قوی
    """
    try:
        # چیکو اول: 26 کندل جلوتر
        chikou_26 = price_26_periods_ago
        
        # چیکو دوم: 78 کندل جلوتر (که در واقع قیمت 52 دوره قبل است)
        chikou_78 = price_52_periods_ago
        
        # تحلیل هر چیکو به صورت جداگانه
        chikou_26_above = chikou_26 > current_price
        chikou_78_above = chikou_78 > current_price
        
        chikou_26_below = chikou_26 < current_price
        chikou_78_below = chikou_78 < current_price
        
        # محاسبه اختلاف‌ها
        diff_26 = ((chikou_26 - current_price) / current_price) * 100 if current_price > 0 else 0
        diff_78 = ((chikou_78 - current_price) / current_price) * 100 if current_price > 0 else 0
        
        # تشخیص سیگنال بر اساس دو چیکو
        signal = "NEUTRAL"
        boost_multiplier = 1.0
        confidence = 0.0
        reasons = []
        
        # حالت 1: هر دو چیکو بالای کندل (فروش قوی)
        if chikou_26_above and chikou_78_above:
            signal = "STRONG_SELL"
            # میانگین اختلاف دو چیکو
            avg_diff = (abs(diff_26) + abs(diff_78)) / 2
            confidence = min(avg_diff / 3.0, 1.0)  # حداکثر 1.0
            boost_multiplier = 1.15  # افزایش 15% برای تطابق کامل
            reasons.append(f"هر دو چیکو بالای کندل (26: +{diff_26:.2f}%, 78: +{diff_78:.2f}%)")
            
        # حالت 2: هر دو چیکو زیر کندل (خرید قوی)
        elif chikou_26_below and chikou_78_below:
            signal = "STRONG_BUY"
            avg_diff = (abs(diff_26) + abs(diff_78)) / 2
            confidence = min(avg_diff / 3.0, 1.0)
            boost_multiplier = 1.15
            reasons.append(f"هر دو چیکو زیر کندل (26: {diff_26:.2f}%, 78: {diff_78:.2f}%)")
            
        # حالت 3: فقط چیکو 26 بالای کندل (فروش ضعیف)
        elif chikou_26_above and not chikou_78_above:
            signal = "WEAK_SELL"
            confidence = min(abs(diff_26) / 2.0, 0.7)
            boost_multiplier = 1.08  # افزایش 8%
            reasons.append(f"فقط چیکو 26 بالای کندل (+{diff_26:.2f}%)")
            
        # حالت 4: فقط چیکو 26 زیر کندل (خرید ضعیف)
        elif chikou_26_below and not chikou_78_below:
            signal = "WEAK_BUY"
            confidence = min(abs(diff_26) / 2.0, 0.7)
            boost_multiplier = 1.08
            reasons.append(f"فقط چیکو 26 زیر کندل ({diff_26:.2f}%)")
            
        # حالت 5: تناقض (چیکو 26 بالا، چیکو 78 پایین)
        elif chikou_26_above and chikou_78_below:
            signal = "NEUTRAL"
            boost_multiplier = 1.0
            confidence = 0.2
            reasons.append("تناقض: چیکو 26 بالا، چیکو 78 پایین")
            
        # حالت 6: تناقض (چیکو 26 پایین، چیکو 78 بالا)
        elif chikou_26_below and chikou_78_above:
            signal = "NEUTRAL"
            boost_multiplier = 1.0
            confidence = 0.2
            reasons.append("تناقض: چیکو 26 پایین، چیکو 78 بالا")
        
        # تطابق با تنکان و کیجون (اگر موجود باشد)
        if tenkan_current is not None and kijun_current is not None:
            if signal in ["STRONG_SELL", "WEAK_SELL"]:
                if chikou_26 > tenkan_current and chikou_26 > kijun_current:
                    boost_multiplier *= 1.05
                    reasons.append("چیکو 26 بالای تنکان و کیجون")
            elif signal in ["STRONG_BUY", "WEAK_BUY"]:
                if chikou_26 < tenkan_current and chikou_26 < kijun_current:
                    boost_multiplier *= 1.05
                    reasons.append("چیکو 26 زیر تنکان و کیجون")
        
        # تطابق با روند (اگر موجود باشد)
        if trend_direction:
            if (signal in ["STRONG_SELL", "WEAK_SELL"] and trend_direction == "bearish") or \
               (signal in ["STRONG_BUY", "WEAK_BUY"] and trend_direction == "bullish"):
                boost_multiplier *= 1.05
                reasons.append(f"تطابق با روند {trend_direction}")
        
        # اعتبارسنجی: حداقل اختلاف
        if abs(diff_26) < 0.2 and abs(diff_78) < 0.2:
            signal = "NEUTRAL"
            boost_multiplier = 1.0
            confidence = 0.0
            reasons.append("اختلاف قیمت ناچیز")
        
        # گرد‌سازی
        boost_multiplier = round(boost_multiplier, 3)
        confidence = round(confidence, 3)
        
        return {
            'signal': signal,
            'boost_multiplier': boost_multiplier,
            'confidence': confidence,
            'chikou_26_diff': round(diff_26, 2),
            'chikou_78_diff': round(diff_78, 2),
            'chikou_26_price': float(chikou_26),
            'chikou_78_price': float(chikou_78),
            'current_price': float(current_price),
            'reasons': reasons,
            'chikou_26_above': chikou_26_above,
            'chikou_78_above': chikou_78_above,
            'both_above': chikou_26_above and chikou_78_above,
            'both_below': chikou_26_below and chikou_78_below
        }
        
    except Exception as e:
        logger.error(f"Dual Chikou Analysis Error: {type(e).__name__}: {str(e)}")
        return {
            'signal': 'NEUTRAL',
            'boost_multiplier': 1.0,
            'confidence': 0.0,
            'reasons': [f'خطا در تحلیل: {str(e)}']
        }

# ==================== VOLUME PROFILE ADVANCED ====================
def get_pro_volume_profile(df: pd.DataFrame, bins: int = 100) -> Dict[str, Any]:
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
        
        volumes = np.zeros(bins - 1)
        
        for idx in range(len(df)):
            close_price = df['Close'].iloc[idx]
            volume = df['Volume'].iloc[idx] if 'Volume' in df.columns else 0
            
            bin_idx = int((close_price - price_min) // bin_width)
            bin_idx = max(0, min(bin_idx, len(volumes) - 1))
            volumes[bin_idx] += volume
        
        poc_idx = np.argmax(volumes)
        poc_price = price_levels[poc_idx] + (bin_width / 2)
        
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
        
        current_price = df['Close'].iloc[-1]
        current_zone = "NEUTRAL"
        
        if current_price < val_price:
            current_zone = "CHEAP"
        elif current_price > vah_price:
            current_zone = "EXPENSIVE"
        
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
        
        returns = df['Close'].pct_change().dropna()
        volatility = returns.rolling(window=window).std()
        current_volatility = volatility.iloc[-1]
        
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        atr_percent = (atr / df['Close']) * 100
        current_atr_pct = atr_percent.iloc[-1]
        
        sma_50 = df['Close'].rolling(window=50).mean()
        sma_20 = df['Close'].rolling(window=20).mean()
        current_price = df['Close'].iloc[-1]
        
        price_vs_sma50 = ((current_price / sma_50.iloc[-1]) - 1) * 100 if sma_50.iloc[-1] > 0 else 0
        sma20_vs_sma50 = ((sma_20.iloc[-1] / sma_50.iloc[-1]) - 1) * 100 if sma_50.iloc[-1] > 0 else 0
        trend_strength = abs(price_vs_sma50) + abs(sma20_vs_sma50)
        
        if current_price > sma_50.iloc[-1] and sma_20.iloc[-1] > sma_50.iloc[-1]:
            direction = "BULLISH"
        elif current_price < sma_50.iloc[-1] and sma_20.iloc[-1] < sma_50.iloc[-1]:
            direction = "BEARISH"
        else:
            direction = "SIDEWAYS"
        
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
    score = 0.0
    
    regime_scores = {
        "TRENDING": 3.0,
        "RANGING": 2.0,
        "DEAD_MARKET": 0.0,
        "VOLATILE": -1.0,
        "HIGH_VOLATILITY": -2.0,
        "LOW_LIQUIDITY": -1.0
    }
    
    score += regime_scores.get(regime, 0.0)
    
    if scalp_safe:
        score += 2.0
    
    if direction == "BULLISH":
        score += 1.5
    elif direction == "BEARISH":
        score += 1.0
    
    if 0.5 <= atr_percent <= 1.5:
        score += 2.0
    elif atr_percent < 0.3:
        score -= 1.0
    
    return float(score)

# ==================== ICHIMOKU ANALYSIS WITH DUAL CHIKOU ====================
def get_ichimoku(df: pd.DataFrame) -> Dict[str, Any]:
    try:
        if len(df) < 78:  # نیاز به حداقل 78 کندل برای چیکوی 78
            return {
                "trend": "NEUTRAL", 
                "price_above_cloud": False, 
                "signal": "NO_DATA",
                "ichimoku_score": 0,
                "dual_chikou_signal": "NEUTRAL",
                "dual_chikou_boost": 1.0,
                "dual_chikou_confidence": 0.0,
                "dual_chikou_details": {}
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
        
        # تحلیل دو چیکوی آینده
        dual_chikou_signal = "NEUTRAL"
        dual_chikou_boost = 1.0
        dual_chikou_confidence = 0.0
        dual_chikou_details = {}
        
        if len(df) >= 78:
            # قیمت ۲۶ دوره قبل (برای چیکوی ۲۶ کندل جلوتر)
            price_26_periods_ago = df['Close'].iloc[-26]
            
            # قیمت ۵۲ دوره قبل (برای چیکوی ۷۸ کندل جلوتر)
            price_52_periods_ago = df['Close'].iloc[-52]
            
            # تشخیص روند برای تطابق با چیکو
            trend = "NEUTRAL"
            if current_price > senkou_a.iloc[-1] and current_price > senkou_b.iloc[-1]:
                trend = "bullish"
            elif current_price < senkou_a.iloc[-1] and current_price < senkou_b.iloc[-1]:
                trend = "bearish"
            
            # تحلیل دو چیکو
            dual_chikou_analysis = analyze_dual_chikou_future(
                current_price=current_price,
                price_26_periods_ago=price_26_periods_ago,
                price_52_periods_ago=price_52_periods_ago,
                tenkan_current=tenkan.iloc[-1],
                kijun_current=kijun.iloc[-1],
                trend_direction=trend
            )
            
            dual_chikou_signal = dual_chikou_analysis['signal']
            dual_chikou_boost = dual_chikou_analysis['boost_multiplier']
            dual_chikou_confidence = dual_chikou_analysis['confidence']
            dual_chikou_details = {
                'chikou_26_price': dual_chikou_analysis.get('chikou_26_price', 0),
                'chikou_78_price': dual_chikou_analysis.get('chikou_78_price', 0),
                'chikou_26_diff': dual_chikou_analysis.get('chikou_26_diff', 0),
                'chikou_78_diff': dual_chikou_analysis.get('chikou_78_diff', 0),
                'both_above': dual_chikou_analysis.get('both_above', False),
                'both_below': dual_chikou_analysis.get('both_below', False),
                'reasons': dual_chikou_analysis.get('reasons', [])
            }
        
        price_above_cloud = current_price > max(senkou_a.iloc[-1], senkou_b.iloc[-1])
        price_below_cloud = current_price < min(senkou_a.iloc[-1], senkou_b.iloc[-1])
        price_in_cloud = not (price_above_cloud or price_below_cloud)
        
        trend = "NEUTRAL"
        signal = "HOLD"
        
        tenkan_kijun_diff = ((tenkan.iloc[-1] / kijun.iloc[-1]) - 1) * 100 if kijun.iloc[-1] > 0 else 0
        
        # سیگنال پایه ایچیموکو
        base_signal = "HOLD"
        if tenkan.iloc[-1] > kijun.iloc[-1] and price_above_cloud:
            trend = "STRONG_BULLISH"
            base_signal = "BUY"
        elif tenkan.iloc[-1] > kijun.iloc[-1]:
            trend = "BULLISH"
            base_signal = "BUY"
        elif tenkan.iloc[-1] < kijun.iloc[-1] and price_below_cloud:
            trend = "STRONG_BEARISH"
            base_signal = "SELL"
        elif tenkan.iloc[-1] < kijun.iloc[-1]:
            trend = "BEARISH"
            base_signal = "SELL"
        
        # ترکیب با سیگنال دو چیکو
        if base_signal == "BUY" and dual_chikou_signal in ["STRONG_BUY", "WEAK_BUY"]:
            if dual_chikou_signal == "STRONG_BUY":
                signal = "STRONG_BUY"
            else:
                signal = "BUY"
        elif base_signal == "SELL" and dual_chikou_signal in ["STRONG_SELL", "WEAK_SELL"]:
            if dual_chikou_signal == "STRONG_SELL":
                signal = "STRONG_SELL"
            else:
                signal = "SELL"
        elif base_signal == "BUY" and dual_chikou_signal in ["STRONG_SELL", "WEAK_SELL"]:
            signal = "WEAK_BUY"
        elif base_signal == "SELL" and dual_chikou_signal in ["STRONG_BUY", "WEAK_BUY"]:
            signal = "WEAK_SELL"
        else:
            signal = base_signal
        
        # امتیاز ایچیموکو
        ichimoku_score = calculate_ichimoku_score(
            trend=trend, 
            signal=signal, 
            price_above_cloud=price_above_cloud, 
            price_below_cloud=price_below_cloud, 
            tenkan_kijun_diff=tenkan_kijun_diff,
            dual_chikou_signal=dual_chikou_signal,
            dual_chikou_boost=dual_chikou_boost,
            dual_chikou_confidence=dual_chikou_confidence
        )
        
        return {
            "trend": trend,
            "signal": signal,
            "price_above_cloud": price_above_cloud,
            "price_below_cloud": price_below_cloud,
            "price_in_cloud": price_in_cloud,
            "tenkan": float(tenkan.iloc[-1]),
            "kijun": float(kijun.iloc[-1]),
            "tenkan_kijun_diff_pct": float(tenkan_kijun_diff),
            "cloud_top": float(max(senkou_a.iloc[-1], senkou_b.iloc[-1])),
            "cloud_bottom": float(min(senkou_a.iloc[-1], senkou_b.iloc[-1])),
            "price_vs_kijun": float((current_price / kijun.iloc[-1] - 1) * 100),
            "ichimoku_score": float(ichimoku_score),
            "dual_chikou_signal": dual_chikou_signal,
            "dual_chikou_boost": float(dual_chikou_boost),
            "dual_chikou_confidence": float(dual_chikou_confidence),
            "dual_chikou_details": dual_chikou_details
        }
        
    except Exception as e:
        logger.error(f"Ichimoku Error: {type(e).__name__}: {str(e)}")
        return {
            "trend": "NEUTRAL", 
            "price_above_cloud": False, 
            "signal": "ERROR",
            "ichimoku_score": 0,
            "dual_chikou_signal": "NEUTRAL",
            "dual_chikou_boost": 1.0,
            "dual_chikou_confidence": 0.0,
            "dual_chikou_details": {}
        }

def calculate_ichimoku_score(trend: str, signal: str, price_above_cloud: bool, 
                            price_below_cloud: bool, tenkan_kijun_diff: float,
                            dual_chikou_signal: str = "NEUTRAL", 
                            dual_chikou_boost: float = 1.0,
                            dual_chikou_confidence: float = 0.0) -> float:
    score = 0.0
    
    trend_scores = {
        "STRONG_BULLISH": 3.0,
        "BULLISH": 2.0,
        "STRONG_BEARISH": -3.0,
        "BEARISH": -2.0,
        "NEUTRAL": 0.0
    }
    
    score += trend_scores.get(trend, 0.0)
    
    signal_scores = {
        "STRONG_BUY": 3.0,
        "BUY": 2.0,
        "WEAK_BUY": 1.0,
        "STRONG_SELL": -3.0,
        "SELL": -2.0,
        "WEAK_SELL": -1.0,
        "HOLD": 0.0
    }
    
    score += signal_scores.get(signal, 0.0)
    
    if price_above_cloud:
        score += 2.0
    elif price_below_cloud:
        score -= 2.0
    
    if tenkan_kijun_diff > 0.5:
        score += 1.0
    elif tenkan_kijun_diff < -0.5:
        score -= 1.0
    
    # اعمال ضریب دو چیکو
    score *= dual_chikou_boost
    
    # اضافه کردن امتیاز بر اساس اعتماد چیکو
    if dual_chikou_confidence > 0.5:
        score += dual_chikou_confidence * 2.0
    
    # امتیاز اضافی برای تطابق کامل دو چیکو
    if dual_chikou_signal in ["STRONG_BUY", "STRONG_SELL"]:
        score += 2.0
    
    return float(score)

# ==================== EXIT LEVELS CALCULATOR ====================
def get_exit_levels(price: float, stop_loss: float, 
                   direction: str = "BUY", 
                   scalping_mode: bool = True,
                   volatility_pct: float = 1.0) -> Dict[str, Any]:
    try:
        direction = direction.upper()
        if direction not in ["BUY", "SELL"]:
            direction = "BUY"
        
        risk = abs(price - stop_loss)
        if risk == 0: 
            risk = price * 0.01
        
        multiplier = 1 if direction == "BUY" else -1
        
        if scalping_mode:
            tp1_ratio = 0.7 + (volatility_pct * 0.1)
            tp2_ratio = 1.5 + (volatility_pct * 0.2)
            trailing_activation_ratio = 0.5
            partial_exit_pct = 0.4
            breakeven_ratio = 0.25
        else:
            tp1_ratio = 0.5 + (volatility_pct * 0.1)
            tp2_ratio = 2.0 + (volatility_pct * 0.3)
            trailing_activation_ratio = 0.3
            partial_exit_pct = 0.3
            breakeven_ratio = 0.15
        
        tp1_ratio = max(0.3, min(tp1_ratio, 1.5))
        tp2_ratio = max(1.0, min(tp2_ratio, 3.0))
        
        tp1 = price + (risk * tp1_ratio * multiplier)
        tp2 = price + (risk * tp2_ratio * multiplier)
        
        trailing_activation = price + (risk * trailing_activation_ratio * multiplier)
        breakeven_level = price + (risk * breakeven_ratio * multiplier)
        
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

# ==================== SCALP SIGNAL GENERATOR WITH DUAL CHIKOU ====================
def generate_scalp_signals(df: pd.DataFrame, test_mode: bool = False, 
                          force_signal: Optional[str] = None) -> Dict[str, Any]:
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
            score += 5.0
            reasons.append("قیمت در ناحیه ارزان حجمی (CHEAP)")
            scoring_details["volume_zone"] = 5.0
        elif vp_zone == "EXPENSIVE":
            score -= 5.0
            reasons.append("قیمت در ناحیه گران حجمی (EXPENSIVE)")
            scoring_details["volume_zone"] = -5.0
        
        if volume_profile.get('in_value_area', False):
            score += 2.0
            reasons.append("قیمت در محدوده ارزش (Value Area)")
            scoring_details["in_value_area"] = 2.0
            
        # اگر POC قوی باشد
        poc_strength = volume_profile.get('poc_strength', 0)
        if poc_strength > 15:
            score += 3.0
            reasons.append(f"نقطه کنترل حجمی قوی (POC: {poc_strength:.1f}%)")
            scoring_details["poc_strength"] = 3.0
        elif poc_strength > 10:
            score += 1.0
            reasons.append(f"نقطه کنترل حجمی متوسط (POC: {poc_strength:.1f}%)")
            scoring_details["poc_strength"] = 1.0
        
        # 2.2 امتیاز Market Regime
        if market_regime.get('scalp_safe', False):
            score += 5.0
            reasons.append("بازار برای اسکالپ امن است")
            scoring_details["scalp_safe"] = 5.0
        else:
            reasons.append(f"بازار برای اسکالپ مناسب نیست (رژیم: {market_regime.get('regime', 'UNKNOWN')})")
            scoring_details["scalp_safe"] = 0.0
        
        direction = market_regime.get('direction', 'NEUTRAL')
        if direction == "BULLISH":
            score += 3.0
            reasons.append("روند صعودی")
            scoring_details["direction"] = 3.0
        elif direction == "BEARISH":
            score += 2.0
            reasons.append("روند نزولی")
            scoring_details["direction"] = 2.0
        
        # اضافه کردن امتیاز رژیم
        regime_score = market_regime.get('regime_score', 0)
        score += regime_score
        scoring_details["regime_score"] = regime_score
        
        # 2.3 امتیاز Ichimoku با دو چیکو
        ichimoku_signal = ichimoku.get('signal', 'HOLD')
        ichimoku_score = ichimoku.get('ichimoku_score', 0)
        dual_chikou_signal = ichimoku.get('dual_chikou_signal', 'NEUTRAL')
        dual_chikou_boost = ichimoku.get('dual_chikou_boost', 1.0)
        dual_chikou_details = ichimoku.get('dual_chikou_details', {})
        
        # اعمال ضریب افزایش دو چیکو
        ichimoku_score *= dual_chikou_boost
        score += ichimoku_score
        scoring_details["ichimoku_score"] = ichimoku_score
        scoring_details["dual_chikou_boost"] = dual_chikou_boost
        
        # ادامه منطق امتیازدهی Ichimoku و Dual Chikou
        if "BUY" in ichimoku_signal:
            reasons.append(f"تاییدیه ایچیموکو: {ichimoku_signal}")
            if dual_chikou_details.get('both_below'):
                reasons.append("🚀 تاییدیه طلایی: هر دو چیکو زیر قیمت (سیگنال خرید قوی)")
        elif "SELL" in ichimoku_signal:
            reasons.append(f"تاییدیه ایچیموکو: {ichimoku_signal}")
            if dual_chikou_details.get('both_above'):
                reasons.append("🔻 تاییدیه طلایی: هر دو چیکو بالای قیمت (سیگنال فروش قوی)")

        # اضافه کردن جزئیات دلایل چیکو به لیست دلایل نهایی
        for r in dual_chikou_details.get('reasons', []):
            reasons.append(f"Chikou: {r}")
        
        # اطلاعات اختلاف چیکوها
        chikou_26_diff = dual_chikou_details.get('chikou_26_diff', 0)
        chikou_78_diff = dual_chikou_details.get('chikou_78_diff', 0)
        if chikou_26_diff != 0 or chikou_78_diff != 0:
            reasons.append(f"اختلاف چیکوها: 26کندل={chikou_26_diff:.2f}%, 78کندل={chikou_78_diff:.2f}%")
        
        if ichimoku_signal == "BUY":
            reasons.append(f"سیگنال ایچیموکو: {ichimoku.get('trend', 'NEUTRAL')}")
        elif ichimoku_signal == "SELL":
            reasons.append(f"سیگنال ایچیموکو: {ichimoku.get('trend', 'NEUTRAL')}")
        
        if ichimoku.get('price_above_cloud', False):
            reasons.append("قیمت بالای ابر کومو (مثبت)")
        elif ichimoku.get('price_below_cloud', False):
            reasons.append("قیمت زیر ابر کومو (منفی)")
        
        # 2.4 امتیاز حجم لحظه‌ای (0-4 امتیاز)
        if 'Volume' in df.columns:
            avg_volume = df['Volume'].rolling(window=20).mean().iloc[-1]
            current_volume = df['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            if volume_ratio > 1.5:
                score += 4.0
                reasons.append(f"حجم معاملات بالا ({volume_ratio:.1f}x میانگین)")
                scoring_details["volume_ratio"] = 4.0
            elif volume_ratio > 1.2:
                score += 2.0
                reasons.append(f"حجم معاملات مناسب ({volume_ratio:.1f}x میانگین)")
                scoring_details["volume_ratio"] = 2.0
            elif volume_ratio < 0.5:
                score -= 2.0
                reasons.append(f"حجم معاملات پایین ({volume_ratio:.1f}x میانگین)")
                scoring_details["volume_ratio"] = -2.0
        
        # 2.5 امتیاز نوسان مناسب برای اسکالپ (0-5 امتیاز)
        atr_percent = market_regime.get('atr_percent', 0)
        if 0.5 <= atr_percent <= 1.5:
            score += 5.0
            reasons.append(f"نوسان مناسب برای اسکالپ ({atr_percent:.2f}%)")
            scoring_details["atr_score"] = 5.0
        elif atr_percent > 2.0:
            score -= 3.0
            reasons.append(f"نوسان بسیار بالا ({atr_percent:.2f}%)")
            scoring_details["atr_score"] = -3.0
        elif atr_percent < 0.3:
            score -= 2.0
            reasons.append(f"نوسان بسیار پایین ({atr_percent:.2f}%)")
            scoring_details["atr_score"] = -2.0
        
        # 2.6 امتیاز تنکان/کیجون کراس (0-5 امتیاز)
        tenkan_kijun_diff = ichimoku.get('tenkan_kijun_diff_pct', 0)
        if abs(tenkan_kijun_diff) > 0.5:
            if tenkan_kijun_diff > 0:
                score += 5.0
                reasons.append(f"تنکان بالای کیجون ({tenkan_kijun_diff:.2f}%)")
                scoring_details["tenkan_kijun_cross"] = 5.0
            else:
                score -= 5.0
                reasons.append(f"تنکان زیر کیجون ({tenkan_kijun_diff:.2f}%)")
                scoring_details["tenkan_kijun_cross"] = -5.0
        
        # 3. نهایی کردن وضعیت سیگنال بر اساس امتیاز کل
        # نرمال‌سازی امتیاز برای تصمیم‌گیری (مقیاس حدود -50 تا +50)
        final_signal = "NEUTRAL"
        
        # حد نصاب برای ورود به معامله (قابل تنظیم)
        buy_threshold = 12.0
        sell_threshold = -12.0

        if score >= buy_threshold:
            final_signal = "STRONG_BUY" if score > 18.0 else "BUY"
        elif score <= sell_threshold:
            final_signal = "STRONG_SELL" if score < -18.0 else "SELL"

        # اجبار به تولید سیگنال در حالت تست (Force Signal)
        if force_signal:
            final_signal = force_signal
            score = 25.0 if "BUY" in force_signal else -25.0

        # 4. محاسبه سطوح خروج (TP/SL) در صورت وجود سیگنال
        exit_levels = None
        if final_signal != "NEUTRAL":
            # تعیین حد ضرر بر اساس کیجون یا کف/سقف اخیر
            sl_price = ichimoku.get('kijun', current_price * 0.99)
            
            # جلوگیری از SL خیلی نزدیک (حداقل فاصله بر اساس ATR)
            min_dist = current_price * (market_regime.get('atr_percent', 0.5) / 100)
            if abs(current_price - sl_price) < min_dist:
                sl_price = current_price - min_dist if "BUY" in final_signal else current_price + min_dist

            exit_levels = get_exit_levels(
                price=current_price,
                stop_loss=sl_price,
                direction="BUY" if "BUY" in final_signal else "SELL",
                scalping_mode=True,
                volatility_pct=market_regime.get('atr_percent', 1.0)
            )

        return {
            "valid": True,
            "score": round(score, 2),
            "signal": final_signal,
            "reasons": reasons,
            "exit_levels": exit_levels,
            "analysis": {
                "ichimoku": ichimoku,
                "volume": volume_profile,
                "regime": market_regime,
                "scoring_details": scoring_details
            }
        }

    except Exception as e:
        logger.error(f"Error in signal generation: {str(e)}")
        return {"valid": False, "score": 0, "signal": "ERROR", "reasons": [str(e)]}

# ==================== DATAFRAME VALIDATOR ====================
def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """بررسی صحت و کامل بودن دیتای ورودی"""
    errors = []
    if df is None or df.empty:
        return {"valid": False, "errors": ["دیتایی یافت نشد"]}
    
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in df.columns:
            errors.append(f"ستون {col} در دیتا وجود ندارد")
    
    # برای محاسبات Dual Chikou (78 کندل) و SMA50، حداقل 80 کندل نیاز داریم
    if len(df) < 80:
        errors.append(f"تعداد کندل ناکافی: نیاز به 80، موجود {len(df)}")
        
    return {"valid": len(errors) == 0, "errors": errors}

# ==================== HELPER FUNCTIONS ====================
def format_price(price: float) -> str:
    return f"{price:,.2f}"

def calculate_pivot_points(df: pd.DataFrame) -> Dict[str, float]:
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

# ==================== TEST FUNCTIONS ====================
def test_all_functions():
    logger.info("🧪 Running tests with Dual Chikou...")
    
    try:
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
        
        # تست تحلیل دو چیکو
        dual_test = analyze_dual_chikou_future(100, 102, 103, 101, 100, "bearish")
        logger.info(f"✅ Dual Chikou Analysis: Signal={dual_test.get('signal', 'N/A')}, Boost={dual_test.get('boost_multiplier', 1.0):.2f}x")
        
        # تست Ichimoku با دو چیکو
        ichi = get_ichimoku(df)
        logger.info(f"✅ Ichimoku with Dual Chikou: Signal={ichi.get('signal', 'N/A')}, Chikou Boost={ichi.get('dual_chikou_boost', 1.0):.2f}x")
        
        # تست سیگنال
        signals = generate_scalp_signals(df, test_mode=False)
        logger.info(f"✅ Signals: {signals.get('signal', 'N/A')}, Score={signals.get('score', 0):.1f}")
        
        logger.info("✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {type(e).__name__}: {str(e)}")
        return False

# ==================== SIGNAL FORMATTER ====================
def format_signal_message(symbol: str, signal_data: Dict[str, Any]) -> str:
    try:
        signal = signal_data.get('signal', 'HOLD')
        price = signal_data.get('price', 0)
        score = signal_data.get('score', 0)
        confidence = signal_data.get('confidence', 0)
        reasons = signal_data.get('reasons', [])
        dual_chikou = signal_data.get('dual_chikou_analysis', {})
        chikou_boost = dual_chikou.get('boost', 1.0)
        
        emoji_map = {
            "STRONG_BUY": "🚀",
            "BUY": "🟢",
            "STRONG_SELL": "🔻",
            "SELL": "🔴",
            "HOLD": "⏸️",
            "TEST": "🧪"
        }
        
        emoji = emoji_map.get(signal, "📊")
        
        lines = [
            f"{emoji} *{signal}*",
            f"`{symbol}`",
            f"💰 قیمت: {format_price(price)}",
            f"📊 امتیاز: {score:.1f}"
        ]
        
        if chikou_boost > 1.0:
            lines.append(f"📈 افزایش اعتبار دو چیکو: {chikou_boost:.2f}x")
        
        lines.append("")
        lines.append("*دلایل:*")
        
        for i, reason in enumerate(reasons[:5]):
            lines.append(f"• {reason}")
        
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
    test_all_functions()
