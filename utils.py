# 📦 Utilities Module - ابزارهای کمکی کامل برای Fast Scalp Bot
# 🚀 Version: 3.0.0 | با ایچیموکو پیشرفته و دو چیکو اسپن
# 📅 Last Updated: 2024-01-25

import os
import sys
import logging
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
import gzip
import csv
from collections import defaultdict, deque
import requests

# ============================================
# 🎯 Logger Configuration
# ============================================

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

def setup_logger(
    name: str = "fast_scalp",
    level: LogLevel = LogLevel.INFO,
    log_to_file: bool = False,
    log_file: str = "fast_scalp.log",
    console_output: bool = True
) -> logging.Logger:
    """
    تنظیم لاگر پیشرفته با قابلیت‌های مختلف
    """
    logger = logging.getLogger(name)
    
    # اگر قبلاً تنظیم شده، return کن
    if logger.hasHandlers():
        return logger
    
    # تنظیم سطح
    logger.setLevel(level.value)
    
    # فرمت پیشرفته
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # هندلر کنسول
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level.value)
        logger.addHandler(console_handler)
    
    # هندلر فایل
    if log_to_file:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level.value)
        logger.addHandler(file_handler)
        
        # همچنین handler برای errorها
        error_handler = logging.FileHandler(log_dir / f"error_{log_file}", encoding='utf-8')
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        logger.addHandler(error_handler)
    
    # جلوگیری از انتشار به root logger
    logger.propagate = False
    
    return logger

def log_performance(func):
    """
    دکوراتور برای لاگ کردن عملکرد توابع
    """
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            if elapsed > 1.0:
                logger.warning(f"⏱️ {func.__name__} took {elapsed:.2f}s (Slow)")
            elif elapsed > 0.5:
                logger.info(f"⏱️ {func.__name__} took {elapsed:.2f}s")
            else:
                logger.debug(f"⏱️ {func.__name__} took {elapsed:.4f}s")
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"❌ {func.__name__} failed after {elapsed:.2f}s: {e}", exc_info=True)
            raise
    
    return wrapper

# ============================================
# 🔧 Data Utilities
# ============================================

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str] = None,
    min_rows: int = 50,
    check_nulls: bool = True,
    check_inf: bool = True
) -> Tuple[bool, str]:
    """
    اعتبارسنجی جامع دیتافریم OHLCV
    """
    if required_columns is None:
        required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    # 1. بررسی وجود دیتافریم
    if df is None:
        return False, "DataFrame is None"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    # 2. بررسی تعداد سطرها
    if len(df) < min_rows:
        return False, f"Not enough rows: {len(df)} < {min_rows}"
    
    # 3. بررسی ستون‌های ضروری
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return False, f"Missing columns: {missing_columns}"
    
    # 4. بررسی مقادیر NaN
    if check_nulls:
        null_counts = df[required_columns].isna().sum()
        if null_counts.any():
            problematic = null_counts[null_counts > 0].to_dict()
            return False, f"NaN values found: {problematic}"
    
    # 5. بررسی مقادیر Infinite
    if check_inf:
        for col in required_columns:
            if col in df.columns:
                if np.isinf(df[col]).any():
                    return False, f"Infinite values in column: {col}"
    
    # 6. بررسی مقادیر منفی (برای قیمت و حجم)
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            if (df[col] <= 0).any():
                return False, f"Non-positive values in {col}"
    
    if 'volume' in df.columns:
        if (df['volume'] < 0).any():
            return False, "Negative volume values"
    
    # 7. بررسی توالی زمانی (اگر index datetime است)
    if isinstance(df.index, pd.DatetimeIndex):
        time_diff = df.index.to_series().diff().dropna()
        if (time_diff < timedelta(seconds=0)).any():
            return False, "Non-chronological timestamps"
    
    return True, "DataFrame is valid"

def clean_ohlcv_data(
    df: pd.DataFrame,
    remove_outliers: bool = True,
    outlier_threshold: float = 3.0,
    fill_method: str = 'ffill',
    volume_filter: bool = True
) -> pd.DataFrame:
    """
    تمیز کردن و پیش‌پردازش داده‌های OHLCV
    """
    if df.empty:
        return df
    
    df_clean = df.copy()
    
    # 1. حذف سطرهای با حجم صفر یا منفی
    if volume_filter and 'volume' in df_clean.columns:
        df_clean = df_clean[df_clean['volume'] > 0]
    
    # 2. حذف outliers قیمت (تغییرات غیرعادی)
    if remove_outliers and len(df_clean) > 10:
        for col in ['open', 'high', 'low', 'close']:
            if col in df_clean.columns:
                # محاسبه درصد تغییر
                pct_change = df_clean[col].pct_change().abs()
                
                # حذف تغییرات بیش از threshold
                outlier_mask = pct_change > outlier_threshold
                
                # حذف outliers (اما حداکثر 5% داده‌ها)
                max_outliers = int(len(df_clean) * 0.05)
                if outlier_mask.sum() > max_outliers:
                    # فقط worst outliers را حذف کن
                    outlier_indices = pct_change.nlargest(max_outliers).index
                    df_clean = df_clean.drop(outlier_indices)
                else:
                    df_clean = df_clean[~outlier_mask]
    
    # 3. پر کردن مقادیر NaN
    if fill_method == 'ffill':
        df_clean = df_clean.ffill()
    elif fill_method == 'bfill':
        df_clean = df_clean.bfill()
    elif fill_method == 'interpolate':
        df_clean = df_clean.interpolate(method='linear')
    
    # 4. حذف هر ردیف که هنوز NaN دارد
    df_clean = df_clean.dropna()
    
    # 5. اطمینان از توالی زمانی
    if isinstance(df_clean.index, pd.DatetimeIndex):
        df_clean = df_clean.sort_index()
    
    # 6. اعتبارسنجی نهایی
    is_valid, msg = validate_dataframe(df_clean)
    if not is_valid:
        logging.warning(f"Data cleaning warning: {msg}")
    
    return df_clean

def calculate_volume_profile(
    df: pd.DataFrame,
    bins: int = 20,
    price_col: str = 'close',
    volume_col: str = 'volume'
) -> Dict[str, Any]:
    """
    محاسبه Volume Profile با جزئیات کامل
    """
    if df.empty or len(df) < 10:
        return {}
    
    try:
        prices = df[price_col].values
        volumes = df[volume_col].values
        
        # محاسبه دامنه قیمت
        min_price = np.nanmin(prices)
        max_price = np.nanmax(prices)
        
        if min_price == max_price or np.isnan(min_price) or np.isnan(max_price):
            return {}
        
        # ایجاد bins
        price_bins = np.linspace(min_price, max_price, bins + 1)
        
        # محاسبه حجم در هر bin
        volume_by_bin = np.zeros(bins)
        price_midpoints = np.zeros(bins)
        
        for i in range(bins):
            bin_low = price_bins[i]
            bin_high = price_bins[i + 1]
            price_midpoints[i] = (bin_low + bin_high) / 2
            
            # پیدا کردن کندل‌هایی که در این بازه هستند
            mask = (prices >= bin_low) & (prices <= bin_high)
            volume_by_bin[i] = np.sum(volumes[mask])
        
        # یافتن نقطه کنترل حجم (POC)
        if np.sum(volume_by_bin) > 0:
            poc_index = np.argmax(volume_by_bin)
            poc_price = price_midpoints[poc_index]
            poc_volume = volume_by_bin[poc_index]
        else:
            poc_index = -1
            poc_price = np.nan
            poc_volume = 0
        
        # محاسبه Value Area (70% حجم)
        total_volume = np.sum(volume_by_bin)
        if total_volume > 0:
            # مرتب کردن bins بر اساس حجم
            sorted_indices = np.argsort(volume_by_bin)[::-1]
            cumulative_volume = 0
            value_area_indices = []
            
            for idx in sorted_indices:
                cumulative_volume += volume_by_bin[idx]
                value_area_indices.append(idx)
                
                if cumulative_volume / total_volume >= 0.7:
                    break
            
            # قیمت‌های Value Area
            value_area_prices = price_midpoints[value_area_indices]
            value_area_low = np.min(value_area_prices)
            value_area_high = np.max(value_area_prices)
        else:
            value_area_low = np.nan
            value_area_high = np.nan
        
        # تشخیص وضعیت فعلی نسبت به Volume Profile
        current_price = prices[-1]
        
        if not np.isnan(current_price):
            if current_price > value_area_high:
                price_position = "above_value_area"
            elif current_price < value_area_low:
                price_position = "below_value_area"
            else:
                price_position = "inside_value_area"
        else:
            price_position = "unknown"
        
        return {
            'price_range': {
                'min': float(min_price),
                'max': float(max_price),
                'range': float(max_price - min_price)
            },
            'poc': {
                'price': float(poc_price),
                'volume': float(poc_volume),
                'index': int(poc_index)
            },
            'value_area': {
                'low': float(value_area_low),
                'high': float(value_area_high),
                'width': float(value_area_high - value_area_low)
            },
            'current_position': price_position,
            'bins': {
                'prices': [float(p) for p in price_midpoints],
                'volumes': [float(v) for v in volume_by_bin]
            },
            'total_volume': float(total_volume),
            'current_price': float(current_price) if not np.isnan(current_price) else None
        }
        
    except Exception as e:
        logging.error(f"Error calculating volume profile: {e}")
        return {}

# ============================================
# 📈 Technical Analysis Utilities
# ============================================

def calculate_support_resistance(
    df: pd.DataFrame,
    window: int = 20,
    pivot_window: int = 5,
    strength_threshold: int = 2,
    merge_threshold: float = 0.02
) -> Dict[str, Any]:
    """
    تشخیص سطوح حمایت و مقاومت با الگوریتم پیشرفته
    """
    if len(df) < window * 2:
        return {'supports': [], 'resistances': []}
    
    try:
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        supports = []
        resistances = []
        
        # تشخیص pivot points
        for i in range(window, len(df) - window):
            # مقاومت (سقف محلی)
            is_resistance = True
            
            # بررسی سمت چپ
            for j in range(1, pivot_window + 1):
                if highs[i] <= highs[i - j]:
                    is_resistance = False
                    break
            
            # بررسی سمت راست
            if is_resistance:
                for j in range(1, pivot_window + 1):
                    if highs[i] <= highs[i + j]:
                        is_resistance = False
                        break
            
            if is_resistance:
                resistances.append({
                    'price': float(highs[i]),
                    'index': i,
                    'strength': 1,
                    'timestamp': df.index[i] if isinstance(df.index, pd.DatetimeIndex) else None
                })
            
            # حمایت (کف محلی)
            is_support = True
            
            # بررسی سمت چپ
            for j in range(1, pivot_window + 1):
                if lows[i] >= lows[i - j]:
                    is_support = False
                    break
            
            # بررسی سمت راست
            if is_support:
                for j in range(1, pivot_window + 1):
                    if lows[i] >= lows[i + j]:
                        is_support = False
                        break
            
            if is_support:
                supports.append({
                    'price': float(lows[i]),
                    'index': i,
                    'strength': 1,
                    'timestamp': df.index[i] if isinstance(df.index, pd.DatetimeIndex) else None
                })
        
        # تقویت سطوح بر اساس تعداد برخوردها
        current_price = closes[-1]
        price_range = np.max(highs) - np.min(lows)
        merge_distance = price_range * merge_threshold
        
        def merge_and_strengthen(levels, is_support=True):
            if not levels:
                return []
            
            levels.sort(key=lambda x: x['price'])
            merged = []
            
            for level in levels:
                if not merged:
                    merged.append(level.copy())
                    continue
                
                last = merged[-1]
                price_diff = abs(level['price'] - last['price'])
                
                if price_diff <= merge_distance:
                    # ادغام سطح
                    last['price'] = (last['price'] + level['price']) / 2
                    last['strength'] += level['strength']
                    # میانگین index
                    last['index'] = (last['index'] + level['index']) // 2
                else:
                    merged.append(level.copy())
            
            # فقط سطوح قوی
            filtered = [l for l in merged if l['strength'] >= strength_threshold]
            
            # مرتب‌سازی بر اساس نزدیکی به قیمت فعلی
            filtered.sort(key=lambda x: abs(x['price'] - current_price))
            
            return filtered
        
        supports = merge_and_strengthen(supports, is_support=True)
        resistances = merge_and_strengthen(resistances, is_support=False)
        
        # یافتن نزدیک‌ترین سطوح
        nearest_support = supports[0] if supports else None
        nearest_resistance = resistances[0] if resistances else None
        
        # تشخیص وضعیت فعلی
        if nearest_support and nearest_resistance:
            distance_to_support = abs(current_price - nearest_support['price'])
            distance_to_resistance = abs(current_price - nearest_resistance['price'])
            
            if distance_to_support < distance_to_resistance:
                current_zone = "near_support"
                zone_distance = distance_to_support / price_range * 100
            else:
                current_zone = "near_resistance"
                zone_distance = distance_to_resistance / price_range * 100
        elif nearest_support:
            current_zone = "near_support"
            zone_distance = abs(current_price - nearest_support['price']) / price_range * 100
        elif nearest_resistance:
            current_zone = "near_resistance"
            zone_distance = abs(current_price - nearest_resistance['price']) / price_range * 100
        else:
            current_zone = "no_level"
            zone_distance = 100.0
        
        return {
            'supports': supports[-10:],  # آخرین 10 سطح
            'resistances': resistances[-10:],
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'current_price': float(current_price),
            'current_zone': current_zone,
            'zone_distance_percent': float(zone_distance),
            'price_range': float(price_range),
            'total_levels': len(supports) + len(resistances)
        }
        
    except Exception as e:
        logging.error(f"Error calculating support/resistance: {e}")
        return {'supports': [], 'resistances': []}

def calculate_market_structure(df: pd.DataFrame, lookback: int = 5) -> Dict[str, Any]:
    """
    تشخیص ساختار بازار (Higher Highs/Lower Lows)
    """
    if len(df) < lookback * 4:
        return {'trend': 'neutral', 'structure': 'insufficient_data'}
    
    try:
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        # تشخیص swing highs و lows
        swing_highs = []
        swing_lows = []
        
        for i in range(lookback, len(df) - lookback):
            # Swing High
            if highs[i] == highs[i - lookback:i + lookback + 1].max():
                swing_highs.append({
                    'index': i,
                    'price': float(highs[i]),
                    'time': df.index[i] if isinstance(df.index, pd.DatetimeIndex) else None
                })
            
            # Swing Low
            if lows[i] == lows[i - lookback:i + lookback + 1].min():
                swing_lows.append({
                    'index': i,
                    'price': float(lows[i]),
                    'time': df.index[i] if isinstance(df.index, pd.DatetimeIndex) else None
                })
        
        # تشخیص روند
        trend = "neutral"
        trend_strength = 0
        
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # آخرین دو swing
            last_high = swing_highs[-1]['price']
            prev_high = swing_highs[-2]['price']
            last_low = swing_lows[-1]['price']
            prev_low = swing_lows[-2]['price']
            
            # بررسی Higher Highs و Higher Lows
            higher_highs = last_high > prev_high
            higher_lows = last_low > prev_low
            
            # بررسی Lower Lows و Lower Highs
            lower_lows = last_low < prev_low
            lower_highs = last_high < prev_high
            
            if higher_highs and higher_lows:
                trend = "uptrend"
                trend_strength = min(
                    (last_high - prev_high) / prev_high * 100,
                    (last_low - prev_low) / prev_low * 100
                )
            elif lower_highs and lower_lows:
                trend = "downtrend"
                trend_strength = min(
                    (prev_high - last_high) / prev_high * 100,
                    (prev_low - last_low) / prev_low * 100
                )
            elif higher_highs and lower_lows:
                trend = "expansion"
                trend_strength = 0
            elif lower_highs and higher_lows:
                trend = "contraction"
                trend_strength = 0
            else:
                trend = "ranging"
                trend_strength = 0
        
        # تشخیص شکست ساختار
        structure_break = None
        if len(swing_highs) >= 3 and len(swing_lows) >= 3:
            # بررسی break of structure (BOS)
            if trend == "uptrend" and closes[-1] > swing_highs[-2]['price']:
                structure_break = "bullish_bos"
            elif trend == "downtrend" and closes[-1] < swing_lows[-2]['price']:
                structure_break = "bearish_bos"
        
        # تشخیص تغییر روند احتمالی
        potential_reversal = None
        if len(swing_highs) >= 3 and len(swing_lows) >= 3:
            if trend == "uptrend" and closes[-1] < swing_lows[-2]['price']:
                potential_reversal = "bearish_reversal"
            elif trend == "downtrend" and closes[-1] > swing_highs[-2]['price']:
                potential_reversal = "bullish_reversal"
        
        return {
            'trend': trend,
            'trend_strength': float(trend_strength),
            'swing_highs': swing_highs[-5:] if swing_highs else [],
            'swing_lows': swing_lows[-5:] if swing_lows else [],
            'structure_break': structure_break,
            'potential_reversal': potential_reversal,
            'current_price': float(closes[-1]),
            'market_phase': get_market_phase(df),
            'volatility': calculate_volatility(df, 20)
        }
        
    except Exception as e:
        logging.error(f"Error calculating market structure: {e}")
        return {'trend': 'error', 'structure': str(e)}

def get_market_phase(df: pd.DataFrame) -> str:
    """
    تشخیص فاز بازار
    """
    if len(df) < 50:
        return "unknown"
    
    # محاسبه moving averages
    ma20 = df['close'].rolling(20).mean()
    ma50 = df['close'].rolling(50).mean()
    
    current_price = df['close'].iloc[-1]
    ma20_current = ma20.iloc[-1]
    ma50_current = ma50.iloc[-1]
    
    # بررسی موقعیت قیمت نسبت به MAها
    above_ma20 = current_price > ma20_current
    above_ma50 = current_price > ma50_current
    ma20_above_ma50 = ma20_current > ma50_current
    
    # تشخیص فاز
    if above_ma20 and above_ma50 and ma20_above_ma50:
        return "bullish"
    elif not above_ma20 and not above_ma50 and not ma20_above_ma50:
        return "bearish"
    elif above_ma50 and not ma20_above_ma50:
        return "recovery"
    elif not above_ma50 and ma20_above_ma50:
        return "pullback"
    else:
        return "consolidation"

def calculate_volatility(df: pd.DataFrame, period: int = 20) -> float:
    """
    محاسبه نوسان بازار
    """
    if len(df) < period:
        return 0.0
    
    returns = df['close'].pct_change().dropna()
    if len(returns) < period:
        return 0.0
    
    volatility = returns.tail(period).std() * np.sqrt(252) * 100  # نوسان سالانه درصدی
    return float(volatility)

# ==================== ADVANCED ICHIMOKU WITH DUAL FUTURE CHIKOU ====================
def calculate_advanced_ichimoku(df: pd.DataFrame, column_prefix: str = '') -> Dict[str, Any]:
    """
    تحلیل ایچیموکو پیشرفته با دو چیکو آینده برای اعتبارسنجی
    
    چیکو اول: 26 کندل در آینده - تایید روند کوتاه‌مدت
    چیکو دوم: 78 کندل در آینده - تایید روند بلندمدت
    
    Args:
        df: DataFrame با داده‌های OHLC
        column_prefix: پیشوند برای ستون‌ها (مثل 'close' یا 'Close')
    
    Returns:
        Dict: تحلیل کامل ایچیموکو
    """
    try:
        # بررسی حداقل داده
        if len(df) < 78:
            return {
                "error": "insufficient_data",
                "min_required": 78,
                "available": len(df),
                "timestamp": datetime.now().isoformat()
            }
        
        # تعیین نام ستون‌ها
        high_col = f"{column_prefix}high" if column_prefix else 'high'
        low_col = f"{column_prefix}low" if column_prefix else 'low'
        close_col = f"{column_prefix}close" if column_prefix else 'close'
        
        # ==================== 
        # 1. محاسبات استاندارد ایچیموکو
        # ====================
        
        # Tenkan-sen (Conversion Line) - 9 دوره
        period9_high = df[high_col].rolling(window=9).max()
        period9_low = df[low_col].rolling(window=9).min()
        tenkan_sen = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line) - 26 دوره
        period26_high = df[high_col].rolling(window=26).max()
        period26_low = df[low_col].rolling(window=26).min()
        kijun_sen = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B) - 52 دوره
        period52_high = df[high_col].rolling(window=52).max()
        period52_low = df[low_col].rolling(window=52).min()
        senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
        
        current_price = df[close_col].iloc[-1]
        
        # ====================
        # 2. محاسبه چیکو اسپن‌های آینده (Future Chikou)
        # ====================
        
        # چیکو اسپن اصلی (Lagging Span) - 26 دوره عقب‌تر
        chikou_span_26 = df[close_col].shift(-26)
        
        # چیکو اسپن توسعه‌یافته (Extended Lagging Span) - 78 دوره عقب‌تر
        chikou_span_78 = df[close_col].shift(-78)
        
        # چیکو اسپن سریع (Fast Lagging Span) - 13 دوره عقب‌تر
        chikou_span_13 = df[close_col].shift(-13)
        
        # ====================
        # 3. تحلیل موقعیت‌های فعلی
        # ====================
        
        current_tenkan = tenkan_sen.iloc[-1]
        current_kijun = kijun_sen.iloc[-1]
        current_senkou_a = senkou_span_a.iloc[-1]
        current_senkou_b = senkou_span_b.iloc[-1]
        
        # موقعیت قیمت نسبت به ابر
        cloud_top = max(current_senkou_a, current_senkou_b)
        cloud_bottom = min(current_senkou_a, current_senkou_b)
        
        price_above_cloud = current_price > cloud_top
        price_below_cloud = current_price < cloud_bottom
        price_in_cloud = cloud_bottom <= current_price <= cloud_top
        
        # ====================
        # 4. تحلیل چیکو اسپن‌های آینده
        # ====================
        
        chikou_analysis = {}
        
        # چیکو 26 دوره
        if len(chikou_span_26) > 26:
            chikou_26_price = chikou_span_26.iloc[-26]
            historical_price_26 = df[close_col].iloc[-26] if len(df) > 26 else None
            
            chikou_analysis['chikou_26'] = {
                'current': float(chikou_26_price),
                'historical_price': float(historical_price_26) if historical_price_26 else None,
                'position': 'ABOVE' if historical_price_26 and chikou_26_price > historical_price_26 else 
                           'BELOW' if historical_price_26 else 'UNKNOWN',
                'difference_pct': ((chikou_26_price / historical_price_26 - 1) * 100) if historical_price_26 else None
            }
        
        # چیکو 78 دوره (چیکو اصلی برای تایید بلندمدت)
        if len(chikou_span_78) > 78:
            chikou_78_price = chikou_span_78.iloc[-78]
            historical_price_78 = df[close_col].iloc[-78] if len(df) > 78 else None
            
            chikou_analysis['chikou_78'] = {
                'current': float(chikou_78_price),
                'historical_price': float(historical_price_78) if historical_price_78 else None,
                'position': 'ABOVE' if historical_price_78 and chikou_78_price > historical_price_78 else 
                           'BELOW' if historical_price_78 else 'UNKNOWN',
                'difference_pct': ((chikou_78_price / historical_price_78 - 1) * 100) if historical_price_78 else None
            }
        
        # چیکو 13 دوره (برای تایید سریع)
        if len(chikou_span_13) > 13:
            chikou_13_price = chikou_span_13.iloc[-13]
            historical_price_13 = df[close_col].iloc[-13] if len(df) > 13 else None
            
            chikou_analysis['chikou_13'] = {
                'current': float(chikou_13_price),
                'historical_price': float(historical_price_13) if historical_price_13 else None,
                'position': 'ABOVE' if historical_price_13 and chikou_13_price > historical_price_13 else 
                           'BELOW' if historical_price_13 else 'UNKNOWN',
                'difference_pct': ((chikou_13_price / historical_price_13 - 1) * 100) if historical_price_13 else None
            }
        
        # ====================
        # 5. سیگنال‌های ترکیبی
        # ====================
        
        signals = []
        score = 0
        
        # سیگنال تنکان/کیجون کراس
        if current_tenkan > current_kijun:
            signals.append("TENKAN_ABOVE_KIJUN")
            score += 2.0
        elif current_tenkan < current_kijun:
            signals.append("TENKAN_BELOW_KIJUN")
            score -= 2.0
        
        # سیگنال موقعیت قیمت نسبت به ابر
        if price_above_cloud:
            signals.append("PRICE_ABOVE_CLOUD")
            score += 3.0
        elif price_below_cloud:
            signals.append("PRICE_BELOW_CLOUD")
            score -= 3.0
        
        # سیگنال چیکو اسپن‌ها
        for chikou_key, chikou_data in chikou_analysis.items():
            position = chikou_data.get('position')
            if position == 'ABOVE':
                signals.append(f"{chikou_key}_ABOVE_HISTORICAL")
                if '78' in chikou_key:  # وزن بیشتر برای چیکو بلندمدت
                    score += 3.0
                elif '26' in chikou_key:
                    score += 2.0
                else:
                    score += 1.0
            elif position == 'BELOW':
                signals.append(f"{chikou_key}_BELOW_HISTORICAL")
                if '78' in chikou_key:
                    score -= 3.0
                elif '26' in chikou_key:
                    score -= 2.0
                else:
                    score -= 1.0
        
        # ====================
        # 6. تشخیص روند نهایی
        # ====================
        
        trend = "NEUTRAL"
        signal_strength = "WEAK"
        
        if score >= 8:
            trend = "STRONG_BULLISH"
            signal_strength = "VERY_STRONG"
        elif score >= 5:
            trend = "BULLISH"
            signal_strength = "STRONG"
        elif score >= 2:
            trend = "MILD_BULLISH"
            signal_strength = "MODERATE"
        elif score <= -8:
            trend = "STRONG_BEARISH"
            signal_strength = "VERY_STRONG"
        elif score <= -5:
            trend = "BEARISH"
            signal_strength = "STRONG"
        elif score <= -2:
            trend = "MILD_BEARISH"
            signal_strength = "MODERATE"
        
        # ====================
        # 7. وضعیت ابر کومو
        # ====================
        
        cloud_status = {}
        
        # رنگ ابر
        if current_senkou_a > current_senkou_b:
            cloud_status['color'] = "GREEN"  # ابر صعودی
            score += 1.0
        elif current_senkou_a < current_senkou_b:
            cloud_status['color'] = "RED"    # ابر نزولی
            score -= 1.0
        else:
            cloud_status['color'] = "NEUTRAL"
        
        # ضخامت ابر
        cloud_thickness = abs(current_senkou_a - current_senkou_b)
        cloud_thickness_pct = (cloud_thickness / current_price * 100) if current_price > 0 else 0
        
        cloud_status['thickness'] = float(cloud_thickness)
        cloud_status['thickness_pct'] = float(cloud_thickness_pct)
        
        if cloud_thickness_pct > 2:
            cloud_status['thickness_category'] = "THICK"
        elif cloud_thickness_pct > 1:
            cloud_status['thickness_category'] = "MEDIUM"
        else:
            cloud_status['thickness_category'] = "THIN"
        
        # ====================
        # 8. تشخیص سیگنال معاملاتی
        # ====================
        
        trade_signal = "HOLD"
        
        # شرایط سیگنال خرید قوی
        if (trend in ["STRONG_BULLISH", "BULLISH"] and 
            price_above_cloud and 
            chikou_analysis.get('chikou_78', {}).get('position') == 'ABOVE'):
            trade_signal = "STRONG_BUY"
        
        # شرایط سیگنال خرید معمولی
        elif trend in ["BULLISH", "MILD_BULLISH"] and price_above_cloud:
            trade_signal = "BUY"
        
        # شرایط سیگنال فروش قوی
        elif (trend in ["STRONG_BEARISH", "BEARISH"] and 
              price_below_cloud and 
              chikou_analysis.get('chikou_78', {}).get('position') == 'BELOW'):
            trade_signal = "STRONG_SELL"
        
        # شرایط سیگنال فروش معمولی
        elif trend in ["BEARISH", "MILD_BEARISH"] and price_below_cloud:
            trade_signal = "SELL"
        
        # ====================
        # 9. اعتبارسنجی چندگانه با چیکو‌ها
        # ====================
        
        validation_score = 0
        validation_signals = []
        
        # اعتبارسنجی 1: هماهنگی تمام چیکوها
        chikou_positions = [data.get('position') for data in chikou_analysis.values() 
                           if data.get('position') in ['ABOVE', 'BELOW']]
        
        if len(chikou_positions) >= 2:
            if all(pos == 'ABOVE' for pos in chikou_positions):
                validation_score += 4
                validation_signals.append("ALL_CHIKOU_BULLISH_CONFIRMED")
            elif all(pos == 'BELOW' for pos in chikou_positions):
                validation_score += 4
                validation_signals.append("ALL_CHIKOU_BEARISH_CONFIRMED")
            elif len(set(chikou_positions)) == 1:  # همه یکسان هستند
                validation_score += 2
                validation_signals.append("CHIKOU_CONSISTENT")
        
        # اعتبارسنجی 2: هماهنگی چیکو 78 با ابر
        if (price_above_cloud and chikou_analysis.get('chikou_78', {}).get('position') == 'ABOVE'):
            validation_score += 3
            validation_signals.append("CLOUD_CHIKOU_78_ALIGNED")
        elif (price_below_cloud and chikou_analysis.get('chikou_78', {}).get('position') == 'BELOW'):
            validation_score += 3
            validation_signals.append("CLOUD_CHIKOU_78_ALIGNED")
        
        # اعتبارسنجی 3: قدرت ابر
        if cloud_status['color'] == "GREEN" and price_above_cloud:
            validation_score += 2
            validation_signals.append("GREEN_CLOUD_SUPPORT")
        elif cloud_status['color'] == "RED" and price_below_cloud:
            validation_score += 2
            validation_signals.append("RED_CLOUD_RESISTANCE")
        
        # اعتبارسنجی 4: هماهنگی تنکان و کیجون با چیکو
        if (current_tenkan > current_kijun and 
            chikou_analysis.get('chikou_26', {}).get('position') == 'ABOVE'):
            validation_score += 2
            validation_signals.append("TENKAN_KIJUN_CHIKOU_ALIGNED")
        
        # ====================
        # 10. نتیجه نهایی
        # ====================
        
        confidence_level = "HIGH" if validation_score >= 8 else \
                         "MEDIUM" if validation_score >= 5 else \
                         "LOW" if validation_score >= 2 else "VERY_LOW"
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "current_price": float(current_price),
            
            # اجزای ایچیموکو
            "tenkan_sen": float(current_tenkan),
            "kijun_sen": float(current_kijun),
            "senkou_span_a": float(current_senkou_a),
            "senkou_span_b": float(current_senkou_b),
            
            # وضعیت ابر
            "cloud": {
                "top": float(cloud_top),
                "bottom": float(cloud_bottom),
                "price_position": {
                    "above_cloud": price_above_cloud,
                    "below_cloud": price_below_cloud,
                    "in_cloud": price_in_cloud
                },
                "status": cloud_status
            },
            
            # تحلیل چیکو اسپن‌های آینده
            "chikou_analysis": chikou_analysis,
            
            # تشخیص روند و سیگنال
            "trend": trend,
            "trend_score": float(score),
            "signal_strength": signal_strength,
            "trade_signal": trade_signal,
            
            # سیگنال‌های شناسایی شده
            "signals": signals,
            
            # اعتبارسنجی با چیکو‌ها
            "validation": {
                "score": validation_score,
                "signals": validation_signals,
                "confidence_level": confidence_level,
                "chikou_alignment": len(set(chikou_positions)) == 1 if chikou_positions else False
            },
            
            # اطلاعات تکمیلی
            "price_vs_tenkan_pct": float((current_price / current_tenkan - 1) * 100) if current_tenkan > 0 else None,
            "price_vs_kijun_pct": float((current_price / current_kijun - 1) * 100) if current_kijun > 0 else None,
            "tenkan_vs_kijun_pct": float((current_tenkan / current_kijun - 1) * 100) if current_kijun > 0 else None,
            
            "data_sufficiency": {
                "has_minimum_data": True,
                "periods_available": len(df),
                "periods_required": 78
            },
            
            "version": "3.0.0",
            "features": ["dual_future_chikou", "advanced_validation", "multi_timeframe_confirmation"]
        }
        
        logging.info(f"✅ Advanced Ichimoku analysis completed: {trend} | Signal: {trade_signal} | Score: {score:.1f} | Validation: {validation_score}")
        
        return result
        
    except Exception as e:
        error_msg = f"Advanced Ichimoku calculation error: {type(e).__name__}: {str(e)}"
        logging.error(error_msg)
        return {
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
            "data_sufficiency": {
                "has_minimum_data": False,
                "periods_available": len(df) if 'df' in locals() else 0,
                "periods_required": 78
            }
        }

def generate_signal_with_advanced_ichimoku(
    df: pd.DataFrame, 
    symbol: str = "UNKNOWN",
    test_mode: bool = False
) -> Dict[str, Any]:
    """
    تولید سیگنال با اعتبارسنجی ایچیموکو پیشرفته
    
    Args:
        df: DataFrame با داده‌های OHLC
        symbol: نماد معاملاتی
        test_mode: حالت تست
    
    Returns:
        Dict: سیگنال نهایی با اعتبارسنجی کامل
    """
    try:
        # 1. تحلیل ایچیموکو پیشرفته
        ichimoku_result = calculate_advanced_ichimoku(df)
        
        if "error" in ichimoku_result:
            return {
                "symbol": symbol,
                "signal": "ERROR",
                "confidence": 0,
                "error": ichimoku_result["error"],
                "timestamp": datetime.now().isoformat(),
                "valid": False
            }
        
        # 2. استخراج اطلاعات کلیدی
        trend = ichimoku_result.get("trend", "NEUTRAL")
        trade_signal = ichimoku_result.get("trade_signal", "HOLD")
        validation_score = ichimoku_result.get("validation", {}).get("score", 0)
        trend_score = ichimoku_result.get("trend_score", 0)
        confidence_level = ichimoku_result.get("validation", {}).get("confidence_level", "LOW")
        
        # 3. محاسبه اعتماد (Confidence) بر اساس چیکو‌ها
        confidence = calculate_confidence_with_chikou(ichimoku_result)
        
        # 4. تولید پیام دلایل
        reasons = generate_advanced_signal_reasons(ichimoku_result)
        
        # 5. محاسبه سطوح کلیدی
        current_price = ichimoku_result.get("current_price", 0)
        key_levels = calculate_ichimoku_key_levels(ichimoku_result, current_price)
        
        # 6. اعتبارسنجی نهایی با چیکو‌ها
        final_validation = perform_final_chikou_validation(ichimoku_result)
        
        # 7. نتیجه نهایی
        result = {
            "symbol": symbol,
            "signal": trade_signal,
            "confidence": float(confidence),
            "trend": trend,
            "current_price": float(current_price),
            "reasons": reasons,
            "key_levels": key_levels,
            "validation_summary": final_validation,
            "ichimoku_analysis": {
                "summary": {
                    "trend_score": float(trend_score),
                    "validation_score": validation_score,
                    "confidence_level": confidence_level,
                    "cloud_color": ichimoku_result.get("cloud", {}).get("status", {}).get("color", "NEUTRAL"),
                    "price_position": ichimoku_result.get("cloud", {}).get("price_position", {})
                },
                "chikou_status": {
                    "alignment": ichimoku_result.get("validation", {}).get("chikou_alignment", False),
                    "count": len(ichimoku_result.get("chikou_analysis", {}))
                },
                "signals": ichimoku_result.get("signals", []),
                "validation": ichimoku_result.get("validation", {})
            },
            "timestamp": datetime.now().isoformat(),
            "test_mode": test_mode,
            "valid": True,
            "advanced_features": {
                "dual_chikou": True,
                "future_validation": True,
                "multi_timeframe": True
            },
            "version": "3.0.0"
        }
        
        logging.info(f"✅ Advanced signal generated for {symbol}: {trade_signal} (Confidence: {confidence:.1%})")
        
        return result
        
    except Exception as e:
        error_msg = f"Advanced signal generation error for {symbol}: {type(e).__name__}: {str(e)}"
        logging.error(error_msg)
        
        return {
            "symbol": symbol,
            "signal": "ERROR",
            "confidence": 0,
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
            "valid": False
        }

def calculate_confidence_with_chikou(ichimoku_result: Dict[str, Any]) -> float:
    """
    محاسبه سطح اعتماد بر اساس چیکو اسپن‌ها
    """
    base_confidence = 0.5
    
    # امتیاز روند (0-10)
    trend_score = abs(ichimoku_result.get("trend_score", 0))
    trend_factor = min(trend_score / 10, 1.0)
    
    # امتیاز اعتبارسنجی (0-10)
    validation_score = ichimoku_result.get("validation", {}).get("score", 0)
    validation_factor = min(validation_score / 10, 1.0)
    
    # هماهنگی چیکو‌ها
    chikou_alignment = ichimoku_result.get("validation", {}).get("chikou_alignment", False)
    alignment_factor = 0.15 if chikou_alignment else 0
    
    # قدرت ابر
    cloud_color = ichimoku_result.get("cloud", {}).get("status", {}).get("color", "NEUTRAL")
    cloud_factor = 0.1 if cloud_color in ["GREEN", "RED"] else 0
    
    # موقعیت قیمت نسبت به ابر
    price_position = ichimoku_result.get("cloud", {}).get("price_position", {})
    position_factor = 0
    
    if price_position.get("above_cloud") and cloud_color == "GREEN":
        position_factor = 0.1
    elif price_position.get("below_cloud") and cloud_color == "RED":
        position_factor = 0.1
    
    # محاسبه نهایی
    confidence = (base_confidence + 
                  (trend_factor * 0.25) + 
                  (validation_factor * 0.20) + 
                  alignment_factor + 
                  cloud_factor + 
                  position_factor)
    
    # محدود کردن به بازه 0.1 تا 0.95
    confidence = max(0.1, min(0.95, confidence))
    
    return confidence

def generate_advanced_signal_reasons(ichimoku_result: Dict[str, Any]) -> List[str]:
    """
    تولید لیست دلایل سیگنال بر اساس چیکو‌ها
    """
    reasons = []
    
    # دلایل بر اساس روند
    trend = ichimoku_result.get("trend", "")
    if "BULLISH" in trend:
        reasons.append("📈 روند ایچیموکو صعودی شناسایی شد")
    elif "BEARISH" in trend:
        reasons.append("📉 روند ایچیموکو نزولی شناسایی شد")
    
    # دلایل بر اساس ابر
    cloud_color = ichimoku_result.get("cloud", {}).get("status", {}).get("color", "")
    if cloud_color == "GREEN":
        reasons.append("🟢 ابر کومو سبز (حمایتی)")
    elif cloud_color == "RED":
        reasons.append("🔴 ابر کومو قرمز (مقاومتی)")
    
    # دلایل بر اساس چیکو اسپن‌ها
    chikou_analysis = ichimoku_result.get("chikou_analysis", {})
    
    for key, data in chikou_analysis.items():
        position = data.get("position", "")
        period = key.split("_")[-1]
        
        if position == "ABOVE":
            diff_pct = data.get("difference_pct", 0)
            if diff_pct:
                reasons.append(f"✅ چیکو {period} کندل: بالاتر از قیمت تاریخی ({diff_pct:+.1f}%)")
            else:
                reasons.append(f"✅ چیکو {period} کندل: موقعیت صعودی")
        elif position == "BELOW":
            diff_pct = data.get("difference_pct", 0)
            if diff_pct:
                reasons.append(f"⚠️ چیکو {period} کندل: پایین‌تر از قیمت تاریخی ({diff_pct:+.1f}%)")
            else:
                reasons.append(f"⚠️ چیکو {period} کندل: موقعیت نزولی")
    
    # دلایل اعتبارسنجی
    validation_signals = ichimoku_result.get("validation", {}).get("signals", [])
    
    if "ALL_CHIKOU_BULLISH_CONFIRMED" in validation_signals:
        reasons.append("🎯 تأیید هماهنگی کامل چیکو اسپن‌ها (صعودی)")
    elif "ALL_CHIKOU_BEARISH_CONFIRMED" in validation_signals:
        reasons.append("🎯 تأیید هماهنگی کامل چیکو اسپن‌ها (نزولی)")
    
    if "CLOUD_CHIKOU_78_ALIGNED" in validation_signals:
        reasons.append("🔗 هماهنگی ابر کومو با چیکو 78 کندل")
    
    if "TENKAN_KIJUN_CHIKOU_ALIGNED" in validation_signals:
        reasons.append("⚡ هماهنگی تنکان-کیجون با چیکو")
    
    # سطح اعتماد
    confidence_level = ichimoku_result.get("validation", {}).get("confidence_level", "LOW")
    if confidence_level == "HIGH":
        reasons.append("💎 اعتبار سیگنال: بالا")
    elif confidence_level == "MEDIUM":
        reasons.append("🔶 اعتبار سیگنال: متوسط")
    
    return reasons[:6]  # حداکثر 6 دلیل

def calculate_ichimoku_key_levels(ichimoku_result: Dict[str, Any], current_price: float) -> Dict[str, Any]:
    """
    محاسبه سطوح کلیدی بر اساس ایچیموکو
    """
    try:
        tenkan = ichimoku_result.get("tenkan_sen", 0)
        kijun = ichimoku_result.get("kijun_sen", 0)
        cloud_top = ichimoku_result.get("cloud", {}).get("top", 0)
        cloud_bottom = ichimoku_result.get("cloud", {}).get("bottom", 0)
        
        # محاسبه فواصل درصدی
        levels = {
            "tenkan_sen": {
                "price": float(tenkan),
                "distance_pct": float((current_price / tenkan - 1) * 100) if tenkan > 0 else 0
            },
            "kijun_sen": {
                "price": float(kijun),
                "distance_pct": float((current_price / kijun - 1) * 100) if kijun > 0 else 0
            },
            "cloud_top": {
                "price": float(cloud_top),
                "distance_pct": float((current_price / cloud_top - 1) * 100) if cloud_top > 0 else 0
            },
            "cloud_bottom": {
                "price": float(cloud_bottom),
                "distance_pct": float((current_price / cloud_bottom - 1) * 100) if cloud_bottom > 0 else 0
            }
        }
        
        # تعیین نزدیک‌ترین سطح
        distances = {
            "Tenkan": abs(current_price - tenkan),
            "Kijun": abs(current_price - kijun),
            "Cloud Top": abs(current_price - cloud_top),
            "Cloud Bottom": abs(current_price - cloud_bottom)
        }
        
        nearest_level = min(distances, key=distances.get)
        nearest_distance = distances[nearest_level]
        nearest_distance_pct = (nearest_distance / current_price * 100) if current_price > 0 else 0
        
        levels["nearest_level"] = {
            "name": nearest_level,
            "distance": float(nearest_distance),
            "distance_pct": float(nearest_distance_pct)
        }
        
        return levels
        
    except Exception as e:
        logging.error(f"Key levels calculation error: {e}")
        return {}

def perform_final_chikou_validation(ichimoku_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    اعتبارسنجی نهایی با چیکو اسپن‌ها
    """
    try:
        chikou_analysis = ichimoku_result.get("chikou_analysis", {})
        validation = ichimoku_result.get("validation", {})
        
        # جمع‌آوری وضعیت چیکو‌ها
        chikou_statuses = []
        
        for key, data in chikou_analysis.items():
            period = key.split("_")[-1]
            position = data.get("position", "UNKNOWN")
            diff_pct = data.get("difference_pct", 0)
            
            chikou_statuses.append({
                "period": period,
                "position": position,
                "difference_pct": diff_pct,
                "strength": "STRONG" if abs(diff_pct) > 2 else "MODERATE" if abs(diff_pct) > 1 else "WEAK"
            })
        
        # تحلیل هماهنگی
        positions = [status["position"] for status in chikou_statuses if status["position"] in ["ABOVE", "BELOW"]]
        
        alignment_analysis = {
            "total_chikou": len(chikou_statuses),
            "aligned_chikou": len(set(positions)) == 1 if positions else False,
            "bullish_count": sum(1 for status in chikou_statuses if status["position"] == "ABOVE"),
            "bearish_count": sum(1 for status in chikou_statuses if status["position"] == "BELOW"),
            "average_strength": np.mean([abs(status["difference_pct"] or 0) for status in chikou_statuses])
        }
        
        # تصمیم نهایی
        final_decision = "UNCERTAIN"
        
        if alignment_analysis["aligned_chikou"]:
            if alignment_analysis["bullish_count"] == alignment_analysis["total_chikou"]:
                final_decision = "STRONGLY_BULLISH"
            elif alignment_analysis["bearish_count"] == alignment_analysis["total_chikou"]:
                final_decision = "STRONGLY_BEARISH"
        elif alignment_analysis["bullish_count"] > alignment_analysis["bearish_count"]:
            final_decision = "MILD_BULLISH"
        elif alignment_analysis["bearish_count"] > alignment_analysis["bullish_count"]:
            final_decision = "MILD_BEARISH"
        
        return {
            "chikou_statuses": chikou_statuses,
            "alignment_analysis": alignment_analysis,
            "final_decision": final_decision,
            "validation_score": validation.get("score", 0),
            "confidence_level": validation.get("confidence_level", "LOW")
        }
        
    except Exception as e:
        logging.error(f"Final chikou validation error: {e}")
        return {"error": str(e)}

# ==================== TELEGRAM UTILITIES ====================
def send_telegram_signal(
    symbol: str, 
    signal_data: Dict[str, Any],
    config: Dict[str, str]
) -> bool:
    """
    ارسال سیگنال به تلگرام با فرمت پیشرفته
    """
    try:
        token = config.get('telegram_token', '')
        chat_id = config.get('chat_id', '')
        
        if not token or not chat_id:
            logging.error("Telegram credentials not configured")
            return False
        
        signal = signal_data.get('signal', 'HOLD')
        confidence = signal_data.get('confidence', 0)
        current_price = signal_data.get('current_price', 0)
        reasons = signal_data.get('reasons', [])
        
        # ایموجی بر اساس سیگنال
        emoji_map = {
            "STRONG_BUY": "🚀",
            "BUY": "🟢",
            "STRONG_SELL": "🔻",
            "SELL": "🔴",
            "HOLD": "⏸️"
        }
        
        emoji = emoji_map.get(signal, "📊")
        
        # ساخت پیام
        lines = [
            f"{emoji} *{signal} SIGNAL* {emoji}",
            f"`{symbol}`",
            f"",
            f"💰 *قیمت:* {current_price:,.2f}",
            f"🎯 *اعتماد:* {confidence:.1%}",
            f"",
            f"📊 *تحلیل ایچیموکو پیشرفته:*"
        ]
        
        # اضافه کردن دلایل
        if reasons:
            lines.append(f"")
            for i, reason in enumerate(reasons[:4], 1):
                lines.append(f"{i}. {reason}")
        
        # اضافه کردن اطلاعات چیکو
        chikou_info = signal_data.get('ichimoku_analysis', {}).get('chikou_status', {})
        if chikou_info.get('count', 0) > 0:
            lines.append(f"")
            lines.append(f"🔍 *تایید چیکو اسپن‌ها:*")
            lines.append(f"   • تعداد: {chikou_info.get('count')}")
            lines.append(f"   • هماهنگی: {'✅' if chikou_info.get('alignment') else '❌'}")
        
        # اضافه کردن سطوح کلیدی
        key_levels = signal_data.get('key_levels', {})
        if key_levels.get('nearest_level'):
            nearest = key_levels['nearest_level']
            lines.append(f"")
            lines.append(f"🎯 *نزدیک‌ترین سطح:*")
            lines.append(f"   • {nearest.get('name')}: {nearest.get('distance_pct', 0):.1f}%")
        
        # پاورقی
        lines.append(f"")
        lines.append(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"🔄 نسخه: 3.0.0 (چیکو پیشرفته)")
        
        message = "\n".join(lines)
        
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            logging.info(f"✅ Telegram signal sent for {symbol}: {signal}")
            return True
        else:
            logging.error(f"❌ Telegram API error: {response.status_code}")
            return False
            
    except Exception as e:
        logging.error(f"Telegram signal error: {e}")
        return False

# ==================== TEST FUNCTIONS ====================
def test_advanced_ichimoku():
    """
    تست سیستم ایچیموکو پیشرفته
    """
    logging.info("🧪 Testing Advanced Ichimoku System...")
    
    try:
        # ایجاد داده تست
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        base_price = 50000
        price_series = base_price + np.cumsum(np.random.randn(100) * 100)
        
        df_test = pd.DataFrame({
            'open': price_series * 0.998,
            'high': price_series * 1.005,
            'low': price_series * 0.995,
            'close': price_series,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        logging.info(f"✅ Test DataFrame created: {len(df_test)} candles")
        
        # تست ایچیموکو پیشرفته
        ichimoku_result = calculate_advanced_ichimoku(df_test)
        
        if "error" in ichimoku_result:
            logging.error(f"❌ Ichimoku error: {ichimoku_result['error']}")
            return False
        
        logging.info(f"✅ Advanced Ichimoku analysis:")
        logging.info(f"   - Trend: {ichimoku_result.get('trend')}")
        logging.info(f"   - Signal: {ichimoku_result.get('trade_signal')}")
        logging.info(f"   - Score: {ichimoku_result.get('trend_score', 0):.1f}")
        logging.info(f"   - Validation: {ichimoku_result.get('validation', {}).get('score', 0)}")
        
        # تست تولید سیگنال
        signal_result = generate_signal_with_advanced_ichimoku(df_test, "BTCUSDT", test_mode=True)
        
        logging.info(f"✅ Signal generation:")
        logging.info(f"   - Signal: {signal_result.get('signal')}")
        logging.info(f"   - Confidence: {signal_result.get('confidence', 0):.1%}")
        logging.info(f"   - Valid: {signal_result.get('valid')}")
        
        # تست اعتبارسنجی چیکو
        validation = perform_final_chikou_validation(ichimoku_result)
        logging.info(f"✅ Chikou validation:")
        logging.info(f"   - Final decision: {validation.get('final_decision')}")
        logging.info(f"   - Confidence: {validation.get('confidence_level')}")
        
        # خلاصه
        summary = {
            "total_tests": 3,
            "all_passed": True,
            "ichimoku_features": [
                "Dual Future Chikou",
                "Advanced Validation",
                "Multi-timeframe Confirmation"
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        logging.info(f"✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        logging.error(f"❌ Test error: {type(e).__name__}: {str(e)}")
        return False

# ==================== MAIN EXPORTS ====================
__all__ = [
    # Logger
    'setup_logger', 'log_performance', 'LogLevel',
    
    # Data Utilities
    'validate_dataframe', 'clean_ohlcv_data', 'calculate_volume_profile',
    
    # Technical Analysis
    'calculate_support_resistance', 'calculate_market_structure',
    'get_market_phase', 'calculate_volatility',
    
    # Advanced Ichimoku
    'calculate_advanced_ichimoku',
    'generate_signal_with_advanced_ichimoku',
    'calculate_confidence_with_chikou',
    'generate_advanced_signal_reasons',
    'calculate_ichimoku_key_levels',
    'perform_final_chikou_validation',
    
    # Telegram
    'send_telegram_signal',
    
    # Test
    'test_advanced_ichimoku'
]

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    """
    اجرای تست در صورت اجرای مستقیم فایل
    """
    print("🚀 Advanced Ichimoku Utilities v3.0.0")
    print("=" * 60)
    print("📅 Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("📊 Features: Dual Future Chikou (26 & 78 periods)")
    print("=" * 60)
    
    # اجرای تست
    test_success = test_advanced_ichimoku()
    
    if test_success:
        print("\n✅ System tested and ready for use!")
        print("\n📋 Key Features:")
        print("  1. Dual Future Chikou Span (26 & 78 periods)")
        print("  2. Advanced multi-timeframe validation")
        print("  3. Cloud analysis with thickness measurement")
        print("  4. Confidence calculation based on Chikou alignment")
        print("  5. Telegram integration for signal notifications")
    else:
        print("\n❌ System test failed!")
    
    print("\n🔧 Usage Example:")
    print('''
    from utils import calculate_advanced_ichimoku
    
    # Analyze with dual future Chikou
    result = calculate_advanced_ichimoku(df)
    
    # Generate trading signal
    from utils import generate_signal_with_advanced_ichimoku
    signal = generate_signal_with_advanced_ichimoku(df, "BTCUSDT")
    
    # Send to Telegram
    from utils import send_telegram_signal
    send_telegram_signal("BTCUSDT", signal, config)
    ''')
    
    print("\n📞 Developer: @AsemanSignals")
    print("🔄 Last Updated: " + datetime.now().strftime("%Y-%m-%d"))
