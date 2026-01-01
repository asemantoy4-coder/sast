import pandas as pd
import numpy as np
from typing import List, Dict, Any

def calculate_volume_profile(df: pd.DataFrame, bins: int = 50):
    """محاسبه POC و سطوح نقدینگی"""
    price_min = df['Low'].min()
    price_max = df['High'].max()
    bins_edges = np.linspace(price_min, price_max, bins + 1)
    
    # محاسبه حجم در هر سطح قیمتی
    vprofile = df.groupby(pd.cut(df['Close'], bins=bins_edges))['Volume'].sum()
    poc_index = vprofile.argmax()
    poc_level = (bins_edges[poc_index] + bins_edges[poc_index+1]) / 2
    
    return {"poc": poc_level, "profile": vprofile}

def get_ichimoku(df: pd.DataFrame) -> Dict[str, float]:
    """محاسبه اجزای ایچیموکو برای تایید اسکلپ"""
    high_9 = df['High'].rolling(window=9).max()
    low_9 = df['Low'].rolling(window=9).min()
    tenkan_sen = (high_9 + low_9) / 2
    
    high_26 = df['High'].rolling(window=26).max()
    low_26 = df['Low'].rolling(window=26).min()
    kijun_sen = (high_26 + low_26) / 2
    
    return {
        "tenkan": tenkan_sen.iloc[-1],
        "kijun": kijun_sen.iloc[-1],
        "current_price": df['Close'].iloc[-1]
    }