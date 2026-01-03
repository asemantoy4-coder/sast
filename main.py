#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بررسی ۵ تحلیل آخر جاوااسکریپت
"""

import json
import os
import sqlite3
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# دیتابیس ساده برای ذخیره تحلیل‌ها
def init_db():
    conn = sqlite3.connect('analyses.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS analyses
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  symbol TEXT,
                  signal TEXT,
                  confidence REAL,
                  price REAL,
                  reasons TEXT,
                  timestamp DATETIME)''')
    conn.commit()
    conn.close()

def save_analysis(symbol, signal, confidence, price, reasons):
    conn = sqlite3.connect('analyses.db')
    c = conn.cursor()
    c.execute('''INSERT INTO analyses 
                 (symbol, signal, confidence, price, reasons, timestamp)
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (symbol, signal, confidence, price, json.dumps(reasons), datetime.now()))
    conn.commit()
    conn.close()

def get_last_5_analyses():
    conn = sqlite3.connect('analyses.db')
    c = conn.cursor()
    c.execute('''SELECT symbol, signal, confidence, price, reasons, timestamp 
                 FROM analyses 
                 ORDER BY timestamp DESC 
                 LIMIT 5''')
    rows = c.fetchall()
    conn.close()
    
    analyses = []
    for row in rows:
        analyses.append({
            'symbol': row[0],
            'signal': row[1],
            'confidence': row[2],
            'price': row[3],
            'reasons': json.loads(row[4]) if row[4] else [],
            'timestamp': row[5]
        })
    return analyses

def analyze_crypto_technical(symbol):
    """تحلیل تکنیکال یک ارز"""
    try:
        # اینجا می‌توانید از API‌های واقعی استفاده کنید
        # برای مثال از yfinance یا binance
        
        # داده نمونه
        import random
        signals = ['BUY', 'SELL', 'HOLD']
        signal = random.choice(signals)
        
        # محاسبات نمونه
        price = random.uniform(40000, 50000)
        confidence = random.uniform(0.6, 0.95)
        
        # دلایل نمونه
        reasons_list = [
            "RSI در منطقه اشباع خرید",
            "واگرایی مثبت در MACD",
            "شکست مقاومت کلیدی",
            "حجم معاملات بالا",
            "میانگین متحرک صعودی"
        ]
        reasons = random.sample(reasons_list, random.randint(2, 4))
        
        return {
            'symbol': symbol,
            'signal': signal,
            'confidence': round(confidence, 2),
            'price': round(price, 2),
            'reasons': reasons,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'symbol': symbol,
            'signal': 'ERROR',
            'confidence': 0,
            'price': 0,
            'reasons': [f"خطا در تحلیل: {str(e)}"],
            'timestamp': datetime.now().isoformat()
        }

@app.route('/')
def home():
    return jsonify({
        'status': 'active',
        'service': 'Crypto Analysis API',
        'endpoints': {
            '/analyze?symbol=BTCUSDT': 'تحلیل یک ارز',
            '/last-5': 'نمایش ۵ تحلیل آخر',
            '/health': 'بررسی سلامت سرویس'
        }
    })

@app.route('/analyze', methods=['GET'])
def analyze():
    """تحلیل یک ارز مشخص"""
    symbol = request.args.get('symbol', 'BTCUSDT').upper()
    
    # تحلیل تکنیکال
    analysis = analyze_crypto_technical(symbol)
    
    # ذخیره در دیتابیس
    save_analysis(
        symbol=analysis['symbol'],
        signal=analysis['signal'],
        confidence=analysis['confidence'],
        price=analysis['price'],
        reasons=analysis['reasons']
    )
    
    return jsonify({
        'status': 'success',
        'analysis': analysis,
        'message': f'تحلیل {symbol} با موفقیت انجام شد'
    })

@app.route('/last-5', methods=['GET'])
def last_5_analyses():
    """نمایش ۵ تحلیل آخر"""
    analyses = get_last_5_analyses()
    
    return jsonify({
        'status': 'success',
        'count': len(analyses),
        'analyses': analyses,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/analyze-top-5', methods=['GET'])
def analyze_top_5():
    """تحلیل ۵ ارز برتر"""
    top_symbols = request.args.get('symbols', 'BTCUSDT,ETHUSDT,BNBUSDT,XRPUSDT,ADAUSDT')
    symbols = [s.strip() for s in top_symbols.split(',')][:5]
    
    results = []
    for symbol in symbols:
        analysis = analyze_crypto_technical(symbol)
        results.append(analysis)
        
        # ذخیره هر تحلیل
        save_analysis(
            symbol=analysis['symbol'],
            signal=analysis['signal'],
            confidence=analysis['confidence'],
            price=analysis['price'],
            reasons=analysis['reasons']
        )
    
    return jsonify({
        'status': 'success',
        'analyzed_count': len(results),
        'results': results,
        'top_recommendation': max(results, key=lambda x: x['confidence'])
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Crypto Analysis API',
        'version': '1.0.0'
    })

if __name__ == '__main__':
    # مقداردهی اولیه دیتابیس
    init_db()
    
    # نمایش اطلاعات شروع
    print("=" * 50)
    print("🔄 شروع سرویس تحلیل ارزهای دیجیتال")
    print("📊 API Endpoints:")
    print("   - GET /analyze?symbol=BTCUSDT")
    print("   - GET /last-5")
    print("   - GET /analyze-top-5?symbols=BTC,ETH,BNB")
    print("   - GET /health")
    print("=" * 50)
    
    # اجرای سرور
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
