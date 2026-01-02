import os
from flask import Flask, request, jsonify
import exchange_handler
import utils
import traceback

app = Flask(__name__)
# پورت را از متغیر محیط Render می‌گیرد، اگر نبود پیش‌فرض 5000 است
port = int(os.environ.get("PORT", 5000))

# ==================== HELPER FUNCTIONS ====================
# این توابع منطق‌های هوشمند قبلی را دارند اما بدون نیاز به کلاس اجرا می‌شوند

def check_multi_timeframe_alignment(symbol):
    """
    بررسی هماهنگی سیگنال در تایم‌فریم‌های مختلف (15m, 1h)
    """
    timeframes = ['15m', '1h']
    aligned_count = 0
    
    for tf in timeframes:
        try:
            # دریافت دیتا برای تایم‌فریم بالاتر
            df_tf = exchange_handler.DataHandler.fetch_data(symbol, tf, limit=50)
            if not df_tf.empty and len(df_tf) > 20:
                analysis_tf = utils.generate_scalp_signals(df_tf)
                score_tf = analysis_tf.get('score', 0)
                if abs(score_tf) >= 2.0: 
                    aligned_count += 1
        except Exception as e:
            continue
    
    # اگر حداقل 1 تایم‌فریم بالاتر همسو بود
    return aligned_count >= 1

def calculate_signal_quality_score(analysis, symbol):
    """
    محاسبه نهایی کیفیت سیگنال
    """
    score_weights = {
        'base_score': 0.4,
        'volume_confirmation': 0.2,
        'multi_timeframe_alignment': 0.15,
        'risk_reward_ratio': 0.15,
        'market_context': 0.1
    }
    
    quality_score = analysis.get('score', 0) * score_weights['base_score']
    
    inner = analysis.get('analysis', {})
    volume_profile = inner.get('volume_profile', {})
    
    # 1. تأیید حجم
    if volume_profile.get('in_value_area', False):
        quality_score += 2 * score_weights['volume_confirmation']
    
    # 2. هماهنگی چند تایم‌فریم
    # فقط یک بار برای ارزیابی کیفیت بررسی می‌کنیم
    if check_multi_timeframe_alignment(symbol):
        quality_score += 1.5 * score_weights['multi_timeframe_alignment']
    
    # 3. نسبت ریسک به ریوارد
    current_price = analysis.get('price', 0)
    if current_price > 0:
        market_regime = inner.get('market_regime', {})
        atr = market_regime.get('atr_percent', 1.0)
        
        if 0.3 <= atr <= 1.5: 
            quality_score += 2 * score_weights['risk_reward_ratio']
        elif atr > 2.0:
            quality_score -= 1
    
    # 4. شرایط بازار
    market_regime = inner.get('market_regime', {})
    if market_regime.get('scalp_safe', False):
        quality_score += 1 * score_weights['market_context']
    
    return min(10, max(0, quality_score))


# ==================== FLASK ROUTES ====================

@app.route('/', methods=['GET'])
def health_check():
    """برای نگه‌داشتن سرویس بیدار (Ping)"""
    return jsonify({
        "status": "online",
        "service": "Aseman Calculation Engine",
        "version": "API_v1.0"
    })

@app.route('/analyze', methods=['GET'])
def analyze_coin():
    """
    اندپوینت اصلی.
    ورودی: symbol (مثال BTC/USDT)
    خروجی: JSON حاوی امتیاز، سیگنال و تحلیل‌ها
    """
    try:
        symbol = request.args.get('symbol')
        
        if not symbol:
            return jsonify({
                "status": "error",
                "message": "Missing parameter 'symbol'. Example: ?symbol=BTC/USDT"
            }), 400
        
        print(f"📩 [API] Analyzing {symbol}...")
        
        # 1. دریافت داده اصلی (5 دقیقه)
        df = exchange_handler.DataHandler.fetch_data(symbol, '5m', limit=100)
        
        if df.empty or len(df) < 20:
            return jsonify({
                "status": "error",
                "message": "Insufficient data"
            }), 404
        
        # 2. تحلیل اصلی
        analysis = utils.generate_scalp_signals(df)
        
        # 3. محاسبه کیفیت نهایی با تابع کمکی
        quality_score = calculate_signal_quality_score(analysis, symbol)
        
        # 4. ارسال پاسخ به جاوا
        return jsonify({
            "status": "success",
            "symbol": symbol,
            "price": analysis.get('price'),
            "score": analysis.get('score'),
            "signal": analysis.get('signal'),
            "quality_score": float(quality_score), # امتیاز کیفیتی محاسبه شده
            "confidence": analysis.get('confidence'),
            "reasons": analysis.get('reasons'),
            "analysis": analysis.get('analysis')
        })
        
    except Exception as e:
        print(f"❌ [API] Error: {e}")
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    print(f"🚀 Starting API Server on port {port}...")
    app.run(host='0.0.0.0', port=port)
