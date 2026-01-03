async function requestAnalysis() {
    const symbol = document.getElementById('symbolInput').value || 'BTCUSDT';
    
    console.log(`درخواست تحلیل برای: ${symbol}`);
    
    try {
        const response = await fetch(`${PYTHON_API_URL}/analyze?symbol=${symbol}`);
        
        if (!response.ok) {
            throw new Error(`خطا: ${response.status}`);
        }
        
        const data = await response.json();
        console.log("داده دریافتی:", data);
        
        // نمایش نتیجه
        const resultDiv = document.getElementById('aiContent');
        resultDiv.innerHTML = `
            <div style="padding: 20px; background: #1a1a2e; border-radius: 10px; border-left: 5px solid #00d4ff;">
                <h3 style="color: #00d4ff; margin: 0 0 10px 0;">
                    ${data.analysis.symbol}: ${data.analysis.signal}
                </h3>
                <div style="color: #fff;">
                    قیمت: <span style="color: #00ff88;">$${data.analysis.price}</span>
                </div>
                <div style="color: #aaa; margin-top: 10px;">
                    ${data.analysis.reasons.join(' • ')}
                </div>
            </div>
        `;
        
    } catch (error) {
        console.error("خطا:", error);
        document.getElementById('aiContent').innerHTML = `
            <div style="color: #ff4444; padding: 20px; text-align: center;">
                خطا در اتصال: ${error.message}
            </div>
        `;
    }
}
