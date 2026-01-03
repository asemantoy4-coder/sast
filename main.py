// مرحله ۲: تحلیل ۵ ارز برتر با پایتون (API واقعی)
async function processTopFiveAI() {
    updateScanStatus("مرحله ۲: تحلیل ۵ ارز برتر با پایتون", 50);
    
    // مرتب‌سازی بر اساس بیشترین امتیاز
    const top5 = [...cryptoScanResults]
        .sort((a, b) => b.preliminaryScore - a.preliminaryScore)
        .slice(0, 5);

    selectedTopSymbols = top5.map(s => s.symbol);
    
    // علامت‌گذاری کاندیدهای AI
    selectedTopSymbols.forEach(symbol => {
        const index = cryptoScanResults.findIndex(c => c.symbol === symbol);
        if (index !== -1) {
            cryptoScanResults[index].isTopCandidate = true;
        }
    });
    
    document.getElementById('aiCount').textContent = selectedTopSymbols.length;
    
    // نمایش در باکس AI
    const aiContent = document.getElementById('aiContent');
    aiContent.innerHTML = `
        <div style="color: var(--success); margin-bottom: 10px; font-weight: bold;">✅ ۵ کاندیدای نهایی شناسایی شدند:</div>
        <div style="display: flex; flex-wrap: wrap; gap:5px; margin-bottom: 10px;">
            ${selectedTopSymbols.map(s => `<span class="target-badge" style="background: rgba(240, 185, 11, 0.2); color: var(--accent);">${s}</span>`).join('')}
        </div>
        <div style="margin-top: 10px; font-size: 12px; color: #aaa;">در حال تحلیل عمیق با AI (پایتون)...</div>
    `;
    
    // تحلیل هر یک از ۵ ارز برتر با AI
    for (let i = 0; i < top5.length; i++) {
        if (!isCryptoScanning) break;
        
        const symbolData = top5[i];
        
        try {
            // تحلیل AI
            const response = await fetch(`${PYTHON_API_URL}?symbol=${symbolData.symbol}`);
            
            if (!response.ok) {
                throw new Error(`HTTP Error: ${response.status}`);
            }

            const pythonAnalysis = await response.json();
            
            // چک کردن اینکه آیا پاسخ موفقیت‌آمیز است
            if (pythonAnalysis.status !== 'success') {
                throw new Error(pythonAnalysis.message || "Unknown Python Error");
            }

            topAISignals.push({
                symbol: symbolData.symbol,
                price: pythonAnalysis.price,
                signal: pythonAnalysis.signal,
                confidence: pythonAnalysis.confidence, // این باید عدد باشد (0.7 یا 0.8)
                reasons: pythonAnalysis.reasons,
                riskReward: `Quality: ${pythonAnalysis.quality_score}`, // استفاده از quality_score
                analyzedByAI: true,
                aiTimestamp: new Date().getTime()
            });
            
            // به‌روزرسانی پیشرفت
            const progress = 50 + Math.round(((i + 1) / top5.length) * 50);
            updateScanStatus(`تحلیل AI: ${i + 1}/${top5.length}`, progress);
            
            // نمایش نتایج AI
            displayAISignals();
            
            await sleep(1000);
            
        } catch (error) {
            console.error(`خطا در تحلیل پایتون برای ${symbolData.symbol}:`, error);
            // اگر خطا از سرور بود، می‌توان اینجا تصمیم گرفت که از تحلیل داخلی استفاده کند
            // برای سادگی فعلا خطا را لوگ می‌کنیم
            showNotification(`خطا در تحلیل ${symbolData.symbol}: ${error.message}`, "error");
        }
    }
    
    // ارسال لیدر برای تحلیل دستی
    if (selectedTopSymbols.length > 0) {
        document.getElementById('symbolInput').value = selectedTopSymbols[0];
        showNotification(`ارز برتر: ${selectedTopSymbols[0]} برای تحلیل انتخاب شد`, "info");
        
        setTimeout(() => {
            requestAnalysis();
        },1000);
    }
}
