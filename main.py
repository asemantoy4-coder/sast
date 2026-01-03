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
        <div style="color: var(--success); margin-bottom: 10px; font-weight: bold;">✅ ${selectedTopSymbols.length} کاندیدای نهایی شناسایی شدند:</div>
        <div style="display: flex; flex-wrap: wrap; gap:5px; margin-bottom: 10px;">
            ${selectedTopSymbols.map(s => `<span class="target-badge" style="background: rgba(240, 185, 11, 0.2); color: var(--accent);">${s}</span>`).join('')}
        </div>
        <div style="margin-top: 10px; font-size: 12px; color: #aaa;">در حال تحلیل عمیق با AI (پایتون)...</div>
    `;
    
    // تنظیمات برای تحلیل AI
    const pythonApiUrl = "http://localhost:10000/analyze";
    
    // تحلیل هر یک از ۵ ارز برتر با AI
    for (let i = 0; i < top5.length; i++) {
        if (!isCryptoScanning) {
            showNotification("تحلیل توسط کاربر متوقف شد", "warning");
            break;
        }
        
        const symbolData = top5[i];
        const symbol = symbolData.symbol;
        
        try {
            // نمایش وضعیت در حال تحلیل
            updateScanStatus(`تحلیل AI: ${symbol} (${i + 1}/${top5.length})`, 50 + Math.round((i / top5.length) * 40));
            
            // تحلیل AI با timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 ثانیه timeout
            
            const response = await fetch(`${pythonApiUrl}?symbol=${symbol}`, {
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                throw new Error(`خطای HTTP: ${response.status} ${response.statusText}`);
            }

            const pythonAnalysis = await response.json();
            
            // چک کردن وضعیت پاسخ
            if (pythonAnalysis.status !== 'success') {
                throw new Error(pythonAnalysis.message || "خطای ناشناخته از سرور پایتون");
            }
            
            // اعتبارسنجی فیلدهای مورد نیاز
            if (!pythonAnalysis.price || !pythonAnalysis.signal || 
                pythonAnalysis.confidence === undefined || !pythonAnalysis.reasons || 
                pythonAnalysis.quality_score === undefined) {
                console.warn(`پاسخ ناقص برای ${symbol}:`, pythonAnalysis);
                // استفاده از مقادیر پیش‌فرض در صورت ناقص بودن پاسخ
                pythonAnalysis.price = pythonAnalysis.price || 0;
                pythonAnalysis.signal = pythonAnalysis.signal || "HOLD";
                pythonAnalysis.confidence = pythonAnalysis.confidence || 0.5;
                pythonAnalysis.reasons = pythonAnalysis.reasons || ["اطلاعات ناکافی"];
                pythonAnalysis.quality_score = pythonAnalysis.quality_score || 5;
            }
            
            // ذخیره نتایج AI
            topAISignals.push({
                symbol: symbol,
                price: pythonAnalysis.price,
                signal: pythonAnalysis.signal,
                confidence: parseFloat(pythonAnalysis.confidence) || 0,
                reasons: Array.isArray(pythonAnalysis.reasons) ? pythonAnalysis.reasons : [pythonAnalysis.reasons],
                riskReward: `کیفیت: ${pythonAnalysis.quality_score}/10`,
                qualityScore: parseFloat(pythonAnalysis.quality_score) || 0,
                analyzedByAI: true,
                aiTimestamp: new Date().getTime(),
                rawAnalysis: pythonAnalysis // ذخیره کامل پاسخ برای دیباگ
            });
            
            // به‌روزرسانی پیشرفت
            const progress = 50 + Math.round(((i + 1) / top5.length) * 50);
            updateScanStatus(`تحلیل AI تکمیل: ${i + 1}/${top5.length}`, progress);
            
            // نمایش لحظه‌ای نتایج AI
            displayAISignals();
            
            // تأخیر بین درخواست‌ها برای جلوگیری از rate limiting
            await sleep(1500);
            
        } catch (error) {
            console.error(`خطا در تحلیل پایتون برای ${symbol}:`, error);
            
            // ذخیره خطا به عنوان یک نتیجه با وضعیت خطا
            topAISignals.push({
                symbol: symbol,
                price: 0,
                signal: "ERROR",
                confidence: 0,
                reasons: [`خطای تحلیل: ${error.message}`],
                riskReward: "خطا در تحلیل",
                qualityScore: 0,
                analyzedByAI: false,
                aiTimestamp: new Date().getTime(),
                error: true,
                errorMessage: error.message
            });
            
            // نمایش خطا
            showNotification(`خطا در تحلیل ${symbol}`, "error");
            
            // همچنان نتایج را نمایش دهید
            displayAISignals();
        }
    }
    
    // بررسی نتایج AI
    const successfulAnalysis = topAISignals.filter(s => !s.error && s.analyzedByAI);
    
    if (successfulAnalysis.length > 0) {
        // مرتب‌سازی بر اساس کیفیت و اعتماد
        const sortedSignals = [...successfulAnalysis].sort((a, b) => {
            const scoreA = (a.confidence * 0.7) + (a.qualityScore * 0.3);
            const scoreB = (b.confidence * 0.7) + (b.qualityScore * 0.3);
            return scoreB - scoreA;
        });
        
        // انتخاب بهترین سیگنال برای تحلیل دستی
        const bestSignal = sortedSignals[0];
        
        if (bestSignal && bestSignal.signal !== "ERROR") {
            document.getElementById('symbolInput').value = bestSignal.symbol;
            showNotification(
                `ارز برتر: ${bestSignal.symbol} (سیگنال: ${bestSignal.signal}, اعتماد: ${(bestSignal.confidence * 100).toFixed(1)}%)`, 
                "success"
            );
            
            // تأخیر قبل از درخواست تحلیل
            setTimeout(() => {
                if (isCryptoScanning) {
                    requestAnalysis();
                }
            }, 2000);
        } else {
            showNotification("هیچ سیگنال معتبری یافت نشد", "warning");
        }
    } else {
        showNotification("همه تحلیل‌های AI با خطا مواجه شدند", "error");
    }
    
    // به‌روزرسانی وضعیت نهایی
    updateScanStatus("تحلیل AI تکمیل شد", 100);
}

function displayAISignals() {
    const aiResultsDiv = document.getElementById('aiResults');
    if (!aiResultsDiv) return;
    
    // فیلتر کردن فقط سیگنال‌های موفق
    const validSignals = topAISignals.filter(s => s.analyzedByAI && !s.error);
    
    if (validSignals.length === 0) {
        aiResultsDiv.innerHTML = `
            <div style="padding: 20px; text-align: center; color: var(--text-muted);">
                <i class="fas fa-exclamation-triangle"></i>
                <p>هیچ نتیجه‌ای از تحلیل AI یافت نشد</p>
            </div>
        `;
        return;
    }
    
    let html = `
        <div style="margin-bottom: 15px; color: var(--text-color);">
            <h4 style="margin-bottom: 10px; color: var(--accent);">
                <i class="fas fa-robot"></i> نتایج تحلیل AI
            </h4>
            <div style="font-size: 12px; color: var(--text-muted); margin-bottom: 15px;">
                ${validSignals.length} ارز تحلیل شدند
            </div>
    `;
    
    validSignals.forEach(signal => {
        const confidencePercent = (signal.confidence * 100).toFixed(1);
        const signalColor = signal.signal === "BUY" ? "var(--success)" : 
                          signal.signal === "SELL" ? "var(--danger)" : "var(--warning)";
        
        html += `
            <div class="ai-signal-card" style="
                background: var(--card-bg);
                border: 1px solid var(--border-color);
                border-radius: 8px;
                padding: 12px;
                margin-bottom: 10px;
                ${signal.signal === "BUY" ? 'border-left: 4px solid var(--success);' : ''}
                ${signal.signal === "SELL" ? 'border-left: 4px solid var(--danger);' : ''}
            ">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div style="font-weight: bold; font-size: 14px;">
                        ${signal.symbol}
                        <span style="color: ${signalColor}; margin-left: 10px;">
                            ${signal.signal}
                        </span>
                    </div>
                    <div style="font-size: 12px; color: var(--text-muted);">
                        قیمت: $${signal.price.toLocaleString()}
                    </div>
                </div>
                
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <div style="font-size: 12px;">
                        اعتماد: <span style="color: var(--accent);">${confidencePercent}%</span>
                    </div>
                    <div style="font-size: 12px;">
                        کیفیت: <span style="color: var(--success);">${signal.qualityScore.toFixed(1)}/10</span>
                    </div>
                </div>
                
                <div style="font-size: 11px; color: var(--text-muted);">
                    دلایل: ${signal.reasons.slice(0, 3).join('، ')}
                    ${signal.reasons.length > 3 ? ` (+${signal.reasons.length - 3} مورد دیگر)` : ''}
                </div>
            </div>
        `;
    });
    
    html += `</div>`;
    aiResultsDiv.innerHTML = html;
}
