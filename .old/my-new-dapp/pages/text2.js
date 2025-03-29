export function initRegisterAnimation() {
    const registerText = document.getElementById('register-animation');
    if (!registerText) return;

    // 顯示的文字序列
    const textSequences = [
        "請完成註冊",
        "已有帳號？",
        "點選 立即登入"
    ];
    
    let currentSequence = 0;
    let charIndex = 0;
    let isTyping = true;

    // 開始漸變效果
    startGradientAnimation();
    
    // 為打字和刪除設定變化的速度
    const typingSpeed = () => Math.floor(Math.random() * 180) + 120; // 120-300ms
    const deletingSpeed = () => Math.floor(Math.random() * 70) + 30; // 30-100ms
    const pauseBeforeDelete = () => Math.floor(Math.random() * 800) + 700; // 700-1500ms
    const pauseBeforeNextSequence = () => Math.floor(Math.random() * 600) + 300; // 300-900ms
    
    // 動畫循環函數
    function animateText() {
        // 當前顯示的文字
        const currentText = textSequences[currentSequence];
        
        if (isTyping) {
            // 正在打字
            if (charIndex < currentText.length) {
                // 添加字符
                registerText.textContent += currentText.charAt(charIndex);
                charIndex++;
                
                // 某些字符可能需要特殊停頓（如標點符號）
                const nextDelay = currentText.charAt(charIndex - 1).match(/[，。？！、]/g) 
                    ? typingSpeed() + 100 
                    : typingSpeed();
                    
                setTimeout(animateText, nextDelay);
            } else {
                // 打字完成，等待片刻後開始刪除
                setTimeout(() => {
                    isTyping = false;
                    animateText();
                }, pauseBeforeDelete());
            }
        } else {
            // 正在刪除
            if (charIndex > 0) {
                charIndex--;
                registerText.textContent = registerText.textContent.slice(0, -1);
                setTimeout(animateText, deletingSpeed());
            } else {
                // 刪除完成，進入下一個序列
                currentSequence = (currentSequence + 1) % textSequences.length;
                isTyping = true;
                setTimeout(animateText, pauseBeforeNextSequence());
            }
        }
    }

    // 漸變效果函數
    function startGradientAnimation() {
        let hue1 = 160; // 開始色相 (綠色)
        let hue2 = 220; // 結束色相 (藍色)
        
        // 定期更新顏色，但速度略微不規則
        let interval = 50;
        
        const updateGradient = () => {
            hue1 = (hue1 + (Math.random() * 0.5 + 0.75)) % 360; // 0.75-1.25度的變化
            hue2 = (hue2 + (Math.random() * 0.5 + 0.75)) % 360;
            
            registerText.style.backgroundImage = `linear-gradient(to right, 
                hsl(${hue1}, 70%, 50%), 
                hsl(${hue2}, 70%, 50%))`;
                
            // 稍微變化更新間隔
            interval = Math.floor(Math.random() * 20) + 40; // 40-60ms
            setTimeout(updateGradient, interval);
        };
        
        updateGradient();
    }

    // 清空文字並開始動畫
    registerText.textContent = '';
    registerText.classList.add('typing-complete');
    setTimeout(animateText, pauseBeforeNextSequence());
}