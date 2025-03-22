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
                setTimeout(animateText, 200);
            } else {
                // 打字完成，等待片刻後開始刪除
                setTimeout(() => {
                    isTyping = false;
                    animateText();
                }, 1000);
            }
        } else {
            // 正在刪除
            if (charIndex > 0) {
                charIndex--;
                registerText.textContent = registerText.textContent.slice(0, -1);
                setTimeout(animateText, 100);
            } else {
                // 刪除完成，進入下一個序列
                currentSequence = (currentSequence + 1) % textSequences.length;
                isTyping = true;
                setTimeout(animateText, 500);
            }
        }
    }

    // 漸變效果函數
    function startGradientAnimation() {
        let hue1 = 160; // 開始色相 (綠色)
        let hue2 = 220; // 結束色相 (藍色)
        
        // 定期更新顏色
        setInterval(() => {
            hue1 = (hue1 + 1) % 360;
            hue2 = (hue2 + 1) % 360;
            
            registerText.style.backgroundImage = `linear-gradient(to right, 
                hsl(${hue1}, 70%, 50%), 
                hsl(${hue2}, 70%, 50%))`;
        }, 50);
    }

    // 清空文字並開始動畫
    registerText.textContent = '';
    registerText.classList.add('typing-complete');
    setTimeout(animateText, 500);
}