// 在 text2.js 中添加 initRegisterAnimation
export function initRegisterAnimation() {
    const registerText = document.getElementById('register-animation');
    if (!registerText) return;

    // 顯示的文字
    const text = "請完成註冊";
    let charIndex = 0;

    // 打字效果函數
    function typeText() {
        if (charIndex < text.length) {
            registerText.textContent += text.charAt(charIndex);
            charIndex++;
            setTimeout(typeText, 200); // 控制打字速度
        } else {
            // 打字完成後，移除光標並開始漸變效果
            setTimeout(() => {
                registerText.classList.add('typing-complete');
                startGradientAnimation();
            }, 500);
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
    setTimeout(typeText, 500); // 延遲後啟動打字效果
}
