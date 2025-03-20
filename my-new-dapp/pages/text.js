// text.js
/**
 * 歡迎文字動畫效果
 * 包含打字效果和顏色漸變效果
 */

export function initWelcomeAnimation() {
    const welcomeText = document.getElementById('welcome-text');
    if (!welcomeText) return;
  
    // 要顯示的文字
    const text = "歡迎登入";
    let charIndex = 0;
  
    // 打字效果函數
    function typeText() {
      if (charIndex < text.length) {
        welcomeText.textContent += text.charAt(charIndex);
        charIndex++;
        setTimeout(typeText, 200);
      } else {
        // 打字完成後，移除光標並開始漸變效果
        setTimeout(() => {
          welcomeText.classList.add('typing-complete');
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
        
        welcomeText.style.backgroundImage = `linear-gradient(to right, 
          hsl(${hue1}, 70%, 50%), 
          hsl(${hue2}, 70%, 50%))`;
      }, 50);
    }
  
    // 清空文字並開始動畫
    welcomeText.textContent = '';
    setTimeout(typeText, 500);
}

/**
 * 重置歡迎文字動畫
 * 用於需要重新播放動畫的場景
 */
export function resetWelcomeAnimation() {
    const welcomeText = document.getElementById('welcome-text');
    if (!welcomeText) return;
    
    welcomeText.textContent = '';
    welcomeText.classList.remove('typing-complete');
    welcomeText.style.backgroundImage = 'linear-gradient(to right, #24b47e, #3861fb)';
    
    // 重新初始化動畫
    initWelcomeAnimation();
}
