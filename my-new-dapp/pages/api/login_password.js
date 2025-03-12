// pages/api/login.js
export default function handler(req, res) {
    if (req.method === 'POST') {
      const { username, password } = req.body;
  
      // 這裡用硬編碼的測試用戶，實際上請用資料庫驗證，並注意密碼安全存儲
      if (username === 'test' && password === '123') {
        // 可在此生成 JWT 或設置 session（此處僅回傳成功訊息）
        res.status(200).json({
          success: true,
          message: 'Login successful!',
          user: { username },
        });
      } else {
        res.status(401).json({ success: false, message: 'Invalid credentials' });
      }
    } else {
      res.setHeader('Allow', ['POST']);
      res.status(405).end(`Method ${req.method} Not Allowed`);
    }
  }
  