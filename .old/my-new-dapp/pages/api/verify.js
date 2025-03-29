// pages/api/verify.js
import jwt from 'jsonwebtoken';

export default function handler(req, res) {
  // 從 header 中取得 Authorization 欄位
  const authHeader = req.headers.authorization;
  if (!authHeader) {
    return res.status(401).json({ message: '未提供 token' });
  }

  // 從 "Bearer <token>" 中提取 token
  const token = authHeader.split(' ')[1];
  if (!token) {
    return res.status(401).json({ message: 'token 格式錯誤' });
  }

  try {
    // 驗證 token，使用與簽發 token 時相同的秘密金鑰
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    // 驗證成功，回傳解析後的 token 內容
    return res.status(200).json({ message: 'token 有效', decoded });
  } catch (error) {
    console.error('token 驗證失敗:', error);
    return res.status(401).json({ message: 'token 無效或已過期', error: error.message });
  }
}
