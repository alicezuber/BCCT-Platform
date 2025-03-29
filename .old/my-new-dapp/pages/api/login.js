// pages/api/login.js
import connectToDatabase from '../../lib/db';
import User from '../../models/User';
import jwt from 'jsonwebtoken';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: '只允許 POST 請求' });
  }

  try {
    await connectToDatabase();
    
    const { username, password } = req.body;
    
    // 查找用戶
    const user = await User.findOne({ username });
    if (!user) {
      return res.status(401).json({ message: '用戶名或密碼錯誤' });
    }
    
    // 驗證密碼
    const isValid = await user.comparePassword(password);
    if (!isValid) {
      return res.status(401).json({ message: '用戶名或密碼錯誤' });
    }
    
    // 創建 JWT Token
    const token = jwt.sign(
      { userId: user._id, username: user.username },
      process.env.JWT_SECRET,
      { expiresIn: '7d' }
    );
    
    // 不返回密碼和私鑰
    const { password: _, blockchain_private_key: __, ...userWithoutSensitiveInfo } = user.toObject();
    
    res.status(200).json({ 
      message: '登入成功', 
      user: userWithoutSensitiveInfo,
      token 
    });
  } catch (error) {
    console.error('登入錯誤:', error);
    res.status(500).json({ message: '伺服器錯誤', error: error.message });
  }
}