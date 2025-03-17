// pages/api/register.js
import connectToDatabase from '../../lib/db';
import User from '../../models/User';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: '只允許 POST 請求' });
  }

  try {
    await connectToDatabase();
    
    const { username, password, name, email } = req.body;
    
    // 檢查用戶名是否已存在
    const existingUser = await User.findOne({ username });
    if (existingUser) {
      return res.status(400).json({ message: '用戶名已被使用' });
    }
    
    // 創建新用戶
    const user = new User({
      username,
      password, // 會在保存時自動加密
      name,
      email,
    });
    
    await user.save();
    
    // 不返回敏感資訊
    const { password: _, ...userWithoutSensitiveInfo } = user.toObject();
    
    res.status(201).json({ 
      message: '註冊成功', 
      user: userWithoutSensitiveInfo 
    });
  } catch (error) {
    console.error('註冊錯誤:', error);
    res.status(500).json({ message: '伺服器錯誤', error: error.message });
  }
}
