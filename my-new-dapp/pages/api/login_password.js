// pages/api/login.js
import { connectToDatabase } from '../../lib/db';
import bcrypt from 'bcryptjs';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', ['POST']);
    return res.status(405).end(`Method ${req.method} Not Allowed`);
  }

  const { username, password } = req.body;

  try {
    // 連接到數據庫
    const { db } = await connectToDatabase();
    
    // 查找用戶
    const user = await db.collection('users').findOne({ username });
    
    if (!user) {
      return res.status(401).json({ success: false, message: '用戶名或密碼錯誤' });
    }
    
    // 驗證密碼
    const isPasswordValid = await bcrypt.compare(password, user.password);
    
    if (!isPasswordValid) {
      return res.status(401).json({ success: false, message: '用戶名或密碼錯誤' });
    }
    
    // 不返回密碼等敏感信息
    const { password: _, ...userWithoutPassword } = user;
    
    // 登入成功
    return res.status(200).json({
      success: true,
      message: '登入成功!',
      user: userWithoutPassword,
    });
  } catch (error) {
    console.error('登入錯誤:', error);
    return res.status(500).json({ success: false, message: '伺服器錯誤' });
  }
}