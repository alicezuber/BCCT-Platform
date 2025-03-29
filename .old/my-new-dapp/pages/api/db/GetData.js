// pages/api/db/GetData.js
import connectToDatabase from '../../../lib/db';
import User from '../../../models/User';

export default async function handler(req, res) {
  if (req.method !== 'GET') {
    return res.status(405).json({ message: '只允許 GET 請求' });
  }
  
  try {
    await connectToDatabase();
    
    const { username } = req.query;  // 前端會用 ?username=xxx 帶上
    if (!username) {
      return res.status(400).json({ message: '缺少 username' });
    }
    
    // 去資料庫查找
    const user = await User.findOne({ username });
    if (!user) {
      return res.status(404).json({ message: '找不到此用戶' });
    }
    
    // 避免把 password、private_key 等敏感資訊直接回傳
    const { password, blockchain_private_key, ...rest } = user.toObject();
    
    return res.status(200).json({ user: rest });
  } catch (error) {
    console.error('取得個人資料時出錯:', error);
    return res.status(500).json({ message: '伺服器錯誤' });
  }
}
