import { connectToDatabase } from '../../lib/db';
import bcrypt from 'bcryptjs';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', ['POST']);
    return res.status(405).end(`Method ${req.method} Not Allowed`);
  }

  try {
    const { db } = await connectToDatabase();
    const collection = db.collection('users');
    
    // 檢查測試用戶是否已存在
    const existingUser = await collection.findOne({ username: 'test' });
    
    if (existingUser) {
      return res.status(200).json({ 
        message: '測試用戶已存在', 
        user: { 
          username: existingUser.username,
          email: existingUser.email
        }
      });
    }
    
    // 創建測試用戶
    const hashedPassword = await bcrypt.hash('123456', 12);
    const result = await collection.insertOne({
      username: 'test',
      password: hashedPassword,
      email: 'test@example.com',
      createdAt: new Date()
    });
    
    res.status(201).json({ 
      message: '測試用戶創建成功!',
      userId: result.insertedId
    });
  } catch (error) {
    console.error('創建測試用戶失敗:', error);
    res.status(500).json({ 
      message: '創建測試用戶失敗', 
      error: error.message 
    });
  }
}