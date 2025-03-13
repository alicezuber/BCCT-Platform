import { connectToDatabase } from '../../lib/db';

export default async function handler(req, res) {
  try {
    const { client, db } = await connectToDatabase();
    // 嘗試訪問users集合（如果不存在會自動創建）
    const collection = db.collection('users');
    const count = await collection.countDocuments();
    
    res.status(200).json({ 
      message: '數據庫連接成功!', 
      usersCount: count,
      dbName: db.databaseName
    });
  } catch (error) {
    console.error('數據庫連接錯誤:', error);
    res.status(500).json({ 
      message: '數據庫連接失敗', 
      error: error.message 
    });
  }
}