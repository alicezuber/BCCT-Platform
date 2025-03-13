// lib/db.js
import { MongoClient } from 'mongodb';
// 或者 import mongoose from 'mongoose';

// 使用環境變量存儲連接字符串，不要硬編碼
const MONGODB_URI = process.env.MONGODB_URI;

if (!MONGODB_URI) {
  throw new Error('Please define the MONGODB_URI environment variable');
}

let cachedClient = null;
let cachedDb = null;

export async function connectToDatabase() {
  // 如果已連接，使用現有連接
  if (cachedClient && cachedDb) {
    return { client: cachedClient, db: cachedDb };
  }

  const client = await MongoClient.connect(MONGODB_URI);
  const db = client.db('your-database-name');
  
  cachedClient = client;
  cachedDb = db;

  return { client, db };
}