// models/User.js
import mongoose from 'mongoose';
import bcrypt from 'bcryptjs';
import crypto from 'crypto';

const UserSchema = new mongoose.Schema({
  username: {
    type: String,
    required: true,
    unique: true,
  },
  password: {
    type: String,
    required: true,
  },
  name: String,
  email: String,
  blockchain_address: String,
  blockchain_private_key: {
    type: String,
    select: false,  // 默認查詢不返回此欄位
  },
  createdAt: {
    type: Date,
    default: Date.now,
  },
});

// 保存前加密密碼
UserSchema.pre('save', async function(next) {
  // 如果密碼被修改，則加密
  if (this.isModified('password')) {
    this.password = await bcrypt.hash(this.password, 12);
  }
  
  // 如果區塊鏈私鑰被修改，則加密
  if (this.isModified('blockchain_private_key') && this.blockchain_private_key) {
    // 使用環境變量中的加密密鑰
    const ENCRYPTION_KEY = process.env.ENCRYPTION_KEY;
    if (!ENCRYPTION_KEY) {
      throw new Error('缺少加密密鑰，請設置 ENCRYPTION_KEY 環境變量');
    }
    
    // 創建初始化向量
    const iv = crypto.randomBytes(16);
    
    // 創建加密器
    const cipher = crypto.createCipheriv(
      'aes-256-cbc',
      Buffer.from(ENCRYPTION_KEY),
      iv
    );
    
    // 加密私鑰
    let encrypted = cipher.update(this.blockchain_private_key, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    
    // 存儲加密的私鑰和初始化向量
    this.blockchain_private_key = `${iv.toString('hex')}:${encrypted}`;
  }
  
  next();
});

// 驗證密碼方法
UserSchema.methods.comparePassword = async function(candidatePassword) {
  return await bcrypt.compare(candidatePassword, this.password);
};

// 解密區塊鏈私鑰的方法
UserSchema.methods.getDecryptedPrivateKey = function() {
  if (!this.blockchain_private_key) {
    return null;
  }
  
  const ENCRYPTION_KEY = process.env.ENCRYPTION_KEY;
  if (!ENCRYPTION_KEY) {
    throw new Error('缺少加密密鑰，請設置 ENCRYPTION_KEY 環境變量');
  }
  
  // 分割初始化向量和加密數據
  const [ivHex, encryptedHex] = this.blockchain_private_key.split(':');
  
  // 創建解密器
  const decipher = crypto.createDecipheriv(
    'aes-256-cbc',
    Buffer.from(ENCRYPTION_KEY),
    Buffer.from(ivHex, 'hex')
  );
  
  // 解密
  let decrypted = decipher.update(encryptedHex, 'hex', 'utf8');
  decrypted += decipher.final('utf8');
  
  return decrypted;
};

export default mongoose.models.User || mongoose.model('User', UserSchema);