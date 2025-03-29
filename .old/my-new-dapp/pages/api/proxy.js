import fetch from 'isomorphic-unfetch';

/**
 * API代理服務
 * 作為前端和Python後端之間的中間層
 * 支援GET和POST請求方法
 */
export default async function handler(req, res) {
  // 從環境變數獲取Python服務器地址
  const pythonServerIP = process.env.ShiLaCow_ip || 'localhost:5000';
  
  try {
    // 從請求中獲取目標路徑和方法
    const { path } = req.query;
    const targetUrl = `http://${pythonServerIP}/${path || ''}`;
    
    // 準備請求選項
    const options = {
      method: req.method,
      headers: {
        'Content-Type': 'application/json',
      },
    };

    // 如果是POST/PUT請求，添加請求體
    if (req.method === 'POST' || req.method === 'PUT') {
      options.body = JSON.stringify(req.body);
    }

    // 發送請求到Python伺服器
    console.log(`[API代理] 轉發${req.method}請求到：${targetUrl}`);
    const response = await fetch(targetUrl, options);
    
    // 獲取響應資料
    const data = await response.json();
    
    // 返回響應
    res.status(response.status).json(data);
  } catch (error) {
    console.error('[API代理] 錯誤:', error);
    res.status(500).json({ 
      error: '代理請求失敗', 
      message: error.message,
      serverIp: pythonServerIP
    });
  }
}