import { useState } from 'react';

export default function Home() {
  const [dbStatus, setDbStatus] = useState(null);
  const [userStatus, setUserStatus] = useState(null);
  const [loading, setLoading] = useState(false);

  // 測試數據庫連接
  const testDatabase = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/test-db');
      const data = await res.json();
      setDbStatus(data);
    } catch (error) {
      setDbStatus({ error: error.message });
    }
    setLoading(false);
  };

  // 創建測試用戶
  const createTestUser = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/login_creataccount', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({})
      });
      const data = await res.json();
      setUserStatus(data);
    } catch (error) {
      setUserStatus({ error: error.message });
    }
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: '800px', margin: '0 auto', padding: '20px' }}>
      <h1>MongoDB 登入系統測試頁面</h1>
      
      <div style={{ marginBottom: '30px' }}>
        <h2>步驟 1: 測試數據庫連接</h2>
        <button 
          onClick={testDatabase} 
          disabled={loading}
          style={{
            padding: '10px 15px',
            backgroundColor: '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: loading ? 'not-allowed' : 'pointer'
          }}
        >
          測試數據庫連接
        </button>
        
        {dbStatus && (
          <div style={{ marginTop: '15px', padding: '15px', border: '1px solid #ddd', borderRadius: '4px' }}>
            <h3>結果:</h3>
            <pre style={{ backgroundColor: '#f5f5f5', padding: '10px', overflow: 'auto' }}>
              {JSON.stringify(dbStatus, null, 2)}
            </pre>
          </div>
        )}
      </div>
      
      <div style={{ marginBottom: '30px' }}>
        <h2>步驟 2: 創建測試用戶</h2>
        <p>創建一個測試用戶 (username: test, password: 123456)</p>
        <button 
          onClick={createTestUser} 
          disabled={loading}
          style={{
            padding: '10px 15px',
            backgroundColor: '#2196F3',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: loading ? 'not-allowed' : 'pointer'
          }}
        >
          創建測試用戶
        </button>
        
        {userStatus && (
          <div style={{ marginTop: '15px', padding: '15px', border: '1px solid #ddd', borderRadius: '4px' }}>
            <h3>結果:</h3>
            <pre style={{ backgroundColor: '#f5f5f5', padding: '10px', overflow: 'auto' }}>
              {JSON.stringify(userStatus, null, 2)}
            </pre>
          </div>
        )}
      </div>
      
      <div>
        <h2>接下來是什麼?</h2>
        <p>成功創建測試用戶後，您可以開始實現:</p>
        <ul>
          <li>登入頁面 (/pages/login.js)</li>
          <li>註冊頁面 (/pages/register.js)</li>
          <li>登入API (/pages/api/login.js)</li>
          <li>受保護的頁面 (/pages/dashboard.js)</li>
        </ul>
      </div>
    </div>
  );
}