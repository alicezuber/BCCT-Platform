import { useState, useEffect } from 'react';
import styles from '../styles/Home.module.css';

export default function ApiDemo() {
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [num1, setNum1] = useState(5);
  const [num2, setNum2] = useState(3);

  // 使用API代理呼叫Python服務
  const callPythonApi = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`/api/proxy?path=add?num1=${num1}&num2=${num2}`);
      const data = await response.json();
      
      if (!response.ok) throw new Error(data.message || '請求失敗');
      
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.container}>
      <h1 className={styles.title}>API 代理演示</h1>
      
      <div className={styles.card}>
        <h2>加法API示例</h2>
        <div style={{ margin: '1rem 0' }}>
          <input 
            type="number" 
            value={num1} 
            onChange={e => setNum1(parseInt(e.target.value))} 
            style={{ width: '4rem', marginRight: '0.5rem' }}
          />
          {' + '}
          <input 
            type="number" 
            value={num2} 
            onChange={e => setNum2(parseInt(e.target.value))} 
            style={{ width: '4rem', marginLeft: '0.5rem' }}
          />
          
          <button 
            onClick={callPythonApi} 
            disabled={loading}
            style={{ marginLeft: '1rem' }}
          >
            {loading ? '計算中...' : '計算'}
          </button>
        </div>
        
        {error && (
          <div style={{ color: 'red', margin: '1rem 0' }}>
            錯誤: {error}
          </div>
        )}
        
        {result && (
          <div style={{ margin: '1rem 0' }}>
            <h3>結果: {result.result}</h3>
            <pre>{JSON.stringify(result, null, 2)}</pre>
          </div>
        )}
      </div>
    </div>
  );
}