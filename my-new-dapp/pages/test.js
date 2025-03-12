import { useState } from 'react';

export default function Home() {
  const [a, setA] = useState('');
  const [b, setB] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setResult(null);
    try {
      const res = await fetch(`http://localhost:5000/api/add?a=${a}&b=${b}`);
      const data = await res.json();
      if (data.success) {
        setResult(data.result);
      } else {
        setError(data.error || '發生錯誤');
      }
    } catch (err) {
      setError('呼叫 API 時發生錯誤');
    }
  };

  return (
    <div style={{ padding: '2rem', fontFamily: 'Arial, sans-serif' }}>
      <h1>加法 API 測試</h1>
      <form onSubmit={handleSubmit} style={{ marginBottom: '1rem' }}>
        <input
          type="number"
          placeholder="數字1"
          value={a}
          onChange={(e) => setA(e.target.value)}
          required
          style={{ marginRight: '0.5rem', padding: '0.5rem' }}
        />
        <input
          type="number"
          placeholder="數字2"
          value={b}
          onChange={(e) => setB(e.target.value)}
          required
          style={{ marginRight: '0.5rem', padding: '0.5rem' }}
        />
        <button type="submit" style={{ padding: '0.5rem 1rem' }}>計算</button>
      </form>
      {result !== null && <p>結果: {result}</p>}
      {error && <p style={{ color: 'red' }}>{error}</p>}
    </div>
  );
}
