// pages/register.js
import { useState } from 'react';
import { useRouter } from 'next/router';

export default function RegisterPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [email, setEmail] = useState('');
  const [name, setName] = useState('');
  const [message, setMessage] = useState('');
  const router = useRouter();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage('');
    
    // 基本驗證
    if (password !== confirmPassword) {
      setMessage('密碼不匹配');
      return;
    }
    
    try {
      const res = await fetch('./api/login_creataccount', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password, email, name }),
      });
      const data = await res.json();
      if (res.ok) {
        setMessage(data.message);
        // 註冊成功後，可以跳轉到登入頁面
        setTimeout(() => router.push('/login'), 2000);
      } else {
        setMessage(data.message || '註冊失敗');
      }
    } catch (error) {
      setMessage('錯誤: ' + error.message);
    }
  };

  return (
    <div style={{ margin: 'auto', maxWidth: '400px', padding: '1rem' }}>
      <h1>註冊</h1>
      <form onSubmit={handleSubmit}>
        {/* 表單欄位... */}
        <button type="submit">註冊</button>
      </form>
      {message && <p>{message}</p>}
    </div>
  );
}