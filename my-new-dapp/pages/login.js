// pages/login.js
import { useState } from 'react';
import { useRouter } from 'next/router';
import Link from 'next/link';
import styles from './styles/Login.module.css';
import { useAuth } from '../context/AuthContext';
import { initWelcomeAnimation } from './text.js';

export default function Login() {
  const { login } = useAuth();
  const [formData, setFormData] = useState({
    username: '',
    password: ''
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    

    try {
      const response = await fetch('/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username: formData.username,
          password: formData.password
        })
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.message || '登入失敗');
      }
      

      // 呼叫 Context API 的 login 方法
      login(data.user.username, data.token);
      
      // 登入成功，跳轉到首頁
      router.push('/home');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.pageContainer}>
      <Link href="/home" className={styles.homeButton}>
        回首頁
      </Link>
      <div className={styles.loginForm}>
        <h1>登入</h1>
        {error && <p className={styles.error}>{error}</p>}
        <form onSubmit={handleSubmit}>
          <div className={styles.formGroup}>
            <label className={styles.label}>用戶名</label>
            <input
              type="text"
              name="username"
              value={formData.username}
              onChange={handleChange}
              className={styles.input}
              required
              placeholder="用戶名"
            />
          </div>
          
          <div className={styles.formGroup}>
            <label className={styles.label}>密碼</label>
            <input
              type="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              className={styles.input}
              required
              placeholder="密碼"
            />
          </div>
          
          <button
            type="submit"
            disabled={loading}
            className={styles.button}
          >
            {loading ? '登入中...' : '登入'}
          </button>
        </form>
        <div className={styles.extra}>
          還沒有帳號？{' '}
          <Link href="/register" className={styles.link}>立即註冊</Link>
        </div>
      </div>
      

    </div>
  );
}