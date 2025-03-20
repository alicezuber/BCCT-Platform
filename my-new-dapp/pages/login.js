// pages/login.js
import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import Link from 'next/link';
import styles from './styles/Login.module.css';
import { initWelcomeAnimation } from './text';

export default function Login() {
  const [formData, setFormData] = useState({
    username: '',
    password: ''
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  useEffect(() => {
    // Initialize the welcome text animation
    initWelcomeAnimation();
  }, []);

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
      
      // 保存 token 到 localStorage
      localStorage.setItem('token', data.token);
      
      // 登入成功，跳轉到首頁或儀表板
      router.push('/home');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.pageContainer}>
      <div className={styles.loginForm}>
        <div className={styles.container}>
          <h1 className={styles.title}>登入</h1>
          
          {error && (
            <div className={styles.error}>{error}</div>
          )}
          
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
            <Link href="/register" className={styles.link}>註冊</Link>
          </div>
        </div>
      </div>
      
      <div className={styles.welcomeSection}>
        <h2 id="welcome-text" className={styles.welcomeText}></h2>
      </div>
    </div>
  );
}