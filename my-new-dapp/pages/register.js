// pages/register.js
import { useState } from 'react';
import { useRouter } from 'next/router';
import Link from 'next/link';
import styles from './styles/Register.module.css';

export default function Register() {
  const [formData, setFormData] = useState({
    username: '',
    password: '',
    name: '',
    email: ''
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
      const response = await fetch('/api/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username: formData.username,
          password: formData.password,
          name: formData.name,
          email: formData.email
        })
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.message || '註冊失敗');
      }
      
      // 註冊成功，跳轉到登入頁面
      router.push('/login');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.container}>
      <h1 className={styles.title}>註冊帳號</h1>
      
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
        
        <div className={styles.formGroup}>
          <label className={styles.label}>姓名</label>
          <input
            type="text"
            name="name"
            value={formData.name}
            onChange={handleChange}
            className={styles.input}
          />
        </div>
        
        <div className={styles.formGroup}>
          <label className={styles.label}>電子郵件</label>
          <input
            type="email"
            name="email"
            value={formData.email}
            onChange={handleChange}
            className={styles.input}
          />
        </div>
        
        <button
          type="submit"
          disabled={loading}
          className={styles.button}
        >
          {loading ? '處理中...' : '註冊'}
        </button>
      </form>
      
      <div className={styles.extra}>
        已有帳號？{' '}
        <Link href="/login" className={styles.link}>登入</Link>
      </div>
      <div className={styles.extra}>
        忘記密碼？{' '}
        <Link href="/forget_password" className={styles.link}>忘記密碼</Link>
      </div>
    </div>
  );
}
