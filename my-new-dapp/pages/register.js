// pages/register.js
import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import Link from 'next/link';
import styles from './styles/Register.module.css';
import { initRegisterAnimation } from './text2';

export default function Register() {
  const [formData, setFormData] = useState({ username: '', password: '', name: '', email: '' });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  // 初始化動畫
  useEffect(() => {
    initRegisterAnimation();
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const response = await fetch('/api/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || '註冊失敗');
      }

      router.push('/login');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.pageContainer}>
      <Link href="/home" className={styles.homeButton}>回首頁</Link>
      <div className={styles.formContainer}>
        <div className={styles.container}>
          <h1>註冊</h1>
          {error && <p className={styles.error}>{error}</p>}
            <form onSubmit={handleSubmit} className={styles.form}>
              <div className={styles.formGroup}>
                <label className={styles.label}>用戶名</label>
                <input
                  name="username"
                  placeholder="用戶名"
                  onChange={(e) => setFormData({ ...formData, username: e.target.value })}
                  required
                  className={styles.input}
                />
              </div>
              <div className={styles.formGroup}>
                <label className={styles.label}>密碼</label>
                <input
                  name="password"
                  type="password"
                  placeholder="密碼"
                  onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                  required
                  className={styles.input}
                />
              </div>
              <div className={styles.formGroup}>
                <label className={styles.label}>姓名</label>
                <input
                  name="name"
                  placeholder="姓名"
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  required
                  className={styles.input}
                />
              </div>
              <div className={styles.formGroup}>
                <label className={styles.label}>電子郵件</label>
                <input
                  name="email"
                  type="email"
                  placeholder="電子郵件"
                  onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                  required
                  className={styles.input}
                />
              </div>
              <button type="submit" disabled={loading} className={styles.button}>
                {loading ? '註冊中...' : '註冊'}
              </button>
            </form>

            <div className={styles.extra}>
            已有帳號？{' '}
              <Link href="/login" className={styles.link}>立即登入</Link>
            </div>
          </div>
        </div>
      {/* 右側動畫區域 */}
      <div className={styles.animationSection}>
        <h2 id="register-animation" className={styles.animationText}></h2>
      </div>
    </div>
  );
}
