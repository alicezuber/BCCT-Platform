// pages/register.js
import { useState } from 'react';
import { useRouter } from 'next/router';
import Link from 'next/link';
import styles from './styles/Register.module.css';
import { useAuth } from '../context/AuthContext';

export default function Register() {
  const { login } = useAuth();
  const [formData, setFormData] = useState({ username: '', password: '', name: '', email: '' });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const router = useRouter();

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
      <h1>註冊</h1>
      {error && <p>{error}</p>}
      <form onSubmit={handleSubmit}>
        <input name="username" placeholder="用戶名" onChange={(e) => setFormData({ ...formData, username: e.target.value })} />
        <input name="password" type="password" placeholder="密碼" onChange={(e) => setFormData({ ...formData, password: e.target.value })} />
        <input name="name" placeholder="姓名" onChange={(e) => setFormData({ ...formData, name: e.target.value })} />
        <input name="email" placeholder="電子郵件" onChange={(e) => setFormData({ ...formData, email: e.target.value })} />
        <button type="submit" disabled={loading}>
          {loading ? '註冊中...' : '註冊'}
        </button>
      </form>
      <Link href="/login">已有帳號？立即登入</Link>
    </div>
  );
}
