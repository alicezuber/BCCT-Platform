// pages/home.js
import Head from 'next/head';
import Link from 'next/link';
import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import styles from './styles/Home.module.css';
// 直接導入 jwt-decode
import { jwtDecode } from 'jwt-decode'; 

export default function Home() {
  const [account, setAccount] = useState(null);
  const router = useRouter();

  useEffect(() => {
    // 檢查是否在瀏覽器環境中
    if (typeof window !== 'undefined') {
      const token = localStorage.getItem('token');
      
      if (token) {
        try {
          // 使用導入的 jwtDecode
          const decoded = jwtDecode(token);
         
          // 確認解析出的令牌中有 username
          if (decoded && decoded.username) {
            setAccount(decoded.username);
          } else {
            console.error('Token does not contain username');
          }
        } catch (error) {
          console.error('Failed to decode token:', error);
          // 當令牌解析失敗時，清除無效令牌
          localStorage.removeItem('token');
        }
      }
    }
    // dsdsd
  }, []);

  const handleLogin = () => {
    router.push('/login');
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    setAccount(null);
  };

  return (
    <div>
      <Head>
        <title>My Next.js App - Home</title>
        <meta name="description" content="這是我的 Next.js 應用首頁" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      {/* Header 區塊，左側是導覽連結，右側顯示登入狀態及按鈕 */}
      <header className={styles.header}>
        <nav className={styles.nav}>
          <Link href="/AddCarbonToken" className={styles.navLink}>
            新增碳權
          </Link>
          <Link href="/page2" className={styles.navLink}>
            Page 2
          </Link>
          <Link href="/page3" className={styles.navLink}>
            Page 3
          </Link>
          <Link href="/profile" className={styles.navLink}>
            個人頁面
          </Link>
        </nav>
        <div className={styles.rightHeader}>
          {account ? (
            <>
              <span className={styles.userInfo}>已登入：{account}</span>
              <button onClick={handleLogout} className={styles.authButton}>
                登出
              </button>
            </>
          ) : (
            <>
              <span className={styles.userInfo}>未登入</span>
              <button onClick={handleLogin} className={styles.authButton}>
                登入
              </button>
            </>
          )}
        </div>
      </header>

      {/* 主內容區 */}
      <main className={styles.main}>
        <h1>歡迎來到大廳</h1>
        <p>這裡是應用的大廳，未登入也能觀看資訊。</p>
      </main>
    </div>
  );
}