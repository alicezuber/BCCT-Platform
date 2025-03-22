// pages/home.js
import Head from 'next/head';
import Link from 'next/link';
import React from 'react';
import { useRouter } from 'next/router';
import styles from './styles/Home.module.css';
import { useAuth } from '../context/AuthContext';

export default function Home() {
  const { account, logout } = useAuth();
  const router = useRouter();

  const handleLogin = () => {
    router.push('/login');
  };

  return (
    <div>
      <Head>
        <title>My Next.js App - Home</title>
        <meta name="description" content="這是我的 Next.js 應用首頁" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

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
          <Link href="/personal_pages" className={styles.navLink}>
            個人頁面
          </Link>
        </nav>
        <div className={styles.rightHeader}>
          {account ? (
            <>
              <span className={styles.userInfo}>已登入：{account}</span>
              <button onClick={logout} className={styles.authButton}>
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

      <main className={styles.main}>
        <h1>歡迎來到大廳</h1>
        <p>這裡是應用的大廳，未登入也能觀看資訊。</p>
      </main>
    </div>
  );
}
