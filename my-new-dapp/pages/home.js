// pages/home.js
import Head from 'next/head';
import Link from 'next/link';
import React, { useState } from 'react';
import { useRouter } from 'next/router';  // <-- 新增這行
import styles from './styles/Home.module.css';

export default function Home() {
  // 模擬登入狀態，若 account 為 null 則表示未登入
  const [account, setAccount] = useState(null);
  const router = useRouter(); // <-- 取得 router 實例

  // 改成跳轉到 /login
  const handleLogin = () => {
    router.push('/login');
  };

  const handleLogout = () => {
    setAccount(null);
  };

  return (
    <div>
      <Head>
        <title>My Next.js App - Home</title>
        <meta name="description" content="這是我的 Next.js 應用首頁" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      {/* Header 區塊 */}
      <header className={styles.header}>
        {/* 左側連結 */}
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
        </nav>
        {/* 右側登入狀態 */}
        <div className={styles.loginStatus}>
          {account ? (
            <>
              <span>已登入：{account}</span>
              <button onClick={handleLogout} className={styles.authButton}>
                登出
              </button>
            </>
          ) : (
            <>
              <span>未登入</span>
              {/* 按下後跳轉到 /login */}
              <button onClick={handleLogin} className={styles.authButton}>
                登入
              </button>
            </>
          )}
        </div>
      </header>

      {/* Main 區域 */}
      <main className={styles.main}>
        <h1>歡迎來到我的應用</h1>
        <p>這裡是主頁內容，你可以在這裡放入更多資訊。</p>
      </main>
    </div>
  );
}
