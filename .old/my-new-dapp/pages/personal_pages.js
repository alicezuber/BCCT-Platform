// pages/PersonalPage.js
import React, { useState, useEffect } from 'react';
import Web3 from 'web3';
import accountsArtifact from '../build/contracts/Accounts.json';
import { useAuth } from '../context/AuthContext';
import styles from './styles/PersonalPage.module.css';

const PersonalPage = () => {
  // 1. 從 AuthContext 取得當前登入的 username
  const { account } = useAuth();  
  //    注意：這裡的 account 是個字串 "alice" 或 "bob" 之類

  // 2. 用於儲存從資料庫撈回來的「完整使用者資料」
  const [userData, setUserData] = useState(null);

  // 3. 錯誤 / 提示訊息
  const [error, setError] = useState('');

  // 4. 用於儲存從合約取得的區塊鏈資料（例如餘額）
  const [blockchainData, setBlockchainData] = useState(null);

  // 只要有 username，就去撈完整的 user 資料
  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const res = await fetch(`/api/db/GetData?username=${account}`);
        if (!res.ok) {
          throw new Error(`API 錯誤: ${res.status}`);
        }
        const data = await res.json();
        setUserData(data.user);  
        // data.user 裏面就包含 name、email、createdAt、blockchain_address 等欄位
      } catch (err) {
        setError(err.message || '撈取用戶資料失敗');
      }
    };
    
    if (account) {
      fetchUserData();
    }
  }, [account]);

  // 当 userData 裏拿到 blockchain_address 後，再去呼叫合約拿餘額
  useEffect(() => {
    if (userData && userData.blockchain_address) {
      initBlockchain();
    }
  }, [userData]);

  const initBlockchain = async () => {
    try {
      const web3 = new Web3('http://25.3.19.183:7545');
      const networkId = await web3.eth.net.getId();
      const deployedNetwork = accountsArtifact.networks[networkId];
      if (!deployedNetwork) {
        setError(`找不到合約的部署資訊 (networkId: ${networkId})`);
        return;
      }
      const contractAddress = deployedNetwork.address;
      const accountsContract = new web3.eth.Contract(
        accountsArtifact.abi,
        contractAddress
      );

      const balance = await accountsContract.methods
        .getBalance(userData.blockchain_address)
        .call();
      setBlockchainData({ balance });
    } catch (err) {
      console.error('區塊鏈初始化或讀取合約時發生錯誤:', err);
      setError('無法連線到區塊鏈或讀取合約資料');
    }
  };

  // 如果尚未登入，就顯示提示
  if (!account) {
    return <div>請先登入以檢視個人資訊。</div>;
  }
  
  // 如果尚未載入 userData，可以先顯示 Loading
  if (!userData) {
    return <div>載入中...</div>;
  }

  return (
    <div className={styles['page-container']}>
      <header className={styles.header}>
        <h1 className={styles.title}>個人頁面</h1>
      </header>
      <main className={styles['main-content']}>
        {error && <p style={{ color: 'red' }}>{error}</p>}

        <div className={styles['user-info']}>
          <h2>帳號資訊</h2>
          <ul>
            <li>使用者名稱：{userData.username}</li>
            <li>姓名：{userData.name}</li>
            <li>電子郵件：{userData.email}</li>
            <li>創建日期：{new Date(userData.createdAt).toLocaleDateString()}</li>
          </ul>
        </div>

        <div className={styles['blockchain-info']}>
          <h2>區塊鏈帳號</h2>
          {userData.blockchain_address ? (
            <div>
              <p>地址：{userData.blockchain_address}</p>
              <p>
                餘額：
                {blockchainData
                  ? blockchainData.balance
                  : '載入中...'}
              </p>
            </div>
          ) : (
            <div>沒有區塊鏈帳號</div>
          )}
        </div>
      </main>
    </div>
  );
};

export default PersonalPage;
