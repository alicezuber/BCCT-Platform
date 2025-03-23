// pages/PersonalPage.js
import React, { useState, useEffect } from 'react';
import Web3 from 'web3';
import accountsArtifact from '../build/contracts/Accounts.json';

// 3. 從全域 AuthContext 中取得登入資訊（含 username, email, blockchain_address 等）
import { useAuth } from '../context/AuthContext';
// 4. 載入樣式檔
import styles from './styles/PersonalPage.module.css';

const PersonalPage = () => {
  // 從 AuthContext 取得當前登入的用戶資料
  const { account } = useAuth();

  // 用於儲存從合約取得的區塊鏈資料（例如餘額）
  const [blockchainData, setBlockchainData] = useState(null);

  // 用於儲存錯誤或提示資訊
  const [error, setError] = useState('');

  useEffect(() => {
    // 只在已登入且有 blockchain_address 時，才嘗試連線區塊鏈
    if (account && account.blockchain_address) {
      initBlockchain();
    }
  }, [account]);

  // 初始化 web3 與合約，並呼叫合約函數
  const initBlockchain = async () => {
    try {
      // -----------------------------------------------------------------------
      // A. 建立 web3 連線
      //    1) 若你要直接連線到 Ganache（不透過 MetaMask），可以指定 HTTP Provider：
      //       const web3 = new Web3('http://127.0.0.1:7545');
      //    2) 若你要透過使用者的瀏覽器錢包 (MetaMask)，則可使用：
      //       const web3 = new Web3(window.ethereum);
      // -----------------------------------------------------------------------
      const web3 = new Web3('http://25.3.19.183:7545');

      // -----------------------------------------------------------------------
      // B. 取得當前網路 ID，Truffle 編譯後的 JSON 裡面會針對每個網路有不同配置
      //    你可以在 Ganache 的「Network ID」中查到 ID，預設常是 5777
      // -----------------------------------------------------------------------
      const networkId = await web3.eth.net.getId();
      const deployedNetwork = accountsArtifact.networks[networkId];

      // 如果在該 networkId 下找不到部署資訊，就表示合約尚未部署在此網路
      if (!deployedNetwork) {
        setError(`找不到合約的部署資訊 (networkId: ${networkId})`);
        return;
      }

      // -----------------------------------------------------------------------
      // C. 建立合約實例，需傳入合約的 ABI 以及對應網路上的地址
      // -----------------------------------------------------------------------
      const contractAddress = deployedNetwork.address;
      const accountsContract = new web3.eth.Contract(
        accountsArtifact.abi,
        contractAddress
      );

      // -----------------------------------------------------------------------
      // D. 呼叫合約函數
      //    假設合約中有個 getBalance(address) 函數可以查詢餘額
      //    呼叫時要用 .methods.<functionName>(...).call() 的方式
      // -----------------------------------------------------------------------
      const balance = await accountsContract.methods
        .getBalance(account.blockchain_address)
        .call();

      // 將取得的餘額存入 state
      setBlockchainData({ balance });
    } catch (err) {
      console.error('區塊鏈初始化或讀取合約時發生錯誤:', err);
      setError('無法連線到區塊鏈或讀取合約資料');
    }
  };

  // 如果尚未登入或沒有 account，則顯示提示
  if (!account) {
    return <div>請先登入以檢視個人資訊。</div>;
  }

  return (
    <div className={styles['page-container']}>
      <header className={styles.header}>
        <h1 className={styles.title}>個人頁面</h1>
      </header>
      <main className={styles['main-content']}>
        {/* 錯誤訊息顯示 */}
        {error && <p style={{ color: 'red' }}>{error}</p>}

        {/* 顯示用戶基本資訊 */}
        <div className={styles['user-info']}>
          <h2>帳號資訊</h2>
          <ul>
            <li>使用者名稱：{account.username}</li>
            <li>姓名：{account.name}</li>
            <li>電子郵件：{account.email}</li>
            <li>創建日期：{new Date(account.createdAt).toLocaleDateString()}</li>
          </ul>
        </div>

        {/* 顯示區塊鏈帳號資訊 */}
        <div className={styles['blockchain-info']}>
          <h2>區塊鏈帳號</h2>
          {account.blockchain_address ? (
            <div>
              <p>地址：{account.blockchain_address}</p>
              <p>
                餘額：
                {blockchainData
                  ? blockchainData.balance
                  : '載入中...'}
              </p>
            </div>
          ) : (
            <div>
              <p>目前還沒有區塊鏈帳號，請前往創建。</p>
              {/* 這裡可以放上你創建區塊鏈帳號的按鈕或邏輯 */}
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default PersonalPage;
