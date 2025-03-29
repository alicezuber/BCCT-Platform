import React, { useState } from 'react';
import Web3 from 'web3';
import CarbonCreditToken from '../build/contracts/CarbonCreditToken.json';

const contractABI = CarbonCreditToken.abi;
// 將此處換成你部署後的合約地址
const contractAddress = '0x1A4c163eD19001c29eC0356B120026C0f97bc8A4';
// 連接到本地 Ganache 節點（預設端口 7545）
const web3 = new Web3('http://25.3.19.183:7545');

export default function HomePage() {
  // 使用者輸入的私鑰（注意：僅用於測試環境，勿於生產環境使用）
  const [privateKey, setPrivateKey] = useState('');
  // 透過私鑰轉換後的帳戶
  const [account, setAccount] = useState(null);
  // 想要鑄造的代幣數量
  const [amount, setAmount] = useState('');
  // 狀態訊息
  const [status, setStatus] = useState('');
  // 顯示該帳戶的 CCT 餘額 (以可讀單位顯示)
  const [balance, setBalance] = useState('');

  // 登入：利用使用者輸入的私鑰建立帳戶 (signer)
  const handleLogin = () => {
    try {
      const acct = web3.eth.accounts.privateKeyToAccount(privateKey);
      // 將此帳戶加入 Web3 的錢包管理中，方便後續簽署交易
      web3.eth.accounts.wallet.add(acct);
      setAccount(acct);
      setStatus(`登入成功，帳戶地址：${acct.address}`);
      // 登入成功後自動查詢餘額
      handleGetBalance(acct.address);
    } catch (error) {
      setStatus('登入失敗：' + error.message);
    }
  };

  // 查詢指定地址的 CCT 餘額
  const handleGetBalance = async (addr) => {
    try {
      // 建立合約實例（不需要指定 from 進行查詢）
      const contract = new web3.eth.Contract(contractABI, contractAddress);
      const cctBalance = await contract.methods.balanceOf(addr).call();
      // 假設代幣為 18 小數位，使用 fromWei 轉換成可讀格式
      const readableBalance = web3.utils.fromWei(cctBalance, 'ether');
      setBalance(readableBalance);
    } catch (error) {
      console.error(error);
      setStatus('查詢餘額失敗：' + error.message);
    }
  };

  // 執行 mint 函式，發行代幣
  const handleMint = async () => {
    if (!account) {
      setStatus('請先登入 (輸入私鑰)');
      return;
    }
    try {
      // 建立合約實例，並指定 from 為登入帳戶
      const contract = new web3.eth.Contract(contractABI, contractAddress, {
        from: account.address,
      });
      
      // 假設代幣小數位為 18，將前端輸入的數字轉換成最小單位
      const tokenAmount = web3.utils.toWei(amount, 'ether');

      setStatus('送出交易中，請稍後...');
      // 呼叫 mint 函式
      const receipt = await contract.methods.mint(tokenAmount).send({ from: account.address });
      setStatus(`成功鑄造 ${amount} 代幣！交易哈希：${receipt.transactionHash}`);
      // 鑄造完成後，重新查詢餘額
      handleGetBalance(account.address);
    } catch (error) {
      setStatus('鑄造失敗：' + error.message);
    }
  };

  return (
    <div style={{ padding: '20px' }}>
      <h1>Carbon Credit Token - Mint & 查詢餘額</h1>

      {/* 登入區塊 */}
      {!account ? (
        <div>
          <h2>登入 (模擬)</h2>
          <input
            type="text"
            placeholder="輸入 Ganache 帳戶私鑰"
            value={privateKey}
            onChange={(e) => setPrivateKey(e.target.value)}
            style={{ width: '400px' }}
          />
          <button onClick={handleLogin} style={{ marginLeft: '8px' }}>
            登入
          </button>
        </div>
      ) : (
        // 已登入時顯示帳戶地址與餘額查詢功能
        <div>
          <h2>已登入</h2>
          <p>帳戶地址：{account.address}</p>
          <p>目前 CCT 餘額：{balance ? balance : '查詢中...'}</p>
          <button onClick={() => handleGetBalance(account.address)}>查詢餘額</button>
          <hr />
          <h2>鑄造代幣</h2>
          <input
            type="text"
            placeholder="輸入鑄造數量 (例如 10)"
            value={amount}
            onChange={(e) => setAmount(e.target.value)}
            style={{ width: '200px' }}
          />
          <button onClick={handleMint} style={{ marginLeft: '8px' }}>
            鑄造
          </button>
        </div>
      )}

      <p style={{ marginTop: '16px' }}>{status}</p>
    </div>
  );
}
