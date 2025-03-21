import React, { useState, useEffect } from 'react';
import axios from 'axios';
import styles from './styles/PersonalPage.module.css';

const PersonalPage = () => {
    const [userData, setUserData] = useState(null);
    const [blockchainData, setBlockchainData] = useState(null);
    const [hasBlockchainAccount, setHasBlockchainAccount] = useState(false);

    useEffect(() => {
        const fetchUserData = async () => {
            try {
                const response = await axios.get('/api/user');
                setUserData(response.data);
                if (response.data.blockchain_address) {
                    setHasBlockchainAccount(true);
                    const balanceResponse = await axios.get(`/api/blockchain/balance/${response.data.blockchain_address}`);
                    setBlockchainData(balanceResponse.data);
                }
            } catch (error) {
                console.error('Error fetching user data:', error);
            }
        };
        fetchUserData();
    }, []);

    const handleCreateBlockchainAccount = async () => {
        try {
            await axios.post('/api/blockchain/create');
            alert('區塊鏈帳號已成功創建！');
            window.location.reload();
        } catch (error) {
            console.error('Error creating blockchain account:', error);
        }
    };

    return (
        <div className={styles['page-container']}>
            <header className={styles.header}>
                <h1 className={styles.title}>個人頁面</h1>
            </header>
            <main className={styles['main-content']}>
                <div className={styles['user-info']}>
                    <h2>帳號資訊</h2>
                    {userData ? (
                        <ul>
                            <li>使用者名稱：{userData.username}</li>
                            <li>姓名：{userData.name}</li>
                            <li>電子郵件：{userData.email}</li>
                            <li>創建日期：{new Date(userData.createdAt).toLocaleDateString()}</li>
                        </ul>
                    ) : (
                        <p>載入中...</p>
                    )}
                </div>
                <div className={styles['blockchain-info']}>
                    <h2>區塊鏈帳號</h2>
                    {hasBlockchainAccount ? (
                        <div>
                            <p>地址：{userData.blockchain_address}</p>
                            <p>餘額：{blockchainData?.balance || '載入中...'}</p>
                        </div>
                    ) : (
                        <div>
                            <p>目前還沒有區塊鏈帳號，請前往創建。</p>
                            <button onClick={handleCreateBlockchainAccount} className={styles['create-btn']}>創建區塊鏈帳號</button>
                        </div>
                    )}
                </div>
            </main>
        </div>
    );
};

export default PersonalPage;
