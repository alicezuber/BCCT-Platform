// context/AuthContext.js
import React, { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
    const [account, setAccount] = useState(null);
    const [isInitialized, setIsInitialized] = useState(false);

    // 初始化時檢查localStorage的token
    useEffect(() => {
        try {
            const token = localStorage.getItem('token');
            if (token) {
                // 嘗試解析token（簡單驗證）
                const tokenData = parseJwt(token);
                
                if (tokenData && tokenData.username) {
                    console.log('從localStorage恢復登入狀態', { username: tokenData.username });
                    setAccount(tokenData.username);
                } else {
                    console.warn('token無效或已過期，清除本地存儲');
                    localStorage.removeItem('token');
                }
            }
        } catch (error) {
            console.error('驗證token時出錯', error);
            localStorage.removeItem('token');
        } finally {
            setIsInitialized(true);
        }
    }, []);

    // 簡單解析JWT（只是讓前端能夠讀取 token 中的資訊，僅供前端驗證，不是加密）
    const parseJwt = (token) => {
        try {
            const base64Url = token.split('.')[1];
            const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
            const jsonPayload = decodeURIComponent(
                atob(base64).split('').map(function(c) {
                    return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
                }).join('')
            );
            return JSON.parse(jsonPayload);
        } catch (e) {
            console.error('無法解析token', e);
            return null;
        }
    };

    const login = (username, token) => {
        console.log('執行登入操作', { username });
        localStorage.setItem('token', token);
        setAccount(username);
    };

    const logout = () => {
        console.log('執行登出操作', { currentUser: account });
        localStorage.removeItem('token');
        setAccount(null);
    };

    
    return (
        <AuthContext.Provider value={{ 
            account, 
            login, 
            logout, 
            isInitialized,

        }}>
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => useContext(AuthContext);