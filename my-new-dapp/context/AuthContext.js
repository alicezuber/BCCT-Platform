// context/AuthContext.js
import React, { createContext, useContext, useState } from 'react';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
    const [account, setAccount] = useState(null);

    const login = (username, token) => {
        localStorage.setItem('token', token);
        setAccount(username);
    };

    const logout = () => {
        localStorage.removeItem('token');
        setAccount(null);
    };

    return (
        <AuthContext.Provider value={{ account, login, logout }}>
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => useContext(AuthContext);
