import { useEffect, useState } from 'react';
import Web3 from 'web3';


export default function Accounts() {
  const [accounts, setAccounts] = useState([]);

  useEffect(() => {
    async function loadAccounts() {
      const web3 = new Web3('http://25.3.19.183:7545'); // Ganache URL
      const accountsList = await web3.eth.getAccounts();
      setAccounts(accountsList);
    }

    loadAccounts();
  }, []);

  return (
    <div style={{ padding: '20px' }}>
      <h1>Ganache 帳戶列表</h1>
      <ul>
        {accounts.map((account, index) => (
          <li key={index}>{account}</li>
        ))}
      </ul>
    </div>
  );
}
