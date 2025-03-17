import { useEffect } from 'react';
import { useRouter } from 'next/router';

export default function Home() {
  // 這個函數正常情況下不會被執行，因為會先被重定向
  return null;
}

export async function getServerSideProps() {
  return {
    redirect: {
      destination: '/home',
      permanent: true, // 設置為 true 表示 301 永久重定向，false 表示 302 臨時重定向
    }
  };
}