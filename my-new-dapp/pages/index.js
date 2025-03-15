import { useEffect } from 'react';
import { useRouter } from 'next/router';

export default function Home() {
  const router = useRouter();
  
  useEffect(() => {
    router.replace('/home');
  }, []);

  // 返回空白或加載指示器，因為頁面會立即重定向
  return <div>正在重定向到首頁...</div>;
}