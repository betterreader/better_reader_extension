import { useEffect } from 'react';
import { Highlighter } from './Highlighter';

export default function App() {
  useEffect(() => {
    console.log('content ui loaded');
  }, []);

  return <Highlighter />;
}
