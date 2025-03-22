import { initHighlighter } from './highlighter';

// Initialize highlighter when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    initHighlighter();
  });
} else {
  initHighlighter();
}

console.log('Content script loaded');
