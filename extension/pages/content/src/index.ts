import { initHighlighter } from './highlighter';
import { processCurrentArticle } from './articleProcessor';

// Initialize highlighter and process article when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    initHighlighter();

    // Process the article content for vector search after a short delay
    // This ensures the page is fully rendered
    setTimeout(() => {
      processCurrentArticle();
    }, 1500);
  });
} else {
  initHighlighter();

  // Process the article content for vector search after a short delay
  setTimeout(() => {
    processCurrentArticle();
  }, 1500);
}

// Also process article on history state change (for SPAs)
window.addEventListener('popstate', () => {
  // Wait for the new page content to load
  setTimeout(() => {
    processCurrentArticle();
  }, 1500);
});

// Listen for messages from the background script
chrome.runtime.onMessage.addListener(message => {
  if (message.type === 'REPROCESS_ARTICLE') {
    processCurrentArticle();
  }
});

console.log('Content script loaded');
