// Content script that runs in the context of web pages
// Listen for messages from the background script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'extractContent') {
    // Extract the page content and send it back
    const content = extractPageContent();
    sendResponse({ content });
  }

  // Return true to indicate we'll respond asynchronously
  return true;
});

// Function to extract the main content of the page
function extractPageContent() {
  // Try to find the main article content
  const selectors = [
    'article',
    '[role="article"]',
    '.article',
    '.post-content',
    '.entry-content',
    '.content',
    'main',
    '#content',
    '.main-content',
  ];

  let article = null;

  for (const selector of selectors) {
    const element = document.querySelector(selector);
    if (element) {
      article = element;
      break;
    }
  }

  // If we can't find a specific article element, fallback to the body
  if (!article) {
    article = document.body;
  }

  // Extract the text content
  return {
    title: document.title,
    url: window.location.href,
    content: article.innerText,
    html: article.innerHTML,
  };
}

// Remove any existing UI injected by previous iterations
const existingUI = document.getElementById('better-reader-ui');
if (existingUI) {
  existingUI.remove();
}

// For debugging purposes
console.log('BetterReader content script loaded');

document.addEventListener('DOMContentLoaded', () => {
  const actionButton = document.getElementById('actionButton');
  if (actionButton) {
    actionButton.addEventListener('click', () => {
      alert('Button clicked!');
    });
  }
});
