import 'webextension-polyfill';
import { exampleThemeStorage } from '@extension/storage';

// Define the expected message format for clarity
interface GetArticleContentMessage {
  action: 'getArticleContent';
}

interface SendSelectedTextMessage {
  action: 'sendSelectedText';
  text: string;
}

type BackgroundMessage = GetArticleContentMessage | SendSelectedTextMessage;

interface ArticleContent {
  content: string;
  title: string;
}

interface SuccessResponse extends ArticleContent {
  url: string;
}

interface ErrorResponse {
  error: string;
}

type MessageResponse = SuccessResponse | ErrorResponse;

exampleThemeStorage.get().then(theme => {
  console.log('theme', theme);
});

console.log('Background loaded');
console.log("Edit 'chrome-extension/src/background/index.ts' and save to reload.");

// Create context menu item for "Explain with AI"
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: 'explainWithAI',
    title: 'Explain with AI',
    contexts: ['selection'],
  });
});

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === 'explainWithAI' && info.selectionText && tab?.id) {
    // Open the side panel if it's not already open
    chrome.sidePanel.setOptions({
      tabId: tab.id,
      path: 'side-panel/index.html',
      enabled: true,
    });

    chrome.sidePanel.open({ tabId: tab.id });

    // Store the selected text to be retrieved by the side panel
    chrome.storage.local.set({
      selectedTextForExplanation: {
        text: info.selectionText,
        timestamp: Date.now(),
      },
    });

    // Send a message to the side panel with the selected text
    // This is a backup approach in case the panel is already open
    chrome.runtime.sendMessage({
      action: 'sendSelectedText',
      text: info.selectionText,
    });
  }
});

// Background script to open side panel on click
chrome.action.onClicked.addListener((tab: chrome.tabs.Tab) => {
  if (!tab.id) return;

  chrome.sidePanel.setOptions({
    tabId: tab.id,
    path: 'side-panel/index.html',
    enabled: true,
  });

  chrome.sidePanel.open({ tabId: tab.id });
});

// Listen for messages from the side panel
chrome.runtime.onMessage.addListener(
  (
    message: BackgroundMessage,
    sender: chrome.runtime.MessageSender,
    sendResponse: (response: MessageResponse) => void,
  ): boolean => {
    if (message.action === 'getArticleContent') {
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs: chrome.tabs.Tab[]) => {
        const activeTab = tabs[0];
        if (activeTab && activeTab.id) {
          chrome.scripting.executeScript(
            {
              target: { tabId: activeTab.id },
              func: extractArticleContent, // Changed 'function' to 'func'
            },
            (results?: chrome.scripting.InjectionResult<ArticleContent>[]) => {
              if (chrome.runtime.lastError) {
                console.error('Error executing script:', chrome.runtime.lastError);
                sendResponse({ error: chrome.runtime.lastError.message || 'Unknown error' });
              } else if (results && results[0]?.result) {
                console.log('Content extracted successfully:', results[0].result);
                sendResponse({
                  content: results[0].result.content,
                  title: results[0].result.title,
                  url: activeTab.url || '',
                });
              } else {
                console.error('No results from content extraction');
                sendResponse({ error: 'Failed to extract content' });
              }
            },
          );
        } else {
          console.error('No active tab found');
          sendResponse({ error: 'No active tab found' });
        }
      });

      return true;
    }
    return false;
  },
);

// Content extraction function to be injected into the page
function extractArticleContent(): ArticleContent {
  const getArticleText = (): string => {
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
      '.post',
      '.blog-post',
      '#main',
    ];

    let article: HTMLElement | null = null;

    for (const selector of selectors) {
      const element = document.querySelector(selector);
      if (element instanceof HTMLElement) {
        article = element;
        break;
      }
    }

    // If no specific article element is found, fallback to the body
    if (!article) {
      article = document.body;
    }

    return article.innerText;
  };

  return {
    content: getArticleText(),
    title: document.title,
  };
}
