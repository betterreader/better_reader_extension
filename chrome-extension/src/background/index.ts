import 'webextension-polyfill';
import { exampleThemeStorage } from '@extension/storage';

// Define the expected message format for clarity
interface GetArticleContentMessage {
  action: 'getArticleContent';
}

interface SendSelectedTextMessage {
  action: 'sendSelectedText';
  text: string;
  paragraph: string;
  title: string;
  mode: string;
}

interface ExplainWithAIMessage {
  action: 'explainWithAI';
  text: string;
  mode: string;
}

interface OpenSidePanelMessage {
  action: 'openSidePanel';
}

type BackgroundMessage =
  | GetArticleContentMessage
  | SendSelectedTextMessage
  | ExplainWithAIMessage
  | OpenSidePanelMessage;

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
  // Create parent menu item
  chrome.contextMenus.create({
    id: 'explainWithAI',
    title: 'Explain with AI',
    contexts: ['selection'],
  });

  // Create child menu items for different explanation modes
  chrome.contextMenus.create({
    id: 'explainWithAI_simple',
    parentId: 'explainWithAI',
    title: 'Simple',
    contexts: ['selection'],
  });

  chrome.contextMenus.create({
    id: 'explainWithAI_detailed',
    parentId: 'explainWithAI',
    title: 'Detailed',
    contexts: ['selection'],
  });

  chrome.contextMenus.create({
    id: 'explainWithAI_examples',
    parentId: 'explainWithAI',
    title: 'Examples',
    contexts: ['selection'],
  });
});

// Handle context menu clicks
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId.toString().startsWith('explainWithAI') && info.selectionText && tab?.id) {
    // Determine which mode was selected
    let mode = 'simple'; // Default mode
    if (info.menuItemId === 'explainWithAI_detailed') {
      mode = 'detailed';
    } else if (info.menuItemId === 'explainWithAI_examples') {
      mode = 'examples';
    }

    // Open the side panel if it's not already open
    chrome.sidePanel.setOptions({
      tabId: tab.id,
      path: 'side-panel/index.html',
      enabled: true,
    });

    chrome.sidePanel.open({ tabId: tab.id });

    // Execute a content script to get the surrounding paragraph and page title
    chrome.scripting.executeScript(
      {
        target: { tabId: tab.id },
        func: selectedText => {
          // Function to get the surrounding paragraph of the selected text
          const getSurroundingParagraph = (selectedText: string) => {
            const selection = window.getSelection();
            if (!selection || selection.rangeCount === 0) return '';

            const range = selection.getRangeAt(0);
            let node = range.startContainer;

            // Navigate up to find the paragraph or block-level element
            while (node && node.nodeType === Node.TEXT_NODE) {
              node = node.parentNode;
            }

            // Get the closest paragraph or block element
            const blockElement = node.closest('p, div, article, section') || node;
            return blockElement.textContent || '';
          };

          return {
            selectedText: selectedText,
            paragraph: getSurroundingParagraph(selectedText),
            title: document.title,
          };
        },
        args: [info.selectionText],
      },
      results => {
        if (results && results[0]?.result) {
          const { selectedText, paragraph, title } = results[0].result;

          // Store the selected text and context to be retrieved by the side panel
          chrome.storage.local.set({
            selectedTextForExplanation: {
              text: selectedText,
              paragraph: paragraph,
              title: title,
              mode: mode,
              timestamp: Date.now(),
            },
          });

          // Try to send a message to the side panel with the selected text and context
          try {
            chrome.runtime.sendMessage(
              {
                action: 'sendSelectedText',
                text: selectedText,
                paragraph: paragraph,
                title: title,
                mode: mode,
              },
              response => {
                // Check for response
                if (chrome.runtime.lastError) {
                  // Suppress errors when the receiving end doesn't exist yet
                  console.log('Side panel not ready to receive messages yet:', chrome.runtime.lastError.message);
                  console.log('Will use storage for communication instead.');
                } else if (response && response.status === 'received') {
                  console.log('Message successfully delivered to side panel');
                }
              },
            );
          } catch (error) {
            // Fallback to storage-only approach if message sending fails
            console.log('Error sending message to side panel:', error);
            console.log('Using storage for communication instead.');
          }
        }
      },
    );
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
    message: BackgroundMessage | any,
    sender: chrome.runtime.MessageSender,
    sendResponse: (response: MessageResponse | any) => void,
  ): boolean => {
    // Handle explainWithAI action from ELI5 button
    if (message.action === 'explainWithAI') {
      console.log('Received explainWithAI message:', message);

      // Get the current active tab
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs: chrome.tabs.Tab[]) => {
        const activeTab = tabs[0];
        if (activeTab && activeTab.id) {
          // Execute script to get surrounding paragraph and title
          chrome.scripting.executeScript(
            {
              target: { tabId: activeTab.id },
              func: selectedText => {
                // Function to get the surrounding paragraph of the selected text
                const getSurroundingParagraph = (selectedText: string) => {
                  const selection = window.getSelection();
                  if (!selection || selection.rangeCount === 0) return '';

                  const range = selection.getRangeAt(0);
                  let node = range.startContainer;

                  // Navigate up to find the paragraph or block-level element
                  while (node && node.nodeType === Node.TEXT_NODE) {
                    node = node.parentNode;
                  }

                  // Get the closest paragraph or block element
                  const blockElement = node.closest('p, div, article, section') || node;
                  return blockElement.textContent || '';
                };

                return {
                  selectedText: selectedText,
                  paragraph: getSurroundingParagraph(selectedText),
                  title: document.title,
                };
              },
              args: [message.text],
            },
            results => {
              if (results && results[0]?.result) {
                const { selectedText, paragraph, title } = results[0].result;

                // Store the selected text and context to be retrieved by the side panel
                chrome.storage.local.set({
                  selectedTextForExplanation: {
                    text: selectedText,
                    paragraph: paragraph,
                    title: title,
                    mode: message.mode,
                    timestamp: Date.now(),
                  },
                });

                // Try to send a message to the side panel with the selected text and context
                try {
                  chrome.runtime.sendMessage(
                    {
                      action: 'sendSelectedText',
                      text: selectedText,
                      paragraph: paragraph,
                      title: title,
                      mode: message.mode,
                    },
                    response => {
                      if (chrome.runtime.lastError) {
                        console.log('Side panel not ready to receive messages yet:', chrome.runtime.lastError.message);
                        console.log('Will use storage for communication instead.');
                      } else if (response && response.status === 'received') {
                        console.log('Message successfully delivered to side panel');
                      }
                    },
                  );
                } catch (error) {
                  console.log('Error sending message to side panel:', error);
                  console.log('Using storage for communication instead.');
                }
              }
            },
          );
        }
      });

      return true;
    }

    // Handle openSidePanel action
    if (message.action === 'openSidePanel') {
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs: chrome.tabs.Tab[]) => {
        const activeTab = tabs[0];
        if (activeTab && activeTab.id) {
          // Open the side panel
          chrome.sidePanel.setOptions({
            tabId: activeTab.id,
            path: 'side-panel/index.html',
            enabled: true,
          });

          chrome.sidePanel.open({ tabId: activeTab.id });
        }
      });

      return true;
    }

    if (message.action === 'getArticleContent') {
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs: chrome.tabs.Tab[]) => {
        const activeTab = tabs[0];
        if (activeTab && activeTab.id) {
          // Check if the URL is a restricted URL (chrome://, chrome-extension://, etc.)
          const url = activeTab.url || '';
          if (url.startsWith('chrome://') || url.startsWith('chrome-extension://') || url.startsWith('devtools://')) {
            console.error('Cannot access restricted URL:', url);
            sendResponse({
              error: 'Cannot access this page. Try opening a regular web page.',
              content: 'Hi there! I can help you understand this article better. Ask me anything about it.',
              title: 'Better Reader',
              url: url,
            });
            return;
          }

          chrome.scripting.executeScript(
            {
              target: { tabId: activeTab.id },
              func: extractArticleContent,
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
