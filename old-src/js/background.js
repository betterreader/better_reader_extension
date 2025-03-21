chrome.runtime.onInstalled.addListener(() => {
  console.log('BetterReader extension installed!');
  // Initialize the side panel
  chrome.sidePanel.setOptions({
    enabled: true,
    path: 'sidepanel.html',
  });
});

// Open the side panel when the action button is clicked
chrome.action.onClicked.addListener(tab => {
  chrome.sidePanel.open({ tabId: tab.id });
});

// Handle messages from the side panel
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'getArticleContent') {
    chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
      if (tabs[0]) {
        chrome.scripting.executeScript(
          {
            target: { tabId: tabs[0].id },
            function: extractArticleContent,
          },
          results => {
            if (chrome.runtime.lastError) {
              console.error('Error executing script:', chrome.runtime.lastError);
              sendResponse({ error: chrome.runtime.lastError.message });
            } else if (results && results[0]) {
              console.log('Content extracted successfully:', results[0].result);
              sendResponse({
                content: results[0].result.content,
                title: results[0].result.title,
                url: tabs[0].url,
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

    // Return true to indicate we'll respond asynchronously
    return true;
  }
});

// Content extraction function to be injected into the page
function extractArticleContent() {
  const getArticleText = () => {
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
      '.post',
      '.blog-post',
      '#main',
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
    return article.innerText;
  };

  // For debugging
  console.log('Extracting content from:', document.title);

  return {
    content: getArticleText(),
    title: document.title,
  };
}
