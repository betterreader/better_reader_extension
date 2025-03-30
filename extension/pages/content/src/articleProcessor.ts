import { processArticle } from '@extension/shared/lib/utils/vectorSearch';

/**
 * Interface for article data
 */
interface ArticleContent {
  content: string;
  title: string;
  url: string;
}

/**
 * Extracts article content from the current page
 * @returns Article content, title, and URL
 */
export function extractArticleContent(): ArticleContent {
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

    // Clean up the text a bit
    let text = article.innerText;

    // Remove excessive whitespace
    text = text.replace(/\s+/g, ' ').trim();

    return text;
  };

  return {
    content: getArticleText(),
    title: document.title,
    url: window.location.href,
  };
}

/**
 * Checks if the current URL should be processed
 * @returns Whether the page should be processed
 */
function shouldProcessPage(): boolean {
  const url = window.location.href;

  // Skip processing for non-article pages
  if (
    url.startsWith('chrome://') ||
    url.startsWith('chrome-extension://') ||
    url.startsWith('about:') ||
    url.startsWith('chrome-search://') ||
    url.startsWith('devtools://') ||
    url.startsWith('file://') ||
    url.includes('github.com') || // Skip GitHub
    url.includes('mail.google.com') // Skip Gmail
  ) {
    return false;
  }

  return true;
}

/**
 * Processes the current article by extracting content and sending it to the server for processing
 */
export async function processCurrentArticle(): Promise<void> {
  console.log('Better Reader: Starting article processing');
  try {
    // Check if this page should be processed
    if (!shouldProcessPage()) {
      console.log('Skipping article processing for this page');
      return;
    }

    // Extract article content
    const article = extractArticleContent();
    console.log('Better Reader: Article extracted', { title: article.title, contentLength: article.content.length });

    // Skip if article content is too short (likely not an actual article)
    if (article.content.length < 500) {
      console.log('Article content too short, skipping processing');
      return;
    }

    console.log('Extracting article content for processing:', {
      title: article.title,
      url: article.url,
      contentLength: article.content.length,
    });

    // Send to API for processing
    const response = await processArticle(article.url, article.title, article.content);

    if (response.success) {
      console.log('Article successfully processed', {
        article_id: response.article_id,
        topics: response.topics,
        summary: response.summary ? response.summary.substring(0, 50) + '...' : 'No summary',
      });

      // Notify the background script that article has been processed
      chrome.runtime.sendMessage({
        type: 'ARTICLE_PROCESSED',
        data: {
          url: article.url,
          title: article.title,
          article_id: response.article_id,
        },
      });
    } else {
      console.error('Failed to process article:', response.error);
    }
  } catch (error) {
    console.error('Error processing article:', error);
  }
}
