/**
 * Utilities for interacting with the vector search API endpoints
 */

/**
 * Base URL for API requests
 */
const API_BASE_URL = 'http://localhost:5007';

/**
 * Process an article to generate embeddings and store in Supabase
 *
 * @param url - The URL of the article
 * @param title - The title of the article
 * @param content - The full text content of the article
 * @param userId - Optional user ID
 * @returns Promise with the API response
 */
export async function processArticle(url: string, title: string, content: string, userId?: string): Promise<any> {
  try {
    console.log(`Processing article: "${title}" (${url.substring(0, 50)}...)`);
    console.log(`Content length: ${content.length} characters`);

    const response = await fetch(`${API_BASE_URL}/api/process_article`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        url,
        title,
        content,
        user_id: userId,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`API error (${response.status}): ${errorText}`);
      throw new Error(`API error: ${response.status} - ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error processing article:', error);
    throw error;
  }
}

/**
 * Perform a vector search query
 *
 * @param query - The search query
 * @param limit - Maximum number of results to return
 * @param articleId - Optional article ID to search within
 * @returns Promise with the search results
 */
export async function vectorSearch(query: string, limit: number = 5, articleId?: string): Promise<any> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/vector_search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query,
        limit,
        article_id: articleId,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`API error (${response.status}): ${errorText}`);
      throw new Error(`API error: ${response.status} - ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error performing vector search:', error);
    throw error;
  }
}

/**
 * Get an answer to a question using vector search and context
 *
 * @param query - The question to answer
 * @param limit - Maximum number of context segments to use
 * @param articleId - Optional article ID to search within
 * @returns Promise with the answer and context
 */
export async function vectorSearchQA(query: string, limit: number = 5, articleId?: string): Promise<any> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/vector_search_qa`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query,
        limit,
        article_id: articleId,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`API error (${response.status}): ${errorText}`);
      throw new Error(`API error: ${response.status} - ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error performing vector search QA:', error);
    throw error;
  }
}

/**
 * Get article recommendations based on similarity
 *
 * @param articleId - ID of the article to get recommendations for
 * @param limit - Maximum number of recommendations to return
 * @returns Promise with the article recommendations
 */
export async function getArticleRecommendations(articleId: string, limit: number = 5): Promise<any> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/get_article_recommendations`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        article_id: articleId,
        limit,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`API error (${response.status}): ${errorText}`);
      throw new Error(`API error: ${response.status} - ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error getting article recommendations:', error);
    throw error;
  }
}

/**
 * Enhanced chat function that leverages vector search across all articles
 *
 * @param message - The user's message or question
 * @param conversationId - Optional ID to track conversation history
 * @param conversationHistory - Optional array of previous messages
 * @param currentArticleId - Optional ID of the current article
 * @param currentArticleContent - Optional content of the current article
 * @returns Promise with the API response
 */
export async function enhancedChat(
  message: string,
  conversationId?: string,
  conversationHistory?: Array<{ role: 'user' | 'assistant'; content: string }>,
  currentArticleId?: string,
  currentArticleContent?: string,
): Promise<any> {
  try {
    console.log(`Sending enhanced chat request: "${message}"`);

    const response = await fetch(`${API_BASE_URL}/api/enhanced_chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
        conversation_id: conversationId,
        conversation_history: conversationHistory,
        current_article_id: currentArticleId,
        current_article_content: currentArticleContent,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`API error (${response.status}): ${errorText}`);
      throw new Error(`API error: ${response.status} - ${errorText}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Error in enhanced chat:', error);
    throw error;
  }
}
