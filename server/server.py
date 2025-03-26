from flask import Flask, request, jsonify
from flask_cors import CORS
import time
from dotenv import load_dotenv
import requests
import json
import os
import re
import uuid
import numpy as np
from supabase import create_client, Client
from typing import List, Dict, Any, Optional, Tuple
import datetime
from functools import wraps
import openai
from flask import g
import atexit

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Enable CORS for all routes with more specific configuration
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_ANON_KEY = os.getenv('SUPABASE_ANON_KEY')
SUPABASE_SERVICE_ROLE_KEY = os.getenv('SUPABASE_SERVICE_ROLE_KEY')

# Use service role key for database operations
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Configure OpenAI API
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

# NOTE: These supabase functions aren't being used at the moment
def require_supabase_user(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # TEMPORARILY DISABLED FOR TESTING - REMOVE THIS COMMENT WHEN DONE TESTING
        # Skip authentication check for now
        return f(*args, **kwargs)

        # Original authentication code - commented out for testing
        """
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401

        access_token = auth_header.split(" ")[1]

        try:
            user_response = supabase.auth.get_user(access_token)
            user = user_response.user
            if not user:
                raise Exception("User not found")
            g.user = user  # Attach to Flask's request context
        except Exception as e:
            print("Auth failed:", e)
            return jsonify({"error": "Invalid or expired token"}), 401
        """

        return f(*args, **kwargs)
    return decorated_function


@app.route('/api/highlights', methods=['POST'])
@require_supabase_user
def create_highlight():
    try:
        data = request.json
        required_fields = ['url', 'color', 'start_xpath', 'start_offset', 'end_xpath', 'end_offset', 'text']
        
        if not all(field in data for field in required_fields):
            print(data)
            return jsonify({'error': 'Missing required fields'}), 400
        
        user_id = g.user.id 
            
        # Insert highlight into Supabase
        highlight_data = {
            'url': data['url'],
            'color': data['color'],
            'start_xpath': data['start_xpath'],
            'start_offset': data['start_offset'],
            'end_xpath': data['end_xpath'],
            'end_offset': data['end_offset'],
            'comment': data.get('comment', ' '),
            'user_id': user_id,
            'text': data['text']
        }
        
        result = supabase.table('highlight').insert(highlight_data).execute()
        return jsonify(result.data[0])
    except Exception as e:
        print(f"Error creating highlight: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/highlights', methods=['GET'])
@require_supabase_user
def get_highlights():
    try:
        url = request.args.get('url')
        if not url:
            return jsonify({'error': 'URL parameter is required'}), 400
            
        # Query highlights for the specific URL and user
        user_id = g.user.id
        result = supabase.table('highlight')\
            .select('*')\
            .eq('user_id', user_id)\
            .eq('url', url)\
            .execute()
            
        return jsonify(result.data)
    except Exception as e:
        print(f"Error getting highlights: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/highlights/<int:highlight_id>', methods=['DELETE'])
@require_supabase_user
def delete_highlight(highlight_id):
    try:
        # Delete the highlight
        user_id = g.user.id
        result = supabase.table('highlight')\
            .delete()\
            .eq('id', highlight_id)\
            .eq('user_id', user_id)\
            .execute()
            
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error deleting highlight: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/highlights/<int:highlight_id>', methods=['PUT'])
@require_supabase_user
def update_highlight(highlight_id):
    try:
        data = request.json
        # TODO: choose which fields to update
        update_data = {k: v for k, v in data.items()}
        
        if not update_data:
            return jsonify({'error': 'No valid fields to update'}), 400
            
        # Update the highlight
        user_id = g.user.id
        result = supabase.table('highlight')\
            .update(update_data)\
            .eq('id', highlight_id)\
            .eq('user_id', user_id)\
            .execute()
            
        return jsonify(result.data[0])
    except Exception as e:
        print(f"Error updating highlight: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Gemini API configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
PORT = int(os.getenv('PORT', 5007))

# Perplexity API configuration
PERPLEXITY_HEADERS = {
    "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
    "Content-Type": "application/json"
}

# Create a session for requests to avoid semaphore leaks
session = requests.Session()

# Register cleanup function to close session on shutdown
def cleanup_resources():
    session.close()
    print("Cleaned up resources")

atexit.register(cleanup_resources)

# Text segmentation and embedding utilities
def segment_text(text: str, max_chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Segment article text into overlapping chunks for embedding.
    
    Args:
        text: The article text to segment
        max_chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text segments
    """
    # Clean the text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # If text is shorter than max_chunk_size, return it as is
    if len(text) <= max_chunk_size:
        return [text]
    
    segments = []
    start = 0
    
    while start < len(text):
        # Get chunk of max_chunk_size
        end = start + max_chunk_size
        
        # If we're not at the end of the text, try to find a sentence break
        if end < len(text):
            # Look for a sentence break (period followed by space)
            sentence_break = text.rfind('. ', start, end)
            if sentence_break != -1:
                end = sentence_break + 1  # Include the period
        
        # Add the segment
        segments.append(text[start:end].strip())
        
        # Move the start position, accounting for overlap
        start = end - overlap if end < len(text) else end
    
    return segments

def generate_embedding(text: str) -> List[float]:
    """
    Generate an embedding for a text segment using OpenAI's API.
    
    Args:
        text: The text to generate an embedding for
        
    Returns:
        Embedding vector as a list of floats
    """
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        raise

def store_article_embeddings(
    url: str, 
    title: str, 
    segments: List[str], 
    embeddings: List[List[float]],
    user_id: Optional[str] = None
) -> bool:
    """
    Store article segments and their embeddings in Supabase.
    
    Args:
        url: The URL of the article
        title: The title of the article
        segments: List of text segments
        embeddings: List of embedding vectors
        user_id: Optional user ID
        
    Returns:
        Success status
    """
    try:
        article_id = str(uuid.uuid4())
        article_data = {
            'id': article_id,
            'url': url,
            'title': title,
            'user_id': user_id,
            'created_at': datetime.datetime.now().isoformat()
        }
        
        # Insert article metadata
        print(f"Inserting article with ID: {article_id}")
        supabase.table('articles').insert(article_data).execute()
        
        # Insert segments and embeddings
        for i, (segment, embedding) in enumerate(zip(segments, embeddings)):
            segment_data = {
                'id': str(uuid.uuid4()),
                'article_id': article_id,
                'segment_text': segment,
                'embedding': embedding,
                'segment_index': i
            }
            supabase.table('article_segments').insert(segment_data).execute()
        
        return True
    except Exception as e:
        print(f"Error storing article embeddings: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return False

def vector_search(
    query: str, 
    limit: int = 5, 
    article_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Perform a vector search using the query embedding.
    
    Args:
        query: The search query
        limit: Maximum number of results to return
        article_id: Optional article ID to filter results
        
    Returns:
        List of matching segments with similarity scores
    """
    try:
        # Generate embedding for the query
        query_embedding = generate_embedding(query)
        
        # Build the search query
        search_query = supabase.table('article_segments')
        
        if article_id:
            search_query = search_query.eq('article_id', article_id)
        
        # Execute the vector search
        # Note: This uses pgvector's '<->' operator for cosine distance
        result = search_query.select('*, articles!inner(*)').\
            order('embedding <-> $1', query_embedding).\
            limit(limit).\
            execute()
        
        return result.data
    except Exception as e:
        print(f"Error in vector search: {str(e)}")
        return []

@app.route('/api/process_article', methods=['POST'])
def process_article():
    """
    Process an article to segment text and generate embeddings.
    
    Request JSON:
    {
        "url": "https://example.com/article",
        "title": "Article Title",
        "content": "Full article text...",
        "user_id": "optional-user-id"
    }
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data received'}), 400
        
        if 'content' not in data or 'url' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
        
        url = data['url']
        title = data.get('title', 'Untitled Article')
        content = data['content']
        user_id = data.get('user_id')
        
        # Segment the article text
        segments = segment_text(content)
        
        # Generate embeddings for each segment
        embeddings = [generate_embedding(segment) for segment in segments]
        
        # Store segments and embeddings
        success = store_article_embeddings(url, title, segments, embeddings, user_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Article processed successfully',
                'segments_count': len(segments)
            })
        else:
            return jsonify({'error': 'Failed to store article data'}), 500
            
    except Exception as e:
        print(f"Error processing article: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/vector_search', methods=['POST'])
def vector_search_endpoint():
    """
    Perform a vector search using the query.
    
    Request JSON:
    {
        "query": "Search query",
        "limit": 5,
        "article_id": "optional-article-id"
    }
    """
    try:
        data = request.json
        
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing query parameter'}), 400
        
        query = data['query']
        limit = data.get('limit', 5)
        article_id = data.get('article_id')
        
        results = vector_search(query, limit, article_id)
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        print(f"Error in vector search: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/vector_search_qa', methods=['POST'])
def vector_search_qa():
    """
    Perform a vector search and generate an answer using the retrieved context.
    
    Request JSON:
    {
        "query": "Question to answer",
        "limit": 5,
        "article_id": "optional-article-id"
    }
    """
    try:
        data = request.json
        
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing query parameter'}), 400
        
        query = data['query']
        limit = data.get('limit', 5)
        article_id = data.get('article_id')
        
        # Get relevant context from vector search
        search_results = vector_search(query, limit, article_id)
        
        if not search_results:
            return jsonify({
                'success': False,
                'answer': "I couldn't find any relevant information to answer your question."
            }), 404
        
        # Extract text from results
        context = "\n\n".join([result['segment_text'] for result in search_results])
        
        # Generate answer using Gemini API
        gemini_prompt = f"""
        I need to answer a question based on the provided context. 
        
        Context:
        {context}
        
        Question: {query}
        
        Provide a clear, concise answer based solely on the information in the context. If the context doesn't contain the answer, say "I don't have enough information to answer that question."
        """
        
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "contents": [{
                "parts": [{
                    "text": gemini_prompt
                }]
            }]
        }
        
        response = session.post(
            GEMINI_API_URL,
            headers=headers,
            json=data
        )
        
        response_json = response.json()
        
        if 'candidates' in response_json and len(response_json['candidates']) > 0:
            answer = response_json['candidates'][0]['content']['parts'][0]['text']
            return jsonify({
                'success': True,
                'answer': answer,
                'context': context
            })
        else:
            return jsonify({
                'success': False,
                'answer': "Failed to generate an answer.",
                'context': context
            }), 500
            
    except Exception as e:
        print(f"Error in vector search QA: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_article_recommendations', methods=['POST'])
def get_article_recommendations():
    """
    Get article recommendations based on similarity to a given article.
    
    Request JSON:
    {
        "article_id": "article-uuid",
        "limit": 5
    }
    """
    try:
        data = request.json
        
        if not data or 'article_id' not in data:
            return jsonify({'error': 'Missing article_id parameter'}), 400
        
        article_id = data['article_id']
        limit = data.get('limit', 5)
        
        # Get the article segments
        article_segments = supabase.table('article_segments')\
            .select('*')\
            .eq('article_id', article_id)\
            .execute()
        
        if not article_segments.data:
            return jsonify({'error': 'Article not found or has no segments'}), 404
        
        # Get a representative embedding (using the first segment as a simple approach)
        article_embedding = article_segments.data[0]['embedding']
        
        # Find similar articles based on vector similarity
        # This query finds articles that are not the current article but have similar content
        similar_articles = supabase.table('article_segments')\
            .select('articles!inner(id, url, title)')\
            .not_('article_id', 'eq', article_id)\
            .order('embedding <-> $1', article_embedding)\
            .limit(limit)\
            .execute()
        
        # Extract unique articles from the results
        seen_article_ids = set()
        recommendations = []
        
        for item in similar_articles.data:
            article = item['articles']
            if article['id'] not in seen_article_ids:
                seen_article_ids.add(article['id'])
                recommendations.append({
                    'id': article['id'],
                    'url': article['url'],
                    'title': article['title']
                })
                
                if len(recommendations) >= limit:
                    break
        
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
        
    except Exception as e:
        print(f"Error getting article recommendations: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({'status': 'ok', 'message': 'Server is running'}), 200

@app.route('/api/chat', methods=['POST'])
def chat():
    print("Received chat request")
    data = request.json
    
    if not data:
        print("No data received")
        return jsonify({'error': 'No data received'}), 400
    
    if 'message' not in data or 'articleContent' not in data:
        print("Missing required parameters")
        return jsonify({'error': 'Missing required parameters'}), 400
    
    message = data['message']
    article_content = data['articleContent']
    article_title = data.get('articleTitle', '')
    article_url = data.get('articleUrl', '')
    quiz_context = data.get('quizContext', '')
    
    print(f"Processing chat request: '{message}' for article '{article_title}'")
    
    # First, determine if the question can be answered from the article
    evaluation_prompt = f"""
    Article Title: {article_title}
    
    Article Content: 
    {article_content[:4000]}  # Limiting to 4000 chars to avoid token limits
    
    User Question: {message}
    
    Task: Determine if the user's question can be directly answered using information from the article.
    
    Instructions:
    1. Analyze the user's question and the article content.
    2. Determine if the article contains the necessary information to provide a complete and accurate answer.
    3. Respond with either "ANSWERABLE_FROM_ARTICLE" or "REQUIRES_GENERAL_KNOWLEDGE" followed by a brief explanation.
    
    Example responses:
    "ANSWERABLE_FROM_ARTICLE: The article directly discusses this topic in paragraph 3."
    "REQUIRES_GENERAL_KNOWLEDGE: The article doesn't cover this specific information about [topic]."
    """
    
    # Prepare request to Gemini API for evaluation
    evaluation_payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": evaluation_prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 256
        }
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print("Evaluating if question can be answered from article")
        evaluation_response = session.post(GEMINI_API_URL, headers=headers, data=json.dumps(evaluation_payload))
        evaluation_data = evaluation_response.json()
        
        if 'candidates' in evaluation_data and len(evaluation_data['candidates']) > 0:
            evaluation_result = evaluation_data['candidates'][0]['content']['parts'][0]['text']
            print(f"Evaluation result: {evaluation_result}")
            
            requires_general_knowledge = "REQUIRES_GENERAL_KNOWLEDGE" in evaluation_result
            
            # Create prompt based on evaluation result
            if quiz_context:
                base_prompt = f"""
                Article Title: {article_title}
                
                Article Content: 
                {article_content[:4000]}  # Limiting to 4000 chars to avoid token limits
                
                Quiz Context:
                {quiz_context}
                
                User Message: {message}
                
                You are an AI assistant helping a user understand an article and answering questions about a quiz based on the article.
                If the user is asking about the quiz, refer to the quiz context and provide helpful information.
                If the user is asking about the article content, focus on providing accurate information from the article.
                Keep your response concise and focused on answering the user's question.
                """
            elif requires_general_knowledge:
                base_prompt = f"""
                Article Title: {article_title}
                
                Article Content: 
                {article_content[:4000]}  # Limiting to 4000 chars to avoid token limits
                
                User Message: {message}
                
                You are an AI assistant helping a user understand an article and related topics.
                
                The user's question appears to be about a topic that is not directly covered in the article.
                You should:
                1. Acknowledge that the specific information isn't covered in the article
                2. Provide a helpful response based on your general knowledge
                3. If relevant, connect your answer back to the article's topic
                
                Keep your response concise and focused on answering the user's question.
                """
            else:
                base_prompt = f"""
                Article Title: {article_title}
                
                Article Content: 
                {article_content[:4000]}  # Limiting to 4000 chars to avoid token limits
                
                User Message: {message}
                
                You are an AI assistant helping a user understand an article. Respond to their message based on the article content.
                Keep your response concise and focused on answering the user's question.
                """
            
            # Prepare request to Gemini API for final response
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": base_prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.7,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 1024
                }
            }
            
            print("Sending request to Gemini API for final response")
            response = session.post(GEMINI_API_URL, headers=headers, data=json.dumps(payload))
            response_data = response.json()
            
            print(f"Received response from Gemini API: {response.status_code}")
            
            if 'candidates' in response_data and len(response_data['candidates']) > 0:
                generated_text = response_data['candidates'][0]['content']['parts'][0]['text']
                print("Successfully generated response")
                return jsonify({'response': generated_text, 'usedGeneralKnowledge': requires_general_knowledge})
            else:
                error_message = response_data.get('error', {}).get('message', 'Unknown error')
                print(f"Failed to generate response: {error_message}")
                return jsonify({'error': 'Failed to generate response', 'details': error_message}), 500
        else:
            error_message = evaluation_data.get('error', {}).get('message', 'Unknown error')
            print(f"Failed to evaluate question: {error_message}")
            return jsonify({'error': 'Failed to evaluate question', 'details': error_message}), 500
    
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze():
    print("Received analyze request")
    data = request.json
    
    if not data:
        print("No data received")
        return jsonify({'error': 'No data received'}), 400
    
    if 'articleContent' not in data:
        print("Missing articleContent parameter")
        return jsonify({'error': 'Missing article content'}), 400
    
    article_content = data['articleContent']
    article_title = data.get('articleTitle', '')
    
    print(f"Processing analyze request for article '{article_title}'")
    
    # Create prompt for article analysis
    prompt = f"""
    Article Title: {article_title}
    
    Article Content: 
    {article_content[:4000]}  # Limiting to 4000 chars to avoid token limits
    
    Please analyze this article and provide the following:
    1. A concise summary (3-5 sentences)
    2. 3-5 key points or main ideas
    3. 3-5 related topics that might be interesting to explore
    
    Format your response as JSON with the following structure:
    {{
        "summary": "...",
        "keyPoints": ["point1", "point2", ...],
        "relatedTopics": ["topic1", "topic2", ...]
    }}
    """
    
    # Log the request
    print(f"Image generation request received for article: {article_title}")
    
    # Call the Gemini API to generate an image
    # For now, we'll return a placeholder image URL since Gemini doesn't directly generate images
    # In a real implementation, you would use an image generation API like DALL-E or Midjourney
    
    # Generate a caption for the image
    caption_prompt = f"""
    Create a brief caption (1-2 sentences) for an image that represents the main theme of this article:
    
    Article Title: {article_title}
    
    Article Content: {article_content[:500]}...
    
    The caption should be concise and capture the essence of what the image would show.
    """
    
    caption_response = session.post(GEMINI_API_URL, headers={"Content-Type": "application/json"}, data=json.dumps({
        "contents": [
            {
                "parts": [
                    {
                        "text": caption_prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 1024
        }
    })).json()
    
    if 'candidates' in caption_response and len(caption_response['candidates']) > 0:
        caption = caption_response['candidates'][0]['content']['parts'][0]['text']
    else:
        caption = "Image based on article content"
    
    # For demonstration purposes, return a placeholder image URL
    # In a real implementation, you would generate an actual image or use a pre-generated one
    placeholder_image_url = "https://via.placeholder.com/800x400?text=Article+Visualization"
    
    return jsonify({
        'imageUrl': placeholder_image_url,
        'caption': caption
    })
    
@app.route('/api/explain', methods=['POST'])
def handle_explanation_request():
    try:
        data = request.json
        
        # Check if the request is using the new format with explanationContext
        if 'explanationContext' in data:
            explanation_context = data.get('explanationContext', {})
            selected_text = explanation_context.get('text', '')
            paragraph = explanation_context.get('paragraph', '')
            article_title = explanation_context.get('title', '') or data.get('articleTitle', '')
            mode = explanation_context.get('mode', 'simple').lower()
        else:
            # Fallback to direct parameters for backward compatibility
            selected_text = data.get('text', '')
            paragraph = data.get('paragraph', '')
            article_title = data.get('title', '')
            mode = data.get('mode', 'simple').lower()

        # Validate input
        if not selected_text:
            return jsonify({'error': 'No text provided for explanation'}), 400

        # Set instruction based on mode
        if mode == 'simple':
            instruction = "Provide a simple, concise explanation in plain language. Focus on clarity and brevity."
        elif mode == 'detailed':
            instruction = "Provide a detailed explanation with nuances and background information. Be comprehensive but clear."
        elif mode == 'examples':
            instruction = "Explain using concrete examples and analogies to illustrate the concept. Make it relatable."
        else:
            instruction = "Provide a simple, concise explanation in plain language."

        # Create prompt for Gemini
        base_prompt = f"""
        You are an AI assistant helping a user understand text from an article. The user has selected a specific text and wants an explanation.
        
        Article Title: {article_title}
        
        Selected Text: "{selected_text}"
        
        Surrounding Paragraph: "{paragraph}"
        
        Explanation Mode: {mode}
        
        {instruction}
        
        IMPORTANT INSTRUCTIONS:
        1. Respond directly with the explanation without prefacing it with phrases like "Here's a simple explanation" or "In simple terms".
        2. DO NOT repeat or quote the original selected text verbatim in your explanation.
        3. Provide a fresh explanation in your own words that helps the user understand the concept.
        4. Focus only on explaining the concept, not repeating what was already said.
        5. Keep your explanation concise and to the point.
        """

        # Add article content if available (limited to avoid token limits)
        article_content = data.get('articleContent', '')
        if article_content:
            # Truncate to avoid token limits
            truncated_content = article_content[:2000] + "..." if len(article_content) > 2000 else article_content
            base_prompt += f"\n\nAdditional article context: {truncated_content}"

        # Get response from Gemini
        response_text = ""
        max_retries = 2
        retries = 0
        
        while retries <= max_retries:
            try:
                response = session.post(GEMINI_API_URL, headers={"Content-Type": "application/json"}, data=json.dumps({
                    "contents": [
                        {
                            "parts": [
                                {
                                    "text": base_prompt
                                }
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.2,
                        "topP": 0.8,
                        "topK": 40,
                        "maxOutputTokens": 800,
                    }
                })).json()
                
                response_text = response['candidates'][0]['content']['parts'][0]['text']
                
                # Check if response is valid (not too short and not just repeating the input)
                words = response_text.split()
                if len(words) < 10:
                    retries += 1
                    print(f"Response too short, retrying ({retries}/{max_retries})")
                    continue
                    
                # Check if response is mostly just repeating the selected text
                if is_mostly_duplicate(response_text, selected_text):
                    retries += 1
                    print(f"Response too similar to input, retrying ({retries}/{max_retries})")
                    # Strengthen the instruction
                    base_prompt += "\n\nIMPORTANT: Your previous response was too similar to the original text. Please provide a completely fresh explanation without repeating the original text."
                    continue
                    
                # Post-process the response to remove any duplicated content
                response_text = post_process_explanation(response_text, selected_text)
                
                # If we got here, the response is valid
                break
                
            except Exception as e:
                print(f"Error generating explanation: {e}")
                retries += 1
                if retries > max_retries:
                    return jsonify({'error': 'Failed to generate explanation after multiple attempts'}), 500
        
        if not response_text or retries > max_retries:
            return jsonify({'error': 'Failed to generate a valid explanation'}), 500
            
        return jsonify({'response': response_text})
        
    except Exception as e:
        print(f"Error in explanation request: {e}")
        return jsonify({'error': str(e)}), 500

def is_mostly_duplicate(response, original_text):
    """Check if the response is mostly just repeating the original text"""
    # Convert to lowercase for comparison
    response_lower = response.lower()
    original_lower = original_text.lower()
    
    # If the original text is very short, be more lenient
    if len(original_lower) < 20:
        return False
        
    # Check if a significant portion of the original text appears verbatim in the response
    chunks = [original_lower[i:i+10] for i in range(0, len(original_lower)-10, 5)]
    matches = sum(1 for chunk in chunks if chunk in response_lower)
    
    # If more than 40% of chunks match, consider it too similar
    return matches > len(chunks) * 0.4

def post_process_explanation(response, original_text):
    """Process the explanation to remove duplicated content from the original text"""
    # If the response contains the entire original text verbatim, remove it
    if original_text in response:
        response = response.replace(original_text, "")
    
    # Check for large chunks of the original text (more than 10 words in sequence)
    original_words = original_text.split()
    if len(original_words) > 10:
        for i in range(len(original_words) - 10):
            chunk = " ".join(original_words[i:i+10])
            if chunk in response:
                response = response.replace(chunk, "")
    
    # Clean up any double spaces or newlines created by the removals
    response = " ".join(response.split())
    
    return response

@app.route('/api/research', methods=['POST'])
def research():
    print("Received research request")
    data = request.get_json()

    if not data or 'articleContent' not in data:
        return jsonify({'error': 'Missing article content'}), 400

    article_content = data['articleContent']
    article_title = data.get('articleTitle', '')
    research_query = data.get('researchQuery', '')

    try:
        # Create a query based on the article content, title, and user's research query
        query = (
            f"Research Query: {research_query}\n"
            f"Based on this article titled '{article_title}', "
            f"which is about: {article_content[:500]}...\n"
            "Find 5 articles that specifically address the research query while relating to the main article's content."
        )

        print(f"Sending request to Perplexity API for article: {article_title}")
        
        # Call Perplexity API with increased timeout
        response = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers=PERPLEXITY_HEADERS,
            json={
                "model": "sonar",
                "messages": [
                    {
                        "role": "system",
                        "content": """You are a research assistant. Your task is to find related articles.
                        IMPORTANT: You must respond with properly formatted JSON only, no other text.
                        Format: {"results": [{"title": "Article Title", "url": "https://article-url.com", "snippet": "Brief description"}]}
                        Rules:
                        - Return exactly 5 results
                        - Ensure URLs are real and accessible
                        - Keep snippets under 200 characters
                        - Articles must be closely related to the topic
                        - No markdown, just pure JSON"""
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            },
            timeout=30  # increased timeout
        )

        print(f"Perplexity API Status Code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Perplexity API error: {response.text}")
            return jsonify({'error': 'Failed to fetch research results'}), 500

        response_data = response.json()
        print(f"Raw API Response: {json.dumps(response_data, indent=2)}")
        
        if not response_data.get('choices') or not isinstance(response_data['choices'], list):
            print("Invalid response structure - missing or invalid choices")
            return jsonify({'error': 'Invalid API response structure'}), 500

        content = response_data['choices'][0].get('message', {}).get('content', '').strip()
        print(f"Content to parse: {content}")
        
        if not content:
            print("Empty content received from API")
            return jsonify({'error': 'Empty research results received'}), 500

        try:
            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # Parse the JSON string into a Python dictionary
            results = json.loads(content)
            
            # Validate results structure
            if not isinstance(results, dict) or 'results' not in results:
                print("Invalid results structure")
                return jsonify({'error': 'Invalid results format'}), 500
                
            print("Successfully parsed research results")
            return jsonify(results)
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"Failed content: {content}")
            return jsonify({'error': 'Failed to parse research results'}), 500

    except requests.Timeout:
        print("Perplexity API request timed out")
        return jsonify({'error': 'The research request timed out. Please try again.'}), 504
    except Exception as e:
        print(f"Exception in research endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/generate_summary', methods=['POST'])
def generate_summary():
    print("Received summary generation request")
    try:
        data = request.json
        
        if not data or 'content' not in data or 'options' not in data:
            return jsonify({'error': 'Missing content or options'}), 400
            
        article_content = data['content']
        options = data['options']
        
        # Create prompt based on selected options
        prompt_parts = ["Based on the following article content, provide:"]
        
        if options['bulletPoints']['enabled']:
            detail_level = "detailed" if options['bulletPoints']['detail'] == 'detailed' else "concise"
            expertise_level = "expert" if options['bulletPoints']['level'] == 'expert' else "beginner"
            prompt_parts.append(f"1. A {detail_level} bullet-point summary at {expertise_level} level")
        
        if options['definitions']:
            prompt_parts.append("2. Key definitions from the article (term: definition format)")
        
        if options['topics']:
            prompt_parts.append("3. Main topics covered in the article")
        
        if options['questions']:
            prompt_parts.append("4. Thought-provoking questions for reflection")
            
        prompt = f"""
        {' '.join(prompt_parts)}
        
        Article Content:
        {article_content[:4000]}
        
        Format your response as a JSON object with these possible fields:
        {{
            "bulletPoints": ["point1", "point2", ...],
            "definitions": [{{"term": "term1", "definition": "definition1"}}, ...],
            "topics": ["topic1", "topic2", ...],
            "questions": ["question1", "question2", ...]
        }}
        
        Only include the fields that were requested in the options.
        Ensure the response is properly formatted JSON.
        """
        
        # Call Gemini API
        response = session.post(
            GEMINI_API_URL,
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.3,
                    "topK": 40,
                    "topP": 0.95,
                    "maxOutputTokens": 1024
                }
            }
        )
        
        if response.status_code != 200:
            return jsonify({'error': 'Failed to generate summary'}), 500
            
        response_data = response.json()
        
        if 'candidates' in response_data and len(response_data['candidates']) > 0:
            generated_text = response_data['candidates'][0]['content']['parts'][0]['text']
            
            # Extract JSON from the response
            try:
                # Handle potential markdown code blocks
                if "```json" in generated_text:
                    json_str = generated_text.split("```json")[1].split("```")[0].strip()
                elif "```" in generated_text:
                    json_str = generated_text.split("```")[1].split("```")[0].strip()
                else:
                    json_str = generated_text
                    
                summary_data = json.loads(json_str)
                return jsonify(summary_data)
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}")
                return jsonify({'error': 'Failed to parse summary data'}), 500
        else:
            return jsonify({'error': 'No summary generated'}), 500
            
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(f"Starting server on port {PORT}...")
    app.run(debug=True, host='0.0.0.0', port=PORT)
