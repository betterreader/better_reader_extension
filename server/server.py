from flask import Flask, request, jsonify
import requests
import json
import os
from flask_cors import CORS
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Enable CORS for all routes with more specific configuration
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Gemini API configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
PORT = int(os.getenv('PORT', 5007))

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
    
    # Create prompt for Gemini
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
    else:
        base_prompt = f"""
        Article Title: {article_title}
        
        Article Content: 
        {article_content[:4000]}  # Limiting to 4000 chars to avoid token limits
        
        User Message: {message}
        
        You are an AI assistant helping a user understand an article. Respond to their message based on the article content.
        Keep your response concise and focused on answering the user's question.
        """
    
    # Prepare request to Gemini API
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
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print("Sending request to Gemini API")
        response = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(payload))
        response_data = response.json()
        
        print(f"Received response from Gemini API: {response.status_code}")
        
        if 'candidates' in response_data and len(response_data['candidates']) > 0:
            generated_text = response_data['candidates'][0]['content']['parts'][0]['text']
            print("Successfully generated response")
            return jsonify({'response': generated_text})
        else:
            error_message = response_data.get('error', {}).get('message', 'Unknown error')
            print(f"Failed to generate response: {error_message}")
            return jsonify({'error': 'Failed to generate response', 'details': error_message}), 500
    
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
    
    # Prepare request to Gemini API
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 1024
        }
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print("Sending request to Gemini API")
        response = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(payload))
        response_data = response.json()
        
        print(f"Received response from Gemini API: {response.status_code}")
        
        if 'candidates' in response_data and len(response_data['candidates']) > 0:
            generated_text = response_data['candidates'][0]['content']['parts'][0]['text']
            
            # Extract JSON from the response
            try:
                # Find JSON in the response (it might be wrapped in markdown code blocks)
                json_str = generated_text
                if "```json" in generated_text:
                    json_str = generated_text.split("```json")[1].split("```")[0].strip()
                elif "```" in generated_text:
                    json_str = generated_text.split("```")[1].split("```")[0].strip()
                
                analysis_data = json.loads(json_str)
                print("Successfully parsed JSON response")
                return jsonify(analysis_data)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}")
                # If JSON parsing fails, return the raw text
                return jsonify({
                    'summary': generated_text[:500],
                    'keyPoints': ["Unable to parse structured data"],
                    'relatedTopics': ["Unable to parse structured data"]
                })
        else:
            error_message = response_data.get('error', {}).get('message', 'Unknown error')
            print(f"Failed to generate response: {error_message}")
            return jsonify({'error': 'Failed to generate response', 'details': error_message}), 500
    
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-quiz', methods=['POST'])
def generate_quiz():
    print("Received quiz generation request")
    try:
        data = request.json
        
        if not data:
            print("No data received")
            return jsonify({'error': 'No data received'}), 400
        
        print(f"Request data: {data.keys()}")
        
        if 'articleContent' not in data:
            print("Missing articleContent parameter")
            return jsonify({'error': 'Missing article content'}), 400
        
        article_content = data['articleContent']
        article_title = data.get('articleTitle', '')
        custom_prompt = data.get('customPrompt', '')
        client_timestamp = data.get('timestamp', '')
        
        # Add timestamp to encourage different questions each time
        import datetime
        import random
        import uuid
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        random_seed = random.randint(1, 1000000)
        unique_id = str(uuid.uuid4())
        
        print(f"Processing quiz generation request for article '{article_title}'")
        print(f"Article content length: {len(article_content)}")
        print(f"Custom prompt: {custom_prompt}")
        print(f"Client timestamp: {client_timestamp}")
        print(f"Request time: {current_time}")
        print(f"Random seed: {random_seed}")
        print(f"Unique ID: {unique_id}")
        
        # Create prompt for quiz generation
        if custom_prompt:
            base_prompt = f"""
            You are an educational assistant that creates tailored quiz questions to help users test their understanding of articles they're reading.

            ARTICLE CONTENT:
            {article_content[:4000]}

            USER REQUEST:
            {custom_prompt}

            CURRENT TIME: {current_time}
            UNIQUE ID: {unique_id}
            CLIENT TIMESTAMP: {client_timestamp}

            TASK:
            Create personalized multiple-choice quiz questions based on the article content that match the user's specific request.

            INSTRUCTIONS:
            1. Analyze the user's request to understand what types of questions they want (e.g., about specific topics, concepts, or sections of the article).
            2. Generate 3-5 multiple-choice questions that align with the user's request while covering important content from the article.
            3. If the user hasn't specified question types, focus on the most important concepts and key takeaways.
            4. Each question should have 4 answer options with exactly one correct answer.
            5. Ensure questions are directly answerable from the article content.
            6. Do not create questions about information not present in the article.
            7. IMPORTANT: Generate unique and diverse questions each time. Do not repeat questions from previous requests.
            8. Format your response as a valid JSON object with the following structure:

            {{
              "questions": [
                {{
                  "question": "Question text goes here?",
                  "options": [
                    "Option A",
                    "Option B",
                    "Option C",
                    "Option D"
                  ],
                  "correctAnswer": 0,
                  "explanation": "Brief explanation of why this answer is correct"
                }},
                ...
              ]
            }}

            The "correctAnswer" field should be the zero-based index of the correct option (0 for first option, 1 for second, etc.).
            Ensure your response is properly formatted JSON that can be parsed by JavaScript's JSON.parse() function.

            If the user's request cannot be fulfilled based on the article content, respond with a friendly message explaining why and offer to generate general questions about the article instead.
            """
        else:
            base_prompt = f"""
            You are an educational assistant that creates tailored quiz questions to help users test their understanding of articles they're reading.

            ARTICLE CONTENT:
            {article_content[:4000]}

            USER REQUEST:
            Generate quiz questions about the main concepts and key points from this article.

            CURRENT TIME: {current_time}
            UNIQUE ID: {unique_id}
            CLIENT TIMESTAMP: {client_timestamp}

            TASK:
            Create personalized multiple-choice quiz questions based on the article content that match the user's specific request.

            INSTRUCTIONS:
            1. Analyze the user's request to understand what types of questions they want (e.g., about specific topics, concepts, or sections of the article).
            2. Generate 3-5 multiple-choice questions that align with the user's request while covering important content from the article.
            3. If the user hasn't specified question types, focus on the most important concepts and key takeaways.
            4. Each question should have 4 answer options with exactly one correct answer.
            5. Ensure questions are directly answerable from the article content.
            6. Do not create questions about information not present in the article.
            7. IMPORTANT: Generate unique and diverse questions each time. Do not repeat questions from previous requests.
            8. Format your response as a valid JSON object with the following structure:

            {{
              "questions": [
                {{
                  "question": "Question text goes here?",
                  "options": [
                    "Option A",
                    "Option B",
                    "Option C",
                    "Option D"
                  ],
                  "correctAnswer": 0,
                  "explanation": "Brief explanation of why this answer is correct"
                }},
                ...
              ]
            }}

            The "correctAnswer" field should be the zero-based index of the correct option (0 for first option, 1 for second, etc.).
            Ensure your response is properly formatted JSON that can be parsed by JavaScript's JSON.parse() function.

            If the user's request cannot be fulfilled based on the article content, respond with a friendly message explaining why and offer to generate general questions about the article instead.
            """
        
        # Prepare request to Gemini API
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
                "temperature": 0.9,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 1024,
                "seed": random_seed
            }
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            print("Sending request to Gemini API")
            response = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(payload))
            response_data = response.json()
            
            print(f"Received response from Gemini API: {response.status_code}")
            
            if 'candidates' in response_data and len(response_data['candidates']) > 0:
                generated_text = response_data['candidates'][0]['content']['parts'][0]['text']
                
                # Extract JSON from the response
                try:
                    # Find JSON in the response (it might be wrapped in markdown code blocks)
                    json_str = generated_text
                    if "```json" in generated_text:
                        json_str = generated_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in generated_text:
                        json_str = generated_text.split("```")[1].split("```")[0].strip()
                    
                    quiz_data = json.loads(json_str)
                    print("Successfully parsed JSON response")
                    return jsonify(quiz_data)
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {str(e)}")
                    # If JSON parsing fails, return an error
                    return jsonify({
                        'error': 'Failed to parse quiz data',
                        'rawResponse': generated_text
                    }), 500
            else:
                error_message = response_data.get('error', {}).get('message', 'Unknown error')
                print(f"Failed to generate response: {error_message}")
                return jsonify({'error': 'Failed to generate quiz', 'details': error_message}), 500
        
        except Exception as e:
            print(f"Exception occurred in generate_quiz: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500
    
    except Exception as e:
        print(f"Exception occurred in generate_quiz: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate-image', methods=['POST'])
def generate_image():
    data = request.json
    if 'articleContent' not in data:
        return jsonify({'error': 'Missing article content'}), 400
    
    article_content = data.get('articleContent', '')
    article_title = data.get('articleTitle', '')
    article_url = data.get('articleUrl', '')
    timestamp = data.get('timestamp', str(time.time()))
    
    try:
        # Create a prompt for image generation
        prompt = f"""
        Generate an image that visually represents the key concepts or themes from this article.
        
        Article Title: {article_title}
        
        Article Content: {article_content[:1000]}...
        
        Create an image that captures the essence of this article in a visually appealing way.
        The image should be informative, relevant to the content, and help enhance understanding.
        
        Timestamp: {timestamp}
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
        
        caption_response = requests.post(GEMINI_API_URL, headers={"Content-Type": "application/json"}, data=json.dumps({
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
        
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(f"Starting server on port {PORT}...")
    app.run(debug=True, host='0.0.0.0', port=PORT)
