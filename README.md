# BetterReader Chrome Extension

A Chrome extension that enhances article reading by providing a sleek chat interface in a side panel, powered by Google's Gemini AI.

## Features

- **Side Panel Interface**: Access the extension via a sidebar in Chrome
- **AI-Powered Chat**: Ask questions about the current article and get intelligent responses from Gemini AI
- **Quiz & Chat**: Test your knowledge with AI-generated quiz questions and discuss them in the Learning tab

## How to Install

1. Clone or download this repository
2. Open Chrome and navigate to `chrome://extensions/`
3. Enable "Developer mode" in the top right
4. Click "Load unpacked" and select the folder containing this extension
5. The BetterReader extension will be added to your browser

## How to Use

1. Start the Python backend server:
   ```
   ./start_server.sh
   ```
2. Navigate to any article or text-based webpage
3. Click the BetterReader icon in your extensions toolbar
4. The side panel will open, showing the chat interface
5. Ask questions about the content or switch to the Learning tab for quiz and chat
6. In the Learning tab, you can:
   - Generate quiz questions about the article
   - Create custom quizzes by providing specific prompts
   - Answer multiple-choice questions and get immediate feedback
   - Chat with the AI about the quiz or article content

## Files

- `manifest.json`: Extension configuration
- `background.js`: Background service worker for extension functionality
- `content.js`: Script that runs in the context of web pages
- `sidepanel.html`: HTML structure for the side panel
- `sidepanel.js`: JavaScript for the side panel functionality
- `server.py`: Python backend server with Gemini AI integration
- `requirements.txt`: Python dependencies
- `start_server.sh`: Script to start the Python server
- `icon.png`: Extension icon

## Requirements

- Python 3.7+
- Flask
- Chrome browser
- Internet connection for Gemini API access

MIT
