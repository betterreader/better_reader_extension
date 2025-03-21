// Constants
const API_BASE_URL = 'http://localhost:5007';

// State management
let state = {
  articleContent: '',
  articleTitle: '',
  articleUrl: '',
  currentQuizData: null,
};

document.addEventListener('DOMContentLoaded', () => {
  // Initialize UI components
  initializeTabs();
  initializeChatInterface();
  initializeQuizInterface();

  // Fetch article content
  fetchArticleContent();
});

// =====================
// Tab Management
// =====================
function initializeTabs() {
  const tabs = document.querySelectorAll('.tab');
  const tabContents = document.querySelectorAll('.tab-content');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      // Remove active class from all tabs and tab contents
      tabs.forEach(t => t.classList.remove('active'));
      tabContents.forEach(content => content.classList.remove('active'));

      // Add active class to clicked tab and corresponding content
      tab.classList.add('active');
      const tabName = tab.getAttribute('data-tab');
      document.getElementById(`${tabName}-tab`).classList.add('active');
    });
  });
}

// =====================
// Article Content
// =====================
function fetchArticleContent() {
  chrome.runtime.sendMessage({ action: 'getArticleContent' }, response => {
    console.log('Response from getArticleContent:', response);

    if (response && response.content) {
      state.articleContent = response.content;
      state.articleTitle = response.title;
      state.articleUrl = response.url;

      console.log('Article content received:', {
        title: state.articleTitle,
        url: state.articleUrl,
        contentLength: state.articleContent.length,
      });
    } else {
      console.error('Failed to get article content:', response?.error || 'Unknown error');

      // Show error message in chat tab
      const chatMessages = document.getElementById('chat-messages');
      chatMessages.innerHTML += `
        <div class="message bot-message">
          I couldn't extract the article content. Please make sure you're on a page with readable content.
        </div>
      `;
    }
  });
}

// =====================
// Chat Interface
// =====================
function initializeChatInterface() {
  const chatMessages = document.getElementById('chat-messages');
  const messageInput = document.getElementById('message-input');
  const sendButton = document.getElementById('send-button');

  // Add event listeners for sending messages
  sendButton.addEventListener('click', () => sendChatMessage(chatMessages, messageInput));
  messageInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendChatMessage(chatMessages, messageInput);
    }
  });
}

function sendChatMessage(chatMessages, messageInput) {
  const messageText = messageInput.value.trim();
  if (messageText === '') return;

  // Add user message to the chat
  appendMessage(chatMessages, messageText, 'user');

  // Clear input
  messageInput.value = '';

  // Show typing indicator
  const thinkingElement = document.createElement('div');
  thinkingElement.classList.add('message', 'bot-message', 'typing-indicator');
  thinkingElement.textContent = '...';
  chatMessages.appendChild(thinkingElement);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  // Process the message with the Gemini API
  processUserMessage(messageText, thinkingElement, chatMessages);
}

function appendMessage(container, text, sender) {
  const messageElement = document.createElement('div');
  messageElement.classList.add('message');
  messageElement.classList.add(sender === 'user' ? 'user-message' : 'bot-message');
  messageElement.textContent = text;

  container.appendChild(messageElement);
  container.scrollTop = container.scrollHeight;
}

function processUserMessage(message, thinkingElement, chatMessages) {
  console.log('Processing message:', message);

  if (!state.articleContent || state.articleContent.length === 0) {
    // Remove thinking indicator
    if (thinkingElement && thinkingElement.parentNode) {
      chatMessages.removeChild(thinkingElement);
    }

    appendMessage(
      chatMessages,
      "Sorry, I couldn't extract any content from this page. Please try on a different article.",
      'bot',
    );
    return;
  }

  const requestData = {
    message: message,
    articleContent: state.articleContent,
    articleTitle: state.articleTitle,
    articleUrl: state.articleUrl,
  };

  console.log('Sending request to API:', `${API_BASE_URL}/api/chat`);

  fetch(`${API_BASE_URL}/api/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestData),
  })
    .then(response => {
      console.log('API response status:', response.status);
      if (!response.ok) {
        throw new Error(`Network response was not ok: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      console.log('API response data:', data);

      // Remove thinking indicator
      if (thinkingElement && thinkingElement.parentNode) {
        chatMessages.removeChild(thinkingElement);
      }

      // Add bot response
      if (data.response) {
        appendMessage(chatMessages, data.response, 'bot');
      } else if (data.error) {
        appendMessage(chatMessages, `Error: ${data.error}`, 'bot');
      }
    })
    .catch(error => {
      console.error('Error:', error);

      // Remove thinking indicator
      if (thinkingElement && thinkingElement.parentNode) {
        chatMessages.removeChild(thinkingElement);
      }

      // Add error message
      appendMessage(
        chatMessages,
        'Sorry, I encountered an error while processing your request. Please try again later.',
        'bot',
      );
    });
}

// =====================
// Quiz Interface
// =====================
function initializeQuizInterface() {
  const generateQuizBtn = document.getElementById('generate-quiz-btn');
  const customQuizBtn = document.getElementById('custom-quiz-btn');
  const customQuizPrompt = document.getElementById('custom-quiz-prompt');
  const quizPromptInput = document.getElementById('quiz-prompt-input');
  const submitPromptBtn = document.getElementById('submit-prompt-btn');

  // Generate quiz with default prompt
  generateQuizBtn.addEventListener('click', () => {
    generateQuiz();
  });

  // Toggle custom quiz prompt
  customQuizBtn.addEventListener('click', () => {
    customQuizPrompt.classList.toggle('active');
    if (customQuizPrompt.classList.contains('active')) {
      quizPromptInput.focus();
    }
  });

  // Generate quiz with custom prompt
  submitPromptBtn.addEventListener('click', () => {
    const customPrompt = quizPromptInput.value.trim();
    if (customPrompt) {
      generateQuiz(customPrompt);
      customQuizPrompt.classList.remove('active');
      quizPromptInput.value = '';
    }
  });

  // Submit custom prompt on Enter key
  quizPromptInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      e.preventDefault();
      submitPromptBtn.click();
    }
  });
}

function generateQuiz(customPrompt = '') {
  const quizChatMessages = document.getElementById('quiz-chat-messages');
  const quizChatContainer = document.getElementById('quiz-chat-container');

  // Add loading message to chat
  const loadingElement = document.createElement('div');
  loadingElement.classList.add('message', 'bot-message', 'typing-indicator');
  loadingElement.textContent = 'Generating quiz questions...';
  quizChatMessages.appendChild(loadingElement);
  quizChatContainer.scrollTop = quizChatContainer.scrollHeight;

  if (!state.articleContent || state.articleContent.length === 0) {
    loadingElement.classList.remove('typing-indicator');
    loadingElement.textContent = "I couldn't extract any content from this page. Please try on a different article.";
    return;
  }

  console.log('Generating quiz questions...');

  // Add a timestamp to ensure each request is unique
  const timestamp = new Date().getTime();

  const requestData = {
    articleContent: state.articleContent,
    articleTitle: state.articleTitle,
    articleUrl: state.articleUrl,
    timestamp: timestamp,
  };

  if (customPrompt) {
    requestData.customPrompt = customPrompt;
  }

  fetch(`${API_BASE_URL}/api/generate-quiz`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestData),
  })
    .then(handleApiResponse)
    .then(data => {
      // Remove loading message
      loadingElement.remove();

      if (data.error) {
        handleQuizError(data.error, quizChatMessages, quizChatContainer);
        return;
      }

      // Store quiz data for later reference
      state.currentQuizData = data;

      // Display quiz
      displayQuiz(data, customPrompt, quizChatMessages, quizChatContainer);
    })
    .catch(error => {
      console.error('Error generating quiz:', error);

      // Remove loading message
      loadingElement.remove();

      handleQuizError(error.message, quizChatMessages, quizChatContainer);
    });
}

function handleApiResponse(response) {
  console.log('API response status:', response.status);
  if (!response.ok) {
    throw new Error(`Network response was not ok: ${response.status}`);
  }
  return response.json();
}

function handleQuizError(errorMessage, quizChatMessages, quizChatContainer) {
  const errorElement = document.createElement('div');
  errorElement.classList.add('message', 'bot-message');
  errorElement.textContent = `Error: ${errorMessage}`;
  quizChatMessages.appendChild(errorElement);
  quizChatContainer.scrollTop = quizChatContainer.scrollHeight;
}

function displayQuiz(quizData, customPrompt, quizChatMessages, quizChatContainer) {
  // Add intro message for quiz
  const introElement = document.createElement('div');
  introElement.classList.add('message', 'bot-message');
  introElement.textContent = customPrompt
    ? `Here are some quiz questions based on your request: "${customPrompt}"`
    : 'Here are some quiz questions based on the article:';
  quizChatMessages.appendChild(introElement);
  quizChatContainer.scrollTop = quizChatContainer.scrollHeight;

  // Render quiz questions
  if (quizData.questions && quizData.questions.length > 0) {
    renderQuizQuestions(quizData.questions, quizChatMessages, quizChatContainer);
  } else {
    const noQuestionsElement = document.createElement('div');
    noQuestionsElement.classList.add('message', 'bot-message');
    noQuestionsElement.textContent =
      "I couldn't generate any quiz questions from this article. Please try a different article or prompt.";
    quizChatMessages.appendChild(noQuestionsElement);
    quizChatContainer.scrollTop = quizChatContainer.scrollHeight;
  }
}

function renderQuizQuestions(questions, quizChatMessages, quizChatContainer) {
  questions.forEach((question, questionIndex) => {
    const questionElement = document.createElement('div');
    questionElement.classList.add('quiz-question');
    questionElement.dataset.questionIndex = questionIndex;

    // Create question text
    const questionText = document.createElement('div');
    questionText.classList.add('quiz-question-text');
    questionText.textContent = `${questionIndex + 1}. ${question.question}`;
    questionElement.appendChild(questionText);

    // Create options
    const optionsContainer = document.createElement('div');
    optionsContainer.classList.add('quiz-options');

    question.options.forEach((option, optionIndex) => {
      const optionElement = document.createElement('div');
      optionElement.classList.add('quiz-option');
      optionElement.dataset.optionIndex = optionIndex;
      optionElement.textContent = option;

      // Add click event to handle option selection
      optionElement.addEventListener('click', () => {
        handleOptionSelection(questionElement, optionElement, question, optionIndex);
      });

      optionsContainer.appendChild(optionElement);
    });

    questionElement.appendChild(optionsContainer);

    // Create explanation element (hidden initially)
    const explanationElement = document.createElement('div');
    explanationElement.classList.add('quiz-explanation');
    explanationElement.style.display = 'none';

    if (question.explanation) {
      explanationElement.textContent = question.explanation;
    } else {
      explanationElement.textContent = 'No explanation provided.';
    }

    questionElement.appendChild(explanationElement);

    quizChatMessages.appendChild(questionElement);
  });

  quizChatContainer.scrollTop = quizChatContainer.scrollHeight;
}

function handleOptionSelection(questionElement, optionElement, question, selectedIndex) {
  // Get all options in this question
  const options = questionElement.querySelectorAll('.quiz-option');

  // Check if question is already answered
  if (questionElement.classList.contains('answered')) {
    return;
  }

  // Mark question as answered
  questionElement.classList.add('answered');

  // Get correct answer index
  const correctIndex = question.answer;

  // Mark selected option
  optionElement.classList.add('selected');

  // Mark correct and incorrect options
  options.forEach((option, index) => {
    if (index === correctIndex) {
      option.classList.add('correct');
    } else if (index === selectedIndex && selectedIndex !== correctIndex) {
      option.classList.add('incorrect');
    }
  });

  // Show explanation
  const explanationElement = questionElement.querySelector('.quiz-explanation');
  if (explanationElement) {
    explanationElement.style.display = 'block';
  }
}

// Helper function to format lists
function formatList(items) {
  return items.map(item => `• ${item}`).join('\n');
}
