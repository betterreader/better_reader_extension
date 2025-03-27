
# BetterReader

**BetterReader** is an open-source browser extension and website that serves as your AI copilot for reading, research, and learning across the web. Whether you're diving into technical papers, news articles, or blog posts, BetterReader helps you understand content more deeply through conversation, quizzes, and interactive tools.

BetterReader supports both online usage (via browser extension stores) and local usage by allowing users to configure their own LLM API keys.

> [!IMPORTANT]
> **BetterReader is currently in early development.**  
> It is not yet available on browser extension stores. However, you can set up and run the extension locally using this repository. It currently supports Chrome-based browsers.


## Features

- **Chat:** Engage in a conversation with the extension about the current article. BetterRead can help answer questions and give examples, counterarguments, and clarifications. Additionally, BetterReader remembers past articles you have read, helping you get a comprehensive understanding on any topic.   
- **Teacher Mode**:   A chat mode that uses the Socratic method to guide you through thought-provoking questions that deepen understanding and encourage critical thinking.
- **Summary:** Generate a concise summary of any article, including key definitions, key topics, and exploration questions to guide your learning.
- **Quiz Mode:** Test your understanding with custom-generated quizzes based on the content tailored to your experience with the topic.
- **Highlights & Annotations:** Highlight and annotate article text in different colors to revisit and organize key points.
- **Research Assistant:** Instantly search for related articles and sources to broaden your perspective and deepen your research.
- **ELI5:** Highlight confusing sections of text and click the ELI5 (Explain Like I'm 5) button to receive simple, accessible explanations.


## Why BetterReader?

Why choose BetterReader over other AI reading assistants?

- **🎓 A Learning-Focused Experience:**  
  Unlike other AI tools that focus on quick summaries, BetterReader is built to enhance comprehension and long-term learning. It's designed for students, researchers, and curious minds who want to *understand*, not just *skim*.
  
- **🧰 All-in-One Toolkit:**  
  BetterReader combines summarization, chat, annotation, quiz generation, and research discovery into a single seamless interface—no need to juggle multiple extensions or apps.
  
- **🌍 Open Source:**  
  Transparent, customizable, and community-driven. BetterReader is open-source under the MIT license, so you can modify, contribute, or self-host it however you like.

- **💻 Run Locally:**  
  Don't want to rely on third-party APIs or share your data? BetterReader allows you to bring your own API keys (e.g., Google Gemini) and run everything locally for full control over privacy and costs. 

---

## Future Development

Here's what's coming soon:

- **📄 PDF Support:**  
  Read, annotate, and analyze PDFs just like you would any webpage.

- **🧠 Connect to Local Models:**  
  Support for integrating with local LLMs (e.g., via Ollama, LM Studio, or open-source models) for a fully offline, private experience.

- **☁️ Online Hosting & User Accounts:**  
  Host BetterReader online with user accounts, saving highlights, chat history, and personalized learning journeys across devices.

## Tech Stack

- Extension: Typescript + React.js + Tailwind
- Website: Typescript + Next.js + Tailwind
- Server: Python + Flask
- Database + Auth: Supabase
- Browser Extension Boilerplate: 

## Contributing

Contributions are welcome! While we are still in an early stage, if you're interested in helping us build BetterReader, please contact us and we can help you get started. 


## License  
BetterReader is released under the Apache 2.0 license. 
