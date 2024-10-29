# Rapids Bot

# Rapids AI Chatbot

This project is a sophisticated chatbot developed for **Rapids AI**, an AI service provider. It utilizes **LangChain** combined with **RAG (Retrieval-Augmented Generation)** and **Google Gemini LLM** to offer intelligent and contextually relevant responses. Initially built with a **Streamlit** interface, the chatbot was later converted to a **Flask API** to facilitate integration with Rapids AI’s platforms and provide seamless service to users.

## Table of Contents
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)

## Features
- **Dynamic Conversational Responses**: Combines LangChain and Google Gemini LLM to generate informative responses.
- **Retrieval-Augmented Generation (RAG)**: Ensures accurate and up-to-date responses by retrieving relevant information from Rapids AI’s document store.
- **Streamlit Interface**: Initial testing and interaction interface.
- **Flask API**: Converts the chatbot into an API endpoint, allowing easy integration into Rapids AI’s services.
- **ChromaDB**: Stores and retrieves document data for faster, more accurate results.

## Tech Stack
- **LangChain**: Enables LLM chaining and document retrieval.
- **Google Gemini LLM**: Advanced language model for accurate responses.
- **RAG**: Combines retrieval and generation for enhanced accuracy.
- **Streamlit**: Frontend interface for initial testing.
- **Flask**: API for backend integration.
- **ChromaDB**: Document storage for efficient retrieval.

## Setup Instructions

### Prerequisites
1. **Python 3.8+**
2. **API keys**:
   - Google Gemini LLM API key
3. **Libraries**: Install required Python libraries.

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/HafizNouman666/RapidsAi-Chatbot.git
   cd RapidsAi-Chatbot
