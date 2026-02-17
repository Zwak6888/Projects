# Projects Persona Memory – AI Chat System with Persistent Memory

Persona Memory is a full-stack AI chat application that I built to implement long-term conversational memory using vector retrieval techniques.The system allows users to log in, interact with an AI assistant, and automatically store conversation history in a structured database while simultaneously embedding important information into a FAISS vector index for semantic retrieval. Unlike basic stateless chatbots, this system retrieves relevant past interactions and incorporates them into future responses, enabling context-aware conversations.

What I Implemented
1. Built FastAPI backend with modular architecture
2. Designed service layer (chat_service, memory_service, retrieval, prompt_builder)
3. Integrated FAISS for semantic memory search
4. Implemented SQLAlchemy-based database storage
5. Developed JWT authentication system
6. Built frontend interface (login + dashboard + chat UI)
7. Structured prompt construction for contextual AI responses
8. Enabled persistent memory retrieval across sessions

Architecture Overview
User → Frontend → FastAPI API
. Store conversation in database
. Generate embeddings
. Store in FAISS index
. Retrieve relevant memory
. Build contextual prompt
. Generate response

Technologies Used
Backend: Python
FastAPI
SQLAlchemy
FAISS
Pydantic
JWT Authentication

Frontend: HTML
CSS
JavaScript

Purpose of the Project:
The goal was to build an AI system that maintains contextual continuity rather than responding independently to each query. This project demonstrates how retrieval-based memory systems can enhance conversational AI by integrating semantic search with structured storage.
