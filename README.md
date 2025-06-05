# Youtube_Video_Summarizer

This is a FastAPI-based web service that generates detailed summaries of YouTube videos using the Google Gemini 2.0 Flash model. It leverages the YouTube Transcript API to fetch video captions, splits long transcripts efficiently, creates vector embeddings, and retrieves relevant context with FAISS before summarizing.

# Features
Fetch YouTube video transcripts automatically via YouTube Transcript API

Chunk large transcripts into manageable pieces with RecursiveCharacterTextSplitter

Generate semantic embeddings of transcript chunks using GoogleGenerativeAIEmbeddings

Efficient similarity search over embeddings with FAISS vector store

Summarize videos in detail using the Google Gemini 2.0 Flash large language model

Easy-to-use REST API with FastAPI

# Tech Stack
FastAPI - lightweight async API framework

YouTube Transcript API - fetch captions from YouTube videos

Google Gemini 2.0 Flash - latest generative AI model for text summarization

# LangChain components:

ChatGoogleGenerativeAI for model inference

GoogleGenerativeAIEmbeddings for creating embeddings

RecursiveCharacterTextSplitter for text chunking

FAISS for efficient vector similarity search

# Installation
Clone the repo

Create and activate a Python virtual environment

Install dependencies

Set up your environment variables (e.g. Google API credentials) in a .env file.



