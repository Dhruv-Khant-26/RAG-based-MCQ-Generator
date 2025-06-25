# RAG-based-MCQ-Generator

Overview-<br>
This application is a Retrieval-Augmented Generation (RAG) system designed to generate educational content from uploaded PDF documents related to Generative AI (GenAI). It extracts content from a PDF, chunks it, stores the chunks in a FAISS vector database, and uses LangChain with Azure OpenAI to generate five multiple-choice questions (MCQs) with four options and answers, as well as five short-answer questions. The output is displayed via a user-friendly Streamlit web interface.<br><br>

Tech Stack-<br>
LangChain: Framework for integrating LLMs and managing RAG pipelines.<br>
FAISS: Vector store for efficient similarity search of document embeddings.<br>
Azure OpenAI: Provides embeddings and LLM for question generation.<br>
Streamlit: Python library for creating interactive web applications.<br>
PyPDFLoader: LangChain's document loader for PDF content extraction.<br><br>

Prerequisites-<br>
Python 3.8 or higher<br>
Azure OpenAI account with access to embeddings and LLM models (e.g., text-embedding-ada-002, gpt-3.5-turbo)<br>
Azure OpenAI API key, endpoint, and deployment names<br>
A Generative AI-related PDF document for input<br>
