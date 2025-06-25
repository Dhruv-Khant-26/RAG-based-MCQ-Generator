import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.chains import RetrievalQA

load_dotenv()

def load_document_chunks(file_path):
    """
    Load and split PDF or DOCX files into chunks.
    Args:
        file_path (str): Path to the uploaded PDF or DOCX file.
    Returns:
        List of document chunks.
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_extension == ".docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF or DOCX file.")
    
    pages = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(pages)
    return docs

def create_faiss_vectorstore(docs):
    embeddings = AzureOpenAIEmbeddings(
        deployment=os.getenv("AZURE_OPENAI_TEXTEMBEDDING_DEPLOYMENT_NAME"),
        model="text-embedding-ada-002",
        openai_api_key=os.getenv("AZURE_OPENAI_TEXTEMBEDDING_API_KEY"),
        openai_api_base=os.getenv("AZURE_OPENAI_API_BASE"),
        openai_api_version="2025-01-01-preview",
        openai_api_type="azure"
    )
    db = FAISS.from_documents(docs, embeddings)
    return db

def get_llm():
    return AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        model_name="gpt-35-turbo",
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        openai_api_base=os.getenv("AZURE_OPENAI_API_BASE"),
        openai_api_version="2023-05-15-preview",
        openai_api_type="azure"
    )

def generate_questions(db):
    llm = get_llm()
    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    mcq_prompt = (
        "Generate 5 multiple-choice questions (MCQs) based on the following content. "
        "Each question should have 4 options and clearly mention the correct answer."
    )

    short_prompt = "Generate 5 short answer questions based on the content."

    mcq_response = qa_chain.run(mcq_prompt)
    short_response = qa_chain.run(short_prompt)

    return mcq_response, short_response