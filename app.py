import streamlit as st
import tempfile
import os
from utils import load_document_chunks, create_faiss_vectorstore, generate_questions

st.set_page_config(page_title="RAG MCQ Generator", layout="wide")
st.title("RAG-based GenAI MCQ Generator")

mode = st.radio("Choose Input Mode:", ("Upload File", "Use Local File"))

file_path = None

if mode == "Upload File":
    uploaded_file = st.file_uploader("Upload your GenAI-related PDF or DOCX", type=["pdf", "docx"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            file_path = tmp.name

elif mode == "Use Local File":
    file_path = "/home/dhruv/Desktop/GenAI/project-1/Generative_AI_Intro.pdf"
    st.success(f"Using local file: {file_path.split('/')[-1]}")

if file_path:
    with st.spinner("Processing document and generating questions..."):
        try:
            docs = load_document_chunks(file_path)
            db = create_faiss_vectorstore(docs)
            mcq_text, short_text = generate_questions(db)

            st.subheader("MCQs (Multiple Choice Questions)")
            st.markdown(mcq_text)

            st.subheader("Short Answer Questions")
            st.markdown(short_text)
        except Exception as e:
            st.error(f"Something went wrong: {e}")