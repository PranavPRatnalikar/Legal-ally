import streamlit as st
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

st.set_page_config(page_title="Document Genie", layout="wide")

# API Key Handling - Get from user input or environment variable
api_key = st.sidebar.text_input("Enter your Google API Key:", type="password")
if not api_key:
    st.warning("Please enter a valid API key.")

def get_pdf_text(pdf_docs):
    """Extracts text from PDFs using pdfplumber."""
    text = ""
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                text += page.extract_text() or ""  # Handle NoneType
    return text.strip()  # Remove extra spaces

def summarize_text(text, api_key):
    """Generates a brief summary of the document using Gemini-Pro."""
    if not text:
        return "Unable to extract meaningful text from the document."
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = f"Summarize the following document in a few sentences:\n\n{text[:5000]}"  # Limit input to first 5000 chars
    response = model.invoke(prompt)
    return response.content if response else "Summary generation failed."

def get_text_chunks(text):
    """Splits extracted text into smaller chunks for better processing."""
    if not text:
        st.error("No text extracted from PDFs. Please check your files.")
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks, api_key):
    """Embeds text chunks and stores them in FAISS for retrieval."""
    if not text_chunks:
        st.error("No valid text chunks to process.")
        return
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(api_key):
    """Creates a conversation chain with a custom prompt."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not available, reply: 'Answer is not available in the context.' 
    Do not make up an answer.

    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question, api_key):
    """Handles user queries by searching the vector store and generating responses."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain(api_key)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply:", response["output_text"])
    except Exception as e:
        st.error(f"Error processing request: {e}")

def main():
    """Main Streamlit app logic."""
    st.header("📄 Document Chatbot")
    
    user_question = st.text_input("🔍 Ask a question from the PDF files:", key="user_question")
    
    if user_question and api_key:
        user_input(user_question, api_key)
    
    with st.sidebar:
        st.title("📌 Menu")
        pdf_docs = st.file_uploader("📂 Upload your PDF files:", accept_multiple_files=True, key="pdf_uploader")
        
        if pdf_docs:
            raw_text = get_pdf_text(pdf_docs)
            summary = summarize_text(raw_text, api_key)
            st.markdown("### 📜 Document Summary:")
            st.info(summary)

        if st.button("🚀 Submit & Process") and api_key:
            with st.spinner("🔄 Processing..."):
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.success("✅ Processing complete!")

if __name__ == "__main__":
    main()
