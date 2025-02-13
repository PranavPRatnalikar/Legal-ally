import streamlit as st
import pdfplumber
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="Legal AI Assistant", layout="wide")

# API Key Input
api_key = st.sidebar.text_input("Enter your Google API Key:", type="password")
if not api_key:
    st.warning("Please enter a valid API key.")

# Directory for pre-stored legal acts
LEGAL_DATASET_DIR = "dataset"

def get_pdf_text(pdf_docs):
    """Extracts text from PDFs using pdfplumber."""
    text = ""
    for pdf in pdf_docs:
        with pdfplumber.open(pdf) as pdf_reader:
            for page in pdf_reader.pages:
                text += page.extract_text() or ""  # Handle NoneType
    return text.strip()

def summarize_text(text, api_key):
    """Generates a brief summary of the document using Gemini-Pro."""
    if not text:
        return "Unable to extract meaningful text from the document."
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = f"Summarize the following legal document in a few sentences:\n\n{text[:5000]}"  
    response = model.invoke(prompt)
    return response.content if response else "Summary generation failed."

def get_text_chunks(text):
    """Splits text into chunks for processing."""
    if not text:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def store_legal_embeddings(api_key):
    """Preprocesses and stores legal act embeddings from the dataset folder."""
    legal_text = ""
    
    for file in os.listdir(LEGAL_DATASET_DIR):
        if file.endswith(".pdf"):
            file_path = os.path.join(LEGAL_DATASET_DIR, file)
            legal_text += get_pdf_text([file_path]) + "\n\n"
    
    if legal_text:
        text_chunks = get_text_chunks(legal_text)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("legal_faiss_index")
        return True
    return False

def get_vector_store(text_chunks, store_name, api_key):
    """Embeds text chunks and stores them in FAISS."""
    if not text_chunks:
        return
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(store_name)

def get_conversational_chain(api_key, advisor_mode=False):
    """Creates a chatbot chain with a legal or general prompt."""
    if advisor_mode:
        prompt_template = """
        You are a legal advisor. Answer based on the given legal acts and laws.
        If the law is unclear, say 'The law does not explicitly cover this scenario.'
        
        Legal Context:\n {context}\n
        Question: \n{question}\n
        
        Answer:
        """
    else:
        prompt_template = """
        Answer questions based on the provided document. 
        If the answer is not in the context, say 'Not available in the provided document.'
        
        Context:\n {context}\n
        Question: \n{question}\n
        
        Answer:
        """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question, api_key, advisor_mode=False):
    """Handles user queries for document chatbot or legal advisor."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    try:
        store_name = "legal_faiss_index" if advisor_mode else "faiss_index"
        new_db = FAISS.load_local(store_name, embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain(api_key, advisor_mode)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply:", response["output_text"])
    except Exception as e:
        st.error(f"Error processing request: {e}")

def main():
    """Main Streamlit UI logic with two modes."""
    st.header("⚖️ Legal AI Assistant")
    
    mode = st.radio("Choose Mode:", ["Legal Document Chatbot", "Legal Advisor"])
    
    user_question = st.text_input("🔍 Ask a legal question:", key="user_question")
    
    if user_question and api_key:
        if mode == "Legal Document Chatbot":
            user_input(user_question, api_key, advisor_mode=False)
        else:
            user_input(user_question, api_key, advisor_mode=True)
    
    if mode == "Legal Document Chatbot":
        with st.sidebar:
            st.title("📌 Upload Legal Documents")
            pdf_docs = st.file_uploader("📂 Upload PDFs:", accept_multiple_files=True, key="pdf_uploader")
            
            if pdf_docs:
                raw_text = get_pdf_text(pdf_docs)
                summary = summarize_text(raw_text, api_key)
                st.markdown("### 📜 Document Summary:")
                st.info(summary)

            if st.button("🚀 Submit & Process") and api_key:
                with st.spinner("🔄 Processing..."):
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks, "faiss_index", api_key)
                    st.success("✅ Processing complete!")

    elif mode == "Legal Advisor":
        with st.sidebar:
            st.title("📌 Legal Dataset")
            if st.button("🔄 Precompute Legal Knowledge") and api_key:
                with st.spinner("Processing legal dataset..."):
                    success = store_legal_embeddings(api_key)
                    if success:
                        st.success("✅ Legal dataset processed!")
                    else:
                        st.error("No valid legal documents found in 'dataset/'.")

if __name__ == "__main__":
    main()
