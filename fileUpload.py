import streamlit as st
import cohere
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from langchain.llms import Cohere as CohereLLM
from langchain.embeddings.base import Embeddings
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Document Chat Bot",
    page_icon="📄",
    layout="centered"
)

# Title
st.title("Ask me about your document")

# Load API Key
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    st.error("COHERE_API_KEY not found in .env file.")
    st.stop()

# Initialize Cohere client
try:
    co = cohere.Client(COHERE_API_KEY)
except Exception as e:
    st.error(f"Error initializing Cohere client: {e}")
    st.stop()

# Custom Embedding Class
class CustomCohereEmbeddings(Embeddings):
    def embed_documents(self, texts):
        response = co.embed(
            texts=texts,
            model="embed-english-v3.0",
            input_type="search_document"
        )
        return response.embeddings

    def embed_query(self, text):
        return self.embed_documents([text])[0]

embedding = CustomCohereEmbeddings()
llm = CohereLLM(cohere_api_key=COHERE_API_KEY, model="command-r-plus")

# File Upload Section
uploaded_file = st.file_uploader("Upload a .txt or .pdf document", type=["txt", "pdf"])

if uploaded_file is not None:
    try:
        # Read file content
        if uploaded_file.type == "application/pdf":
            pdf_reader = PdfReader(uploaded_file)
            raw_text = ""
            for page in pdf_reader.pages:
                raw_text += page.extract_text()
            content = raw_text
        else:
            content = uploaded_file.read().decode("utf-8")

        # Create document chunks
        doc = Document(page_content=content)
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents([doc])

        # Build vectorstore
        vectorstore = FAISS.from_documents(chunks, embedding)
        retriever = vectorstore.as_retriever()

        # Setup QA chain
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff"
        )

        # Session state for chat history
        if "history" not in st.session_state:
            st.session_state.history = []

        # Display chat history
        for q, a in st.session_state.history:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**AI:** {a}")

        # User input
        user_input = st.text_input("Ask something about your document:")

        if user_input:
            answer = qa.run(user_input)
            st.session_state.history.append((user_input, answer))
            st.markdown(f"**You:** {user_input}")
            st.markdown(f"**AI:** {answer}")

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Please upload a document (.txt or .pdf) to begin.")