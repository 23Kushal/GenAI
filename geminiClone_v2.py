import streamlit as st
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import html
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain.llms import Cohere as CohereLLM
from langchain.embeddings.base import Embeddings
import cohere

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

if not GEMINI_API_KEY or not COHERE_API_KEY:
    st.error("Missing API keys in .env file.")
    st.stop()

# Initialize Gemini client
genai_client = genai.Client(api_key=GEMINI_API_KEY)

# Initialize Cohere client
cohere_client = cohere.Client(COHERE_API_KEY)

# Custom Embedding Class
class CustomCohereEmbeddings(Embeddings):
    def embed_documents(self, texts):
        response = cohere_client.embed(
            texts=texts,
            model="embed-english-v3.0",
            input_type="search_document"
        ).embeddings
        return response

    def embed_query(self, text):
        return self.embed_documents([text])[0]

# Set page config
st.set_page_config(page_title="Gemini Chat with Files", page_icon="🧠", layout="centered")

# Title
st.title("🧠 Gemini Chat with File Support")

# Session state initialization
if "history" not in st.session_state:
    st.session_state.history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# File Upload Section
uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"], key="file_uploader")

# Process File Button
if uploaded_file is not None:
    if st.button("📄 Process File"):
        try:
            with st.spinner("Processing your file... Please wait."):
                # Read file content
                if uploaded_file.type == "application/pdf":
                    pdf_reader = PdfReader(uploaded_file)
                    raw_text = ""
                    for page in pdf_reader.pages:
                        raw_text += page.extract_text()
                    content = raw_text
                else:
                    content = uploaded_file.read().decode("utf-8")

                # Split into chunks
                splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                docs = [Document(page_content=chunk) for chunk in splitter.split_text(content)]

                # Create vectorstore
                embeddings = CustomCohereEmbeddings()
                vectorstore = FAISS.from_documents(docs, embeddings)

                # Save to session state
                st.session_state.vectorstore = vectorstore
                st.session_state.file_processed = True

                st.success("File processed successfully!")

        except Exception as e:
            st.error(f"Error processing file: {e}")

elif uploaded_file is None:
    st.info("You can upload a file to ask questions about its content.")

# Display chat history
for idx, (user_msg, ai_msg) in enumerate(st.session_state.history):
    # User message card
    st.markdown(
        f"""
        <div style="
            border: 1px solid #444;
            padding: 15px;
            border-radius: 10px;
            background-color: #1e1e1e;
            color: #f0f0f0;
            margin-bottom: 10px;
        ">
            <strong>🧑 You:</strong><br>{user_msg}
        </div>
        """,
        unsafe_allow_html=True
    )

    # AI message card
    st.markdown(
        f"""
        <div style="
            border: 1px solid #444;
            padding: 15px;
            border-radius: 10px;
            background-color: #2a2a2a;
            color: #f0f0f0;
            margin-bottom: 20px;
        ">
            <strong>🤖 Gemini:</strong><br>{html.escape(ai_msg)}
        </div>
        """,
        unsafe_allow_html=True
    )

# Ensure user_input is initialized in session state (already done at the top, but good practice)
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Fetch current value for the text input from session state
user_input_value = st.session_state.get("user_input", "")

with st.form(key="chat_form_v2"):
    user_input_from_field = st.text_input("Ask something:", key="_gemini_input_v2", value=user_input_value)
    submit_button = st.form_submit_button("📩 Send")

if submit_button:
    st.session_state.user_input = user_input_from_field # Capture text from field
    if st.session_state.user_input.strip(): # Process if not empty
        with st.spinner("Thinking..."):
            try:
                # Use st.session_state.user_input for processing
                user_query = st.session_state.user_input 
                if st.session_state.file_processed:
                    # Use Gemini over the uploaded document
                    retriever = st.session_state.vectorstore.as_retriever()
                    qa = RetrievalQA.from_chain_type(
                        llm=CohereLLM(cohere_api_key=COHERE_API_KEY, model="command-r-plus"),
                        chain_type="stuff",
                        retriever=retriever
                    )
                    answer = qa.run(user_query) # Use user_query
                else:
                    # Use Gemini directly
                    response = genai_client.models.generate_content(
                        model="gemini-2.0-flash",
                        config=types.GenerateContentConfig(
                            system_instruction="You are a helpful assistant."
                        ),
                        contents=user_query # Use user_query
                    )
                    answer = response.text

                # Store the raw answer in history
                # Use st.session_state.user_input (which is user_query) for history
                st.session_state.history.append((user_query, answer)) 
                st.session_state.user_input = ""  # Clear session state for the input field
                st.rerun()

            except Exception as e:
                st.error(f"Gemini error: {e}")

# Clear chat button
if st.button("🗑 Clear Chat"):
    st.session_state.history = []
    st.session_state.user_input = ""
    st.rerun()
