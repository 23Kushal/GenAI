import streamlit as st
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import html

# Load API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Missing GEMINI_API_KEY in .env file.")
    st.stop()

# Configure Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

# Set page config
st.set_page_config(
    page_title="IPC Chat-Bot",
    page_icon="🧠",
    layout="centered"
)

# Title
st.title("🧠 What do you want to know about IPC")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# Function to call Gemini API
def get_gemini_response(prompt):
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction="You are an expert in IPC (Indian Penal codes) and can answer questions related to it. Please provide detailed and accurate information."
        ),
        contents=prompt
    )
    return response.text

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
            <strong>🤖 Gemini:</strong><br>{ai_msg}
        </div>
        """,
        unsafe_allow_html=True
    )

# Input box
user_input = st.text_input("Ask something about IPC:", key="_gemini_input", placeholder="Type your question here...")

# Send button
if st.button("📩 Send", disabled=not user_input.strip()):
    if user_input.strip():
        with st.spinner("Thinking..."):
            try:
                response = get_gemini_response(user_input)
                safe_ai_msg = html.escape(response)
                print(safe_ai_msg)
                st.session_state.history.append((user_input, response))
                #st.session_state._gemini_input = ""  # Clear input
                st.rerun()
            except Exception as e:
                st.error(f"Gemini error: {e}")

# Clear chat button
if st.button("🗑 Clear Chat"):
    st.session_state.history = []
    st.rerun()