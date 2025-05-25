import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# --- Configuration ---
st.set_page_config(
    page_title="Institution Details Extractor (Card View)",
    page_icon="🏢",
    layout="centered"
)

# Load environment variables (for API Key)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("🚨 **GEMINI_API_KEY not found in environment variables.** Please set it in a `.env` file.")
    st.stop()

# Configure Google Generative AI
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini Model
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# --- Session State for History ---
if "extraction_history" not in st.session_state:
    st.session_state.extraction_history = []

st.title("🏢 Extract Institution Details")
st.markdown("Enter the name or a brief description of an institution (including educational institutions) below.")
st.markdown("The AI will find and display key details in a **dynamic card format**.")



## Settings

#Use the sidebar to adjust the model's generation parameters.

with st.sidebar:
    st.header("Model Settings")

    temperature = st.slider(
        "**Temperature (Randomness)**",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Controls the creativity and randomness of the output. Higher values lead to more diverse, surprising outputs."
    )

    max_output_tokens = st.slider(
        "**Max Output Tokens**",
        min_value=50,
        max_value=2048,
        value=700,
        step=50,
        help="Maximum number of words or sub-word units the model will generate in its response."
    )

    if st.button("Clear History"):
        st.session_state.extraction_history = []
        st.success("Extraction history cleared!")
        st.rerun()



## Enter Institution

prompted_institution_input = st.text_input(
    "**Institution Name or Description:**",
    placeholder="e.g., 'Stanford University', 'SpaceX', 'University of Oxford', 'Tata Consultancy Services'"
)

# --- PROMPT TEMPLATE (Kept consistent for robust parsing) ---
extraction_prompt_template = """
You are an expert information extractor. Your task is to accurately find and present details about any institution, including educational institutions. Use your internal knowledge or access information from the web if necessary to gather the required details.

**Strictly adhere to the following output format: each detail must be on a NEW LINE and prefixed as shown.**
If a specific detail is not found or is not applicable based on the available information, state "Not Found" for that detail. Ensure the summary is concise and exactly 4 lines.

**Details to Extract (in this order):**
1. The founder of the Institution.
2. When it was founded.
3. The current branches/campuses in the institution.
4. How many employees (or faculty/staff for educational institutions) are working in it.
5. A brief 4-line summary of the institution.

**Example Output Format:**
Founder: [Founder's Name or "Not Found"]
Founded: [Founding Date/Year or "Not Found"]
Branches: [List of Branches/Campuses, comma-separated, or "Not Found"]
Employees: [Number of Employees/Faculty/Staff, or "Not Found"]
Summary: [A brief, exactly 4-line summary of the institution, covering its core function/history. Focus on key aspects and be concise. If information is scarce, provide a summary based on available data, still aiming for 4 lines if possible.]

**Institution to Analyze:**
{institution_input}
"""
# --- END OF PROMPT TEMPLATE ---


if st.button("Extract Details"):
    if not prompted_institution_input.strip():
        st.warning("⚠️ Please enter an institution name or description.")
    else:
        full_prompt = extraction_prompt_template.format(institution_input=prompted_institution_input)

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "response_mime_type": "text/plain",
        }

        st.info("💡 The AI is gathering information. This might involve web access.")
        with st.spinner("⏳ Extracting details... (This may take a moment)"):
            try:
                response = gemini_model.generate_content(full_prompt, generation_config=generation_config)
                extracted_text = response.text

                # --- NEW POST-PROCESSING TO PARSE INTO DICTIONARY ---
                extracted_data_dict = {}
                labels_and_prefixes = [
                    "Founder:",
                    "Founded:",
                    "Branches:",
                    "Employees:",
                    "Summary:"
                ]
                
                normalized_text = extracted_text.replace('\r\n', '\n').strip()

                for i, prefix in enumerate(labels_and_prefixes):
                    start_idx = normalized_text.find(prefix)

                    if start_idx != -1:
                        content_end_idx = len(normalized_text)
                        
                        for j in range(i + 1, len(labels_and_prefixes)):
                            next_prefix = labels_and_prefixes[j]
                            temp_next_idx = normalized_text.find(next_prefix, start_idx + len(prefix))
                            if temp_next_idx != -1:
                                content_end_idx = temp_next_idx
                                break

                        # Extract only the value, strip the prefix
                        value = normalized_text[start_idx + len(prefix) : content_end_idx].strip()
                        extracted_data_dict[prefix[:-1]] = value # Store without the colon (e.g., 'Founder')
                        
                        # Remove the extracted part from normalized_text for the next iteration
                        normalized_text = normalized_text[content_end_idx:].strip()
                    else:
                        # If a prefix is not found, set its value to "Not Found"
                        extracted_data_dict[prefix[:-1]] = "Not Found"

                # --- END OF NEW POST-PROCESSING ---

                st.session_state.extraction_history.append({
                    "input": prompted_institution_input,
                    "output_dict": extracted_data_dict # Store the dictionary
                })

                st.success("✅ Details extracted!")
                st.rerun()

            except Exception as e:
                st.error(f"❌ An error occurred during extraction: {e}")
                st.info("Please check your API key, network connection, or try a different input.")



## Previous Extractions

if st.session_state.extraction_history:
    # Display the most recent extraction at the top directly as a card
    most_recent_entry = st.session_state.extraction_history[-1]
    
    st.subheader("Latest Extraction:")
    with st.container(border=True): # Create a container to act as the card
        st.markdown(f"**Input:** `{most_recent_entry['input']}`")
        st.markdown("---") # Separator within the card
        st.markdown("**Extracted Details:**")
        
        # Dynamically display each parameter from the dictionary
        for param, value in most_recent_entry['output_dict'].items():
            # Apply different styling if it's the Summary (which might be multiline)
            if param == "Summary":
                st.markdown(f"**{param}:**") # New line for Summary header
                st.write(value) # Use st.write for multi-line text to render better
            else:
                st.markdown(f"**{param}:** {value}") # Parameter and value on one line
    
    st.subheader("Previous Extractions History:")
    # Display older extractions in collapsible expanders
    # Start from second to last (since last was displayed above)
    for i, entry in enumerate(reversed(st.session_state.extraction_history[:-1])):
        expander_title = f"**Input:** `{entry['input']}`"
        with st.expander(expander_title):
            st.markdown(f"**Extracted Details:**")
            # Dynamically display each parameter from the dictionary
            for param, value in entry['output_dict'].items():
                if param == "Summary":
                    st.markdown(f"**{param}:**")
                    st.write(value)
                else:
                    st.markdown(f"**{param}:** {value}")
else:
    st.info("No previous extractions yet. Enter an institution above to get started!")