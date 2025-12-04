
import os
os.environ["LITELLM_MODEL_PROVIDER"] = "gemini"
os.environ["GEMINI_API_KEY"] = "AIzaSyAm62k_mpMqcdQn8q3IqCjaOP4bCtE-qzE"

import asyncio  # Add this import near the top with others
import pickle
import hashlib
try:
    from langchain_core.documents import Document
except Exception:
    try:
        from langchain.schema import Document
    except Exception:
        Document = None  # Will error later if actually used without being available

# --- Fix for "There is no current event loop in thread 'ScriptRunner.scriptThread'" ---
# This error occurs in Streamlit when using asyncio in a thread that does not have an event loop.
# To fix, always ensure an event loop exists before using asyncio-related code.

def ensure_event_loop():
    import asyncio
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)



import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import (
    PyPDFLoader, CSVLoader, UnstructuredFileLoader,
    UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader
)
from langchain_text_splitters import NLTKTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
import nltk
nltk.download('averaged_perceptron_tagger')
import warnings
import os
import logging
from dotenv import load_dotenv
import tempfile
import csv
import pandas as pd
from datetime import datetime
import json
import yaml  # Add YAML import
# Add OpenAI import
try:
    import openai
except ImportError:
    openai = None
# Add gTTS for TTS
try:
    from gtts import gTTS
except ImportError:
    gTTS = None
# Add CrewAI imports
try:
    from crewai import Agent, Task, Crew, LLM
    from crewai_tools import WebsiteSearchTool
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False

# Add Vertex AI imports
try:
    import vertexai
    from vertexai.vision_models import VideoGenerationModel, Video
    from google.cloud import storage
    VERTEXAI_AVAILABLE = True
except ImportError:
    VERTEXAI_AVAILABLE = False

# Load environment variables
load_dotenv()

# Path Variables
#faviconPath = "../dashboard/images/****.png"

import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import requests
import json
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import NLTKTextSplitter, RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS  # Changed from Chroma to FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_community.document_loaders import UnstructuredFileLoader  # Fixed deprecated import
from langchain_community.document_loaders import UnstructuredPowerPointLoader, UnstructuredWordDocumentLoader, UnstructuredXMLLoader, JSONLoader
from tqdm import tqdm
from io import BytesIO  # Add this at the top with other imports
from PIL import Image
import mimetypes
# from google.generativeai import GenerativeAIError
import nltk
import warnings
import uuid
import os, time, tempfile
import sys
import argparse
import logging
from bs4 import BeautifulSoup
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from audio_recorder_streamlit import audio_recorder
import openai
import io
try:
    import speech_recognition as sr
except ImportError:
    sr = None
try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None
from pydantic import SecretStr
from google.generativeai.types import GenerationConfig
# FIRST_EDIT: additional imports for image generation feature
from google import genai as genai_new
from google.genai import types

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("CPR_RAG_App")

# Quiet the NLTK messages
nltk.download('punkt', quiet=True)

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Created a chunk of size .*, which is longer than the specified .*")
warnings.filterwarnings("ignore", message="NLTK punkt already downloaded")

# No need for pysqlite3 with FAISS
# try:
#     __import__('pysqlite3')
#     sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
#     print("Using pysqlite3 for ChromaDB")
# except ImportError:
#     print("pysqlite3 not available, using default sqlite3")

def _resolve_google_api_key():
    # Prefer GOOGLE_API_KEY from environment first, then GEMINI_API_KEY; fall back to constant
    key_env = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if key_env and isinstance(key_env, str) and key_env.strip():
        return key_env.strip()
    # Fallback to the previously hardcoded key if no env key present
    return "AIzaSyBIUiyk-Ksb0j3UxOzJqrTxqnNffzG_Y2M"

GOOGLE_API_KEY = _resolve_google_api_key()
Imagen_API_KEY = os.environ.get("IMAGEN_API_KEY") or GOOGLE_API_KEY

# Ensure the generative AI client is configured for paths that use google.generativeai directly
try:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    genai.configure(api_key=GOOGLE_API_KEY)  # type: ignore[attr-defined]
except Exception as _e:
    logger.warning(f"Could not configure google-generativeai: {_e}")

def ensure_genai_config():
    """Ensure google.generativeai is configured with the current API key."""
    try:
        genai.configure(api_key=GOOGLE_API_KEY)  # type: ignore[attr-defined]
    except Exception as _e:
        logger.warning(f"Failed to configure google-generativeai: {_e}")

# **Voice Functionality Helper Functions**
def render_voice_interaction_mode(key_prefix):
    """Render the voice/text interaction mode selector"""
    return st.radio(
        "Choose interaction mode:",
        ["Text", "Voice"],
        horizontal=True,
        key=f"{key_prefix}_interaction_mode"
    )

def render_voice_input(key_prefix, prefill_key=None):
    """Render voice input component with transcription"""
    voice_text = ""
    transcription_error = False

    with st.container(border=True):
        st.subheader("🎙️ Voice Recorder", anchor=False)
        st.caption("Tap the microphone, speak your question, then tap again to stop. The audio will be transcribed automatically.")
        audio_bytes_voice = audio_recorder(
            text="🎤 Tap to Record / Stop",
            icon_name="microphone",
            icon_size="2x",
            neutral_color="#6C757D",
            recording_color="#FF4B4B",
            key=f"{key_prefix}_voice_rec",
        )

    if audio_bytes_voice:
        # Playback preview
        st.audio(audio_bytes_voice, format='audio/wav', start_time=0)
        if sr is not None and AudioSegment is not None:
            with st.spinner("🎧 Transcribing..."):
                try:
                    audio_file_voice = io.BytesIO(audio_bytes_voice)
                    audio_file_voice.seek(0)
                    audio_segment_voice = AudioSegment.from_file(audio_file_voice, format="wav")
                    temp_wav_voice = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    audio_segment_voice.export(temp_wav_voice.name, format="wav")
                    temp_wav_voice.close()
                    recognizer_voice = sr.Recognizer()
                    with sr.AudioFile(temp_wav_voice.name) as source:
                        audio_data_voice = recognizer_voice.record(source)
                        voice_text = recognizer_voice.recognize_google(audio_data_voice)
                    os.remove(temp_wav_voice.name)
                    st.success(f"Transcription: {voice_text}")
                except Exception as e:
                    transcription_error = True
                    st.warning(f"Automatic transcription failed: {e}. Please type what you said.")
        else:
            transcription_error = True
            st.info("SpeechRecognition or pydub not installed. Please type what you said.")

        if (transcription_error or (audio_bytes_voice and not voice_text)):
            manual_voice_text = st.text_area("Manual transcription (type what you said):", "", height=100)
            if manual_voice_text:
                voice_text = manual_voice_text

        # Prefill the session state or return the voice text
        if voice_text and prefill_key:
            st.session_state[prefill_key] = voice_text

    return voice_text

def render_voice_output(text_content, interaction_mode):
    """Render text-to-speech output for voice mode"""
    if interaction_mode == "Voice" and gTTS is not None:
        try:
            # Remove any leading role prefix like 'agent:'
            spoken_text = re.sub(r'^(agent|expert):\s*', '', text_content, flags=re.I)
            tts = gTTS(text=spoken_text, lang='en')
            tts_fp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            tts.save(tts_fp.name)
            tts_fp.close()
            with open(tts_fp.name, 'rb') as audio_file_out:
                audio_bytes_out = audio_file_out.read()
            st.audio(audio_bytes_out, format='audio/mp3')
            os.remove(tts_fp.name)
        except Exception as e:
            st.warning(f"TTS generation failed: {e}")

def get_voice_or_text_input(interaction_mode, text_label, key_prefix, prefill_key=None, placeholder=None):
    """Get input from either voice or text based on interaction mode"""
    if interaction_mode == "Voice":
        voice_text = render_voice_input(key_prefix, prefill_key)
        # If we have a transcribed voice text, use it immediately
        if voice_text:
            return voice_text
        # Otherwise, fall back to text input for manual entry/editing
        if prefill_key and prefill_key in st.session_state:
            return st.text_area(text_label, value=st.session_state[prefill_key], key=f"{key_prefix}_text", height=100, placeholder=placeholder)
        else:
            return st.text_area(text_label, key=f"{key_prefix}_text", height=100, placeholder=placeholder)
    else:
        return st.text_area(text_label, key=f"{key_prefix}_text", height=100, placeholder=placeholder)

# ---- Common UI helpers for saved vs upload flows ----
def select_source_and_saved_group(file_type, key_prefix):
    """Render an Upload new vs Use saved selector and return (use_saved, selected_group_id)."""
    saved_groups = st.session_state.get('saved_files', {}).get(file_type, {})
    use_saved_source = False
    selected_saved_group_id = None

    # Always render a source choice for consistent UI
    if saved_groups:
        source_options = ["Upload new", "Use saved"]
    else:
        # Indicate none available; treat as not actionable
        source_options = ["Upload new", "Use saved (none available)"]

    source_choice = st.radio(
        "Choose source:",
        source_options,
        horizontal=True,
        key=f"{key_prefix}_source_choice"
    )

    # Only enable saved flow when groups exist and user picked it
    if saved_groups and source_choice == "Use saved":
        use_saved_source = True
        display_map = {f"{gid} - {meta['group_name']}": gid for gid, meta in saved_groups.items()}
        selected_display = st.selectbox(
            f"Select a saved {file_type} vector database:",
            ["Select a saved group..."] + list(display_map.keys()),
            key=f"{key_prefix}_saved_select"
        )
        if selected_display != "Select a saved group...":
            selected_saved_group_id = display_map[selected_display]
    elif not saved_groups and "Use saved" in source_choice:
        st.info("No saved groups available yet. Upload and save a group first.")

    return use_saved_source, selected_saved_group_id

def build_combined_db_name_from_uploads(uploaded_files, default_name):
    db_name_list = []
    for file in uploaded_files:
        original_name = file.name
        if len(original_name) > 20:
            name, ext = os.path.splitext(original_name)
            truncated_name = name[:20 - len(ext)] + ext
            db_name_list.append(os.path.splitext(truncated_name)[0])
        else:
            db_name_list.append(os.path.splitext(original_name)[0])
    combined_db_name = "_".join(db_name_list) if db_name_list else default_name
    if len(combined_db_name) > 50:
        combined_db_name = combined_db_name[:50]
    return combined_db_name

# **Retrieve all top documents and their scores**
def retrieve_with_scores(query, combined_db_name):
    # Embeddings
    embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=SecretStr(GOOGLE_API_KEY), model="models/embedding-001")

    # Resolve FAISS directory: support absolute/relative saved paths and legacy short names
    try:
        base_dir = combined_db_name
        # If provided name doesn't look like a path to an existing dir, try ./name
        if not os.path.isdir(base_dir):
            candidate = os.path.join(".", combined_db_name)
            if os.path.isdir(candidate):
                base_dir = candidate

        index_path = os.path.join(base_dir, "index.faiss")
        if not os.path.exists(index_path):
            logger.warning(f"FAISS index not found at {index_path}")
            return [], []

        # Load the FAISS index with allow_dangerous_deserialization=True
        db_connection = FAISS.load_local(
            base_dir,
            embedding_model,
            allow_dangerous_deserialization=True
        )

        # Use similarity_search_with_score for FAISS
        retrieved_docs = db_connection.similarity_search_with_score(query, k=30)  # Retrieve top 30
        if not retrieved_docs:
            logger.warning(f"No documents found for query: {query}")
            return [], []  # No results case

        docs, scores = zip(*retrieved_docs)  # Separate docs and scores

        # Log retrieved context (truncated for brevity)
        logger.info(f"Retrieved {len(docs)} documents for query: {query}")
        for i, (doc, score) in enumerate(zip(docs, scores)):
            content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            logger.info(f"Doc {i+1} (Score: {score:.4f}): {content_preview}")

        return [doc.page_content for doc in docs], scores
    except Exception as e:
        logger.error(f"Error loading vector database: {e}")
        return [], []

chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
        You are an expert in text analysis. Your task is to generate an answer based on the retrieved text chunks, then determine which chunks were used and compute the overall similarity score.

        #### **Task**:
        1. **Generate an answer** based on the retrieved chunks.
        2. **Identify which chunks contributed** to the generated answer.
        3. Fetch the **similarity scores** of the identified chunks.
        4. Compute the **overall similarity score** using the formula:
           \[
           \text{Overall Similarity} = \frac{\sum (\text{similarity of used chunks})}{\text{total used chunks}}
           \]
        5. **Output the result in the following JSON format without \n:**

        ```json
        {
            "Answer": "[Your response]","Used Chunks": "[chunk numbers as a comma-separated list]","Similarity Scores": "[scores as a comma-separated list]","Overall Similarity": "[calculated score between 0 and 1]","Reasoning": "[Reason for choosing chunk]"
        }
        ```
    """),
    HumanMessagePromptTemplate.from_template("""
    **Retrieved Contexts and Scores:**
    {context_with_scores}

    **Question:** {question}


    """)
])

# Chain for retrieval and response generation
def rag_pipeline(question, model, combined_db_name, temperature=1.0, top_p=0.94, max_tokens=9000):
    logger.info(f"Processing query: {question}")
    contexts, scores = retrieve_with_scores(question, combined_db_name)

    if not contexts:
        logger.warning("No relevant information found for the query.")
        fallback_json = json.dumps({
            "Answer": "No relevant information found for your query. Please try a different question or provide more context.",
            "Used Chunks": "None",
            "Similarity Scores": "None",
            "Overall Similarity": "0.0",
            "Reasoning": "No relevant documents were found in the knowledge base."
        })
        return type('RAGMockResponse', (), {"__init__": lambda self, content: setattr(self, 'content', content)})(fallback_json)

    formatted_contexts = "\n\n".join(
        f"Context {i+1} (Score: {scores[i]:.4f}):\n{contexts[i]}" for i in range(len(contexts))
    )
    prompt = f"""
You are an expert in text analysis. Your task is to generate an answer based on the retrieved text chunks, then determine which chunks were used and compute the overall similarity score.

#### **Task**:
1. **Generate an answer** based on the retrieved chunks.
2. **Identify which chunks contributed** to the generated answer.
3. Fetch the **similarity scores** of the identified chunks.
4. Compute the **overall similarity score** using the formula:
   [Overall Similarity = sum(similarity of used chunks) / total used chunks]
5. **Output the result in the following JSON format without \n:**

```json
{{
    "Answer": "[Your response]",
    "Used Chunks": "[chunk numbers as a comma-separated list]",
    "Similarity Scores": "[scores as a comma-separated list]",
    "Overall Similarity": "[calculated score between 0 and 1]",
    "Reasoning": "[Reason for choosing chunk]"
}}
```

**Retrieved Contexts and Scores:**
{formatted_contexts}

**Question:** {question}
"""
    llm = ChatGoogleGenerativeAI(
        api_key=SecretStr(GOOGLE_API_KEY),
        model=model,
        temperature=temperature,
        top_p=top_p,
    )
    response = llm.invoke(prompt)

    # Save to CSV log
    try:
        response_content = response.content if hasattr(response, 'content') else str(response)
        save_query_response_to_csv(
            user_query=question,
            bot_response=response_content,
            query_type="RAG Query",
            model_used=model,
            temperature=temperature,
            additional_metadata={
                'top_p': top_p,
                'contexts_found': len(contexts),
                'avg_similarity': sum(scores) / len(scores) if scores else 0
            }
        )
    except Exception as e:
        logger.error(f"Error saving RAG response to CSV: {e}")

    return type('RAGMockResponse', (), {"__init__": lambda self, content: setattr(self, 'content', content)})(response.content if hasattr(response, 'content') else response)

def sidebar_accordion():
    # Accordion state
    if 'sidebar_section' not in st.session_state:
        st.session_state['sidebar_section'] = 'media'
    if 'sidebar_use_agent' not in st.session_state:
        st.session_state['sidebar_use_agent'] = False
    if 'media_type' not in st.session_state:
        st.session_state['media_type'] = 'Text Prompt'
    if 'selected_model' not in st.session_state:
        st.session_state['selected_model'] = 'gemini-2.0-flash'
    if 'temperature' not in st.session_state:
        st.session_state['temperature'] = 1.0
    if 'top_p' not in st.session_state:
        st.session_state['top_p'] = 0.94
    if 'max_tokens' not in st.session_state:
        st.session_state['max_tokens'] = 7000
    if 'agent_enabled' not in st.session_state:
        st.session_state['agent_enabled'] = False

    def set_section(section):
        st.session_state['sidebar_section'] = section
        if section == 'agent':
            st.session_state['sidebar_use_agent'] = True
            st.session_state['agent_enabled'] = True
        elif section == 'media':
            st.session_state['sidebar_use_agent'] = False
            st.session_state['agent_enabled'] = False

    def toggle_media_section():
        if st.session_state['sidebar_section'] == 'media':
            st.session_state['sidebar_section'] = None
        else:
            st.session_state['sidebar_section'] = 'media'
            st.session_state['sidebar_use_agent'] = False
            st.session_state['agent_enabled'] = False

    with st.sidebar:
        # 1. Media Section (header always visible)
        media_selected = st.session_state['sidebar_section'] == 'media'
        if st.button("🗂️ Select type of Media", key="media_section_btn", use_container_width=True, help="Choose your media type", on_click=toggle_media_section):
            pass
        if media_selected:
            if st.session_state['sidebar_use_agent']:
                st.info("Agent is active. Select this to switch to media mode.")
            else:
                media_options = [
                    "Text Prompt", "URL Search", "PDF files", "CSV files", "Excel files", "Images", "Image editing", "Generate Images", "Video, mp4 file", "Audio files", "TXT files", "PPT files", "DOCX files", "XML files", "JSON files", "YAML files", "H files", "DAT files", "Markdown files", "Voice to Voice", "Multiple Media"
                ]
                st.session_state['media_type'] = st.radio(
                    "Choose one:",
                    media_options,
                    key="media_type_radio",
                    index=media_options.index(st.session_state.get('media_type', 'Text Prompt')),
                    help="Select the type of content you want to work with"
                )

        # 2. LLM Model Section
        llm_selected = st.session_state['sidebar_section'] == 'llm'
        if st.button("🤖 Choose LLM", key="llm_section_btn", use_container_width=True, help="Choose LLM model", on_click=lambda: set_section('llm')):
            pass
        if llm_selected:
            llm_models = [
                "gemini-2.0-flash",
                "gemini-2.5-flash",
                "gemini-2.5-flash-lite",
                "gemini-2.5-pro",
                "gemini-3-pro-preview"
            ]
            st.session_state['selected_model'] = st.radio(
                "Choose LLM:",
                llm_models,
                key="model_select",
                index=llm_models.index(st.session_state.get('selected_model', 'gemini-2.0-flash')),
                help="Select a Gemini model you want to use."
            )

        # 3. LLM Config Section
        config_selected = st.session_state['sidebar_section'] == 'config'
        if st.button("⚙️ LLM Config", key="config_section_btn", use_container_width=True, help="LLM generation settings", on_click=lambda: set_section('config')):
            pass
        if config_selected:
            st.session_state['temperature'] = st.slider("Temperature:", min_value=0.0, max_value=2.0, value=st.session_state.get('temperature', 1.0), step=0.25, help="Lower = less creative, higher = more creative.")
            st.session_state['top_p'] = st.slider("Top P:", min_value=0.0, max_value=1.0, value=st.session_state.get('top_p', 0.94), step=0.01, help="Lower = less random, higher = more random.")
            st.session_state['max_tokens'] = st.slider("Max Tokens:", min_value=100, max_value=9000, value=st.session_state.get('max_tokens', 8000), step=100, help="Number of response tokens, 8194 is limit.")

        # 4. Agent Section
        agent_selected = st.session_state['sidebar_section'] == 'agent'
        if st.button("🔧 Use Agent (CrewAI)", key="agent_section_btn", use_container_width=True, help="Enable CrewAI agent", on_click=lambda: set_section('agent')):
            pass
        if agent_selected:
            st.session_state['sidebar_use_agent'] = True
            st.session_state['agent_enabled'] = True
            st.info("✅ Agent Active")
        else:
            st.session_state['sidebar_use_agent'] = False

        # 5. Query Log Section
        st.markdown("---")
        with st.expander("📊 Query-Response Log", expanded=False):
            st.markdown("**All your queries and responses are automatically saved to a CSV file for record keeping.**")

            # Check if log file exists and show stats
            csv_filename = "query_response_log.csv"
            if os.path.exists(csv_filename):
                try:
                    df = pd.read_csv(csv_filename)
                    st.metric("Total Interactions", len(df))
                    st.metric("Current Session", st.session_state.get('session_id', 'Unknown')[:20] + "...")

                    # Create two columns for Download and Clear buttons
                    col1, col2 = st.columns(2)

                    with col1:
                        # Download button
                        with open(csv_filename, 'rb') as file:
                            st.download_button(
                                label="📥 Download Log CSV",
                                data=file.read(),
                                file_name=f"query_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                help="Download the complete interaction log",
                                use_container_width=True
                            )

                    with col2:
                        # Clear logs button
                        if st.button(
                            "🗑️ Clear All Logs",
                            help="Permanently delete all log entries",
                            use_container_width=True,
                            type="secondary"
                        ):
                            try:
                                os.remove(csv_filename)
                                st.success("✅ All logs cleared successfully!")
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error clearing logs: {e}")

                    # Show recent interactions
                    if len(df) > 0:
                        st.subheader("Recent Interactions")
                        recent_df = df.tail(5)[['timestamp', 'query_type', 'user_query']]
                        recent_df['user_query'] = recent_df['user_query'].str[:50] + "..."
                        st.dataframe(recent_df, use_container_width=True)

                except Exception as e:
                    st.warning(f"Could not read log file: {e}")
            else:
                st.info("No interactions logged yet. Start chatting to create the log!")
                st.metric("Current Session", st.session_state.get('session_id', 'Unknown')[:20] + "...")

def page_setup():
    st.header("Turbo LLM (RAG with FAISS) Chatbot Playground", divider="orange", anchor=False)
    st.title(":blue[Chat App]", anchor=False)
    st.subheader("This chatbot is running in an IL4 environment.")

    # Global button style: make primary buttons light blue (affects red primary buttons)
    st.markdown(
        """
        <style>
        /* Streamlit primary buttons to light blue */
        div[data-testid="baseButton-primary"] > button, button[kind="primary"] {
            background-color: #87CEFA !important; /* LightSkyBlue */
            border-color: #87CEFA !important;
            color: #000 !important;
        }
        div[data-testid="baseButton-primary"] > button:hover, button[kind="primary"]:hover {
            background-color: #6CB8E6 !important; /* Slightly darker on hover */
            border-color: #6CB8E6 !important;
            color: #000 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    #hide_menu_style = """
      #      <style>
        #    MainMenu {visibility: hidden;}
         #   </style>
          #  """
    #st.markdown(hide_menu_style, unsafe_allow_html=True)


def initialize_session_state():
    if 'session_id' not in st.session_state:
        # Generate a unique session ID using timestamp and random component
        import uuid
        st.session_state['session_id'] = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    if 'media_type' not in st.session_state:
        st.session_state['media_type'] = 'Text Prompt'
    if 'selected_model' not in st.session_state:
        st.session_state['selected_model'] = 'gemini-2.0-flash'
    if 'temperature' not in st.session_state:
        st.session_state['temperature'] = 1.0
    if 'top_p' not in st.session_state:
        st.session_state['top_p'] = 0.94
    if 'max_tokens' not in st.session_state:
        st.session_state['max_tokens'] = 7000
    if 'agent_enabled' not in st.session_state:
        st.session_state['agent_enabled'] = False
    if 'sidebar_use_agent' not in st.session_state:
        st.session_state['sidebar_use_agent'] = False
    if 'sidebar_section' not in st.session_state:
        st.session_state['sidebar_section'] = 'media'
    if 'db_initialized' not in st.session_state:
        st.session_state['db_initialized'] = {}
    if 'agent_chat_history' not in st.session_state:
        st.session_state['agent_chat_history'] = []
    # File storage system initialization
    if 'saved_files' not in st.session_state:
        st.session_state['saved_files'] = {}
    if 'selected_file_group' not in st.session_state:
        st.session_state['selected_file_group'] = None
    if 'file_storage_dir' not in st.session_state:
        st.session_state['file_storage_dir'] = 'saved_embeddings'

def update_media_type():
    st.session_state['media_type'] = st.session_state['media_type_radio']

# ===== FILE STORAGE MANAGEMENT SYSTEM =====

def ensure_storage_directory():
    """Ensure the storage directory exists"""
    storage_dir = st.session_state['file_storage_dir']
    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)
    return storage_dir

def get_file_hash(file_content, file_name):
    """Generate a unique hash for a file based on content and name"""
    hasher = hashlib.md5()
    # Handle both string and bytes inputs
    if isinstance(file_content, str):
        hasher.update(file_content.encode('utf-8'))
    else:
        hasher.update(file_content)
    hasher.update(file_name.encode())
    return hasher.hexdigest()

def save_file_group(file_group_data, file_type, group_name=None):
    """Save a file group (single or multiple files) with embeddings"""
    storage_dir = ensure_storage_directory()

    # Generate group name if not provided
    if not group_name:
        if len(file_group_data['files']) == 1:
            group_name = file_group_data['files'][0]['name']
        else:
            group_name = f"Multiple_{file_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create unique group ID
    group_id = f"{file_type}_{get_file_hash(str(file_group_data['files']), group_name)}"

    # Save metadata
    metadata = {
        'group_id': group_id,
        'group_name': group_name,
        'file_type': file_type,
        'files': file_group_data['files'],
        'created_at': datetime.now().isoformat(),
        'db_path': os.path.join(storage_dir, f"{group_id}_db.pkl"),
        'metadata_path': os.path.join(storage_dir, f"{group_id}_metadata.json")
    }

    # Save database if present
    if 'db' in file_group_data:
        db = file_group_data['db']
        # Check if it's a FAISS database
        if hasattr(db, 'save_local') and callable(getattr(db, 'save_local')):
            # For FAISS, use save_local() method instead of pickle
            # FAISS expects a directory path, not a file path
            faiss_dir = os.path.join(storage_dir, f"{group_id}_db")
            db.save_local(faiss_dir)
            metadata['db_type'] = 'faiss'
            metadata['db_path'] = faiss_dir  # Update to directory path for FAISS
        else:
            # For other databases, use pickle
            with open(metadata['db_path'], 'wb') as f:
                pickle.dump(db, f)
            metadata['db_type'] = 'pickle'

    # Save metadata
    with open(metadata['metadata_path'], 'w') as f:
        json.dump(metadata, f, indent=2)

    # Update session state
    if file_type not in st.session_state['saved_files']:
        st.session_state['saved_files'][file_type] = {}

    st.session_state['saved_files'][file_type][group_id] = metadata

    return group_id, metadata

def load_file_group(group_id, file_type):
    """Load a saved file group"""
    if file_type not in st.session_state['saved_files']:
        return None

    if group_id not in st.session_state['saved_files'][file_type]:
        return None

    metadata = st.session_state['saved_files'][file_type][group_id]

    # Load database
    if os.path.exists(metadata['db_path']):
        db_type = metadata.get('db_type', 'pickle')  # Default to pickle for backward compatibility
        if db_type == 'faiss':
            # For FAISS, load from the saved directory
            try:
                from langchain_community.vectorstores import FAISS
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                GOOGLE_API_KEY = _resolve_google_api_key()
                embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=SecretStr(GOOGLE_API_KEY), model="models/embedding-001")
                # FAISS loads from directory path
                db = FAISS.load_local(metadata['db_path'], embedding_model, allow_dangerous_deserialization=True)
                metadata['db'] = db
            except Exception as e:
                st.error(f"Failed to load FAISS database: {e}")
                return None
        else:
            # For other databases, use pickle
            with open(metadata['db_path'], 'rb') as f:
                db = pickle.load(f)
            metadata['db'] = db

    return metadata

def delete_file_group(group_id, file_type):
    """Delete a saved file group"""
    if file_type not in st.session_state['saved_files']:
        return False

    if group_id not in st.session_state['saved_files'][file_type]:
        return False

    metadata = st.session_state['saved_files'][file_type][group_id]

    # Delete files
    try:
        if os.path.exists(metadata['db_path']):
            db_type = metadata.get('db_type', 'pickle')
            if db_type == 'faiss':
                # For FAISS, remove the entire directory
                import shutil
                shutil.rmtree(metadata['db_path'])
            else:
                # For other types, remove the file
                os.remove(metadata['db_path'])
        if os.path.exists(metadata['metadata_path']):
            os.remove(metadata['metadata_path'])
    except Exception as e:
        st.error(f"Error deleting files: {e}")
        return False

    # Remove from session state
    del st.session_state['saved_files'][file_type][group_id]

    # Clear selection if this was selected
    if st.session_state.get('selected_file_group') == group_id:
        st.session_state['selected_file_group'] = None

    return True

def save_media_embeddings(uploaded_files, file_type, embedding_model, combined_db_name, progress_bar=None, status_text=None):
    """
    Unified function to save embeddings for all media file types
    Returns the database object and success status
    """
    if not uploaded_files:
        return None, False

    try:
        # Show progress
        if status_text:
            status_text.text(f"Processing {len(uploaded_files)} {file_type.lower()}...")

        # Process files based on type
        all_chunks = []

        for i, file in enumerate(uploaded_files):
            if progress_bar:
                progress_bar.progress((i + 1) / len(uploaded_files))

            try:
                # Get file extension
                file_extension = file.name.split('.')[-1].lower()

                # Load and process file based on type
                if file_extension in ['txt']:
                    # Text files
                    content = file.read().decode('utf-8')
                    chunks = [Document(page_content=content, metadata={"source": file.name})]
                    all_chunks.extend(chunks)

                elif file_extension in ['ppt', 'pptx']:
                    # PowerPoint files
                    temp_path = f"temp_{file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(file.getbuffer())

                    loader = UnstructuredPowerPointLoader(temp_path)
                    docs = loader.load()
                    all_chunks.extend(docs)
                    os.remove(temp_path)

                elif file_extension in ['docx']:
                    # Word documents
                    temp_path = f"temp_{file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(file.getbuffer())

                    loader = UnstructuredWordDocumentLoader(temp_path)
                    docs = loader.load()
                    all_chunks.extend(docs)
                    os.remove(temp_path)

                elif file_extension in ['xml']:
                    # XML files
                    content = file.read().decode('utf-8')
                    chunks = [Document(page_content=content, metadata={"source": file.name})]
                    all_chunks.extend(chunks)

                elif file_extension in ['json']:
                    # JSON files
                    content = file.read().decode('utf-8')
                    chunks = [Document(page_content=content, metadata={"source": file.name})]
                    all_chunks.extend(chunks)

                elif file_extension in ['yaml', 'yml']:
                    # YAML files
                    content = file.read().decode('utf-8')
                    chunks = [Document(page_content=content, metadata={"source": file.name})]
                    all_chunks.extend(chunks)

                elif file_extension in ['h', 'hpp', 'cpp', 'c']:
                    # Header/C++ files
                    content = file.read().decode('utf-8')
                    chunks = [Document(page_content=content, metadata={"source": file.name})]
                    all_chunks.extend(chunks)

                elif file_extension in ['dat']:
                    # DAT files
                    content = file.read().decode('utf-8')
                    chunks = [Document(page_content=content, metadata={"source": file.name})]
                    all_chunks.extend(chunks)

                elif file_extension in ['md', 'markdown']:
                    # Markdown files
                    content = file.read().decode('utf-8')
                    chunks = [Document(page_content=content, metadata={"source": file.name})]
                    all_chunks.extend(chunks)

                else:
                    # Generic text processing for unknown types
                    try:
                        content = file.read().decode('utf-8')
                        chunks = [Document(page_content=content, metadata={"source": file.name})]
                        all_chunks.extend(chunks)
                    except:
                        st.warning(f"Could not process {file.name} - unsupported format")
                        continue

            except Exception as e:
                st.warning(f"Error processing {file.name}: {str(e)}")
                continue

        if not all_chunks:
            st.error("No content could be extracted from the uploaded files.")
            return None, False

        # Create embeddings
        if status_text:
            status_text.text("Creating embeddings...")

        # Create FAISS index
        db = FAISS.from_documents(all_chunks, embedding_model)

        # Save to disk
        if status_text:
            status_text.text("Saving to vector database...")

        db.save_local(f"./{combined_db_name}")

        # Store in session state
        if 'db_initialized' not in st.session_state:
            st.session_state['db_initialized'] = {}
        st.session_state['db_initialized'][combined_db_name] = True

        if status_text:
            status_text.text("✅ Processing complete!")

        return db, True

    except Exception as e:
        st.error(f"Error processing {file_type}: {str(e)}")
        return None, False

def load_media_embeddings(file_type, group_id=None):
    """
    Unified function to load saved embeddings for media file types
    Returns the database object and metadata
    """
    try:
        if group_id:
            # Load specific group
            metadata = load_file_group(group_id, file_type)
            if metadata and 'db' in metadata:
                return metadata['db'], metadata
            return None, None
        else:
            # Load from session state
            if file_type in st.session_state.get('saved_files', {}):
                # Get the most recent group
                groups = st.session_state['saved_files'][file_type]
                if groups:
                    latest_group_id = max(groups.keys(), key=lambda x: groups[x]['created_at'])
                    return load_media_embeddings(file_type, latest_group_id)
            return None, None
    except Exception as e:
        st.error(f"Error loading {file_type} embeddings: {str(e)}")
        return None, None

def load_saved_files():
    """Load all saved files from storage directory"""
    storage_dir = ensure_storage_directory()
    saved_files = {}

    # Scan for metadata files
    for filename in os.listdir(storage_dir):
        if filename.endswith('_metadata.json'):
            try:
                with open(os.path.join(storage_dir, filename), 'r') as f:
                    metadata = json.load(f)

                file_type = metadata['file_type']
                group_id = metadata['group_id']

                if file_type not in saved_files:
                    saved_files[file_type] = {}

                saved_files[file_type][group_id] = metadata
            except Exception as e:
                print(f"Error loading metadata file {filename}: {e}")
                continue

    st.session_state['saved_files'] = saved_files
    return saved_files

def render_file_management_sidebar():
    """Render the file management sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Saved Files")

    # Load saved files if not already loaded
    if not st.session_state['saved_files']:
        load_saved_files()

    saved_files = st.session_state['saved_files']

    if not any(saved_files.values()):
        st.sidebar.info("No saved files yet. Upload and process files to see them here.")
        return

    # File type tabs
    file_types = list(saved_files.keys())
    if file_types:
        selected_tab = st.sidebar.selectbox("File Type", file_types, key="file_type_tab")

        if selected_tab in saved_files and saved_files[selected_tab]:
            st.sidebar.markdown(f"**{selected_tab} Files:**")

            # Show groups as collapsible folders
            for group_id, metadata in saved_files[selected_tab].items():
                group_name = metadata['group_name']
                file_count = len(metadata['files'])
                created_at = datetime.fromisoformat(metadata['created_at']).strftime('%Y-%m-%d %H:%M')

                if file_count == 1:
                    display_name = group_name
                else:
                    display_name = f"{group_name} ({file_count} files)"

                with st.sidebar.expander(f"📁 {display_name}", expanded=False):
                    st.caption(f"Created: {created_at}")

                    # List a few files inside the folder
                    try:
                        file_names = [f.get('name', 'unknown') for f in metadata.get('files', [])]
                        if file_names:
                            preview = "\n".join([f"- {n}" for n in file_names[:5]])
                            if len(file_names) > 5:
                                preview += f"\n- ... (+{len(file_names)-5} more)"
                            st.markdown(preview)
                    except Exception:
                        pass

                    col_sel, col_del = st.columns([3, 1])
                    with col_sel:
                        if st.button("Open", key=f"select_{group_id}", use_container_width=True):
                            st.session_state['selected_file_group'] = group_id
                            st.session_state['selected_file_type'] = selected_tab
                            st.rerun()
                    with col_del:
                        if st.button("🗑️", key=f"delete_{group_id}", help="Delete this file group"):
                            if delete_file_group(group_id, selected_tab):
                                st.success("File group deleted!")
                                st.rerun()
                            else:
                                st.error("Failed to delete file group")

        # Clear selection button
        if st.session_state.get('selected_file_group'):
            if st.sidebar.button("❌ Clear Selection", use_container_width=True):
                st.session_state['selected_file_group'] = None
                st.session_state['selected_file_type'] = None
                st.rerun()

    # Show current selection
    if st.session_state.get('selected_file_group'):
        selected_type = st.session_state.get('selected_file_type', 'Unknown')
        st.sidebar.success(f"Selected: {selected_type} file group")

def get_typeofpdf():
    st.sidebar.header("Select type of Media", divider='orange')

    media_options = [
        "Text Prompt",
        "URL Search",
        "Agent Web Search",  # Dedicated agent page
        "PDF files",
        "CSV files",
        "Excel files",  # Added Excel files as a separate option
        "Images",
        "Generate Images",
        "Video, mp4 file",
        "Audio files",
        "TXT files",
        "PPT files",
        "DOCX files",
        "XML files",
        "JSON files",
        "H files",
        "DAT files",      # NEW
        "Markdown files",      # NEW
        "Voice to Voice",      # NEW
        "Multiple Media",  # Added new option
    ]

    # Ensure session state has a default value
    if 'media_type' not in st.session_state:
        st.session_state['media_type'] = "Text Prompt"

    selected = st.sidebar.radio(
        "Choose one:",
        media_options,
        key="media_type_radio",
        index=media_options.index(st.session_state['media_type']),
        on_change=update_media_type
    )

    return st.session_state['media_type']




def get_llminfo():
    st.sidebar.header("Options", divider='rainbow')
    tip1="Select a Gemini model you want to use."
    model = st.sidebar.radio("Choose LLM:",
                           ("gemini-2.0-flash",
                            "gemini-2.5-flash",
                            "gemini-2.5-flash-lite",
                            "gemini-2.5-pro",
                            "gemini-3-pro-preview"),
                           help=tip1,
                           key="model_select",
                           on_change=lambda: setattr(st.session_state, 'selected_model', st.session_state.model_select))
    tip2="Lower temperatures are good for prompts that require a less open-ended or creative response, while higher temperatures can lead to more diverse or creative results. A temperature of 0 means that the highest probability tokens are always selected."
    temp = st.sidebar.slider("Temperature:", min_value=0.0,
                           max_value=2.0, value=1.0, step=0.25, help=tip2)
    tip3="Used for nucleus sampling. Specify a lower value for less random responses and a higher value for more random responses."
    topp = st.sidebar.slider("Top P:", min_value=0.0,
                           max_value=1.0, value=0.94, step=0.01, help=tip3)
    tip4="Number of response tokens, 8194 is limit."
    maxtokens = st.sidebar.slider("Maximum Tokens:", min_value=100,
                                max_value=9000, value=8000, step=100, help=tip4)

    # Add Agent toggle to sidebar
    tip5="Enable CrewAI agent for web search and enhanced Q&A capabilities. When enabled, the agent will search the web and provide comprehensive answers with references."
    use_agent = st.sidebar.checkbox("Use Agent (CrewAI)", value=False, help=tip5, key="sidebar_use_agent")

    # Store agent state for main area
    if use_agent:
        st.session_state['agent_enabled'] = True
    else:
        st.session_state['agent_enabled'] = False

    return model, temp, topp, maxtokens, use_agent


def delete_files_in_directory(directory_path):
   try:
     files = os.listdir(directory_path)
     for file in files:
       file_path = os.path.join(directory_path, file)
       if os.path.isfile(file_path):
         os.remove(file_path)
   except OSError:
     print("Error occurred while deleting files.")

def get_confidence_scores(response):
    """
    Extract and process confidence scores from the Gemini API response.
    Returns both individual scores and average, or None if no scores found.
    """
    try:
        # Initialize variables
        confidence_scores = []

        # Get the response text
        if hasattr(response, 'text'):
            # For newer versions of google-generativeai
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'safety_ratings'):
                        for rating in candidate.safety_ratings:
                            if hasattr(rating, 'probability'):
                                confidence_scores.append(rating.probability)
            # For older versions
            elif hasattr(response, 'prompt_feedback'):
                if hasattr(response.prompt_feedback, 'safety_ratings'):
                    for rating in response.prompt_feedback.safety_ratings:
                        if hasattr(rating, 'probability'):
                            confidence_scores.append(rating.probability)

        # Calculate average if scores exist
        if confidence_scores:
            average_score = sum(confidence_scores) / len(confidence_scores)
            return {
                'individual_scores': confidence_scores,
                'average_score': average_score
            }
        else:
            return {
                'individual_scores': [0.0],
                'average_score': 0.0
            }

    except Exception as e:
        logger.warning(f"Error processing confidence scores: {str(e)}")
        return {
            'individual_scores': [0.0],
            'average_score': 0.0
        }


def save_query_response_to_csv(user_query, bot_response, query_type, model_used=None, temperature=None, additional_metadata=None):
    """
    Save user query and bot response to a CSV file for record keeping with consistent column structure.

    Args:
        user_query (str): The user's input query
        bot_response (str): The generated response from the bot
        query_type (str): Type of query (e.g., 'Text Prompt', 'RAG Query', 'Agent Search', 'Voice', etc.)
        model_used (str): The model used to generate the response
        temperature (float): Temperature setting used
        additional_metadata (dict): Any additional metadata to save
    """
    try:
        # Create the CSV filename
        csv_filename = "query_response_log.csv"

        # Prepare the row data
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Clean the response text (properly escape for CSV)
        clean_response = str(bot_response).replace('"', '""').replace('\n', ' ').replace('\r', ' ').strip()
        clean_query = str(user_query).replace('"', '""').replace('\n', ' ').replace('\r', ' ').strip()

        # Define standard metadata fields that all media types will use
        standard_metadata = {
            'interaction_mode': 'N/A',
            'prompt_template': 'N/A',
            'role': 'N/A',
            'num_files': 'N/A',
            'file_names': 'N/A',
            'file_types': 'N/A',
            'similarity_score': 'N/A',
            'used_chunks': 'N/A',
            'reasoning': 'N/A',
            'references': 'N/A',
            'url': 'N/A',
            'visualization_enabled': 'N/A',
            'transcription_method': 'N/A',
            'tts_enabled': 'N/A'
        }

        # Update standard metadata with provided values
        if additional_metadata:
            for key, value in additional_metadata.items():
                if key in standard_metadata:
                    # Properly format lists and complex data
                    if isinstance(value, list):
                        standard_metadata[key] = str(value).replace('"', '""')
                    else:
                        standard_metadata[key] = str(value).replace('"', '""') if value is not None else 'N/A'

        # Define consistent column order
        fieldnames = [
            'timestamp',
            'user_query',
            'bot_response',
            'query_type',
            'model_used',
            'temperature',
            'session_id',
            'metadata_interaction_mode',
            'metadata_prompt_template',
            'metadata_role',
            'metadata_num_files',
            'metadata_file_names',
            'metadata_file_types',
            'metadata_similarity_score',
            'metadata_used_chunks',
            'metadata_reasoning',
            'metadata_references',
            'metadata_url',
            'metadata_visualization_enabled',
            'metadata_transcription_method',
            'metadata_tts_enabled'
        ]

        # Create row data with consistent structure
        row_data = {
            'timestamp': timestamp,
            'user_query': clean_query,
            'bot_response': clean_response,
            'query_type': query_type,
            'model_used': model_used or 'Unknown',
            'temperature': temperature or 'N/A',
            'session_id': st.session_state.get('session_id', 'Unknown'),
            'metadata_interaction_mode': standard_metadata['interaction_mode'],
            'metadata_prompt_template': standard_metadata['prompt_template'],
            'metadata_role': standard_metadata['role'],
            'metadata_num_files': standard_metadata['num_files'],
            'metadata_file_names': standard_metadata['file_names'],
            'metadata_file_types': standard_metadata['file_types'],
            'metadata_similarity_score': standard_metadata['similarity_score'],
            'metadata_used_chunks': standard_metadata['used_chunks'],
            'metadata_reasoning': standard_metadata['reasoning'],
            'metadata_references': standard_metadata['references'],
            'metadata_url': standard_metadata['url'],
            'metadata_visualization_enabled': standard_metadata['visualization_enabled'],
            'metadata_transcription_method': standard_metadata['transcription_method'],
            'metadata_tts_enabled': standard_metadata['tts_enabled']
        }

        # Check if file exists to determine if we need to write headers
        file_exists = os.path.exists(csv_filename)

        # Write to CSV with consistent structure
        with open(csv_filename, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header if file is new
            if not file_exists:
                writer.writeheader()

            writer.writerow(row_data)

        logger.info(f"Successfully saved query-response pair to {csv_filename}")

    except Exception as e:
        logger.error(f"Error saving to CSV: {e}")
        # Don't raise the exception to avoid breaking the main flow
        st.error(f"Warning: Could not save query-response log: {e}")


def load_and_embed_csv(file_paths, combined_db_name):
    """
    Loads multiple CSV files, splits text into chunks, generates embeddings,
    and stores them in a FAISS index.
    """
    all_chunks = []

    # Create progress tracking elements
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_files = len(file_paths)

    for i, file_path in enumerate(file_paths):
        # Update progress
        progress = (i + 1) / total_files
        progress_bar.progress(progress)
        status_text.text(f"Processing file {i+1} of {total_files}: {os.path.basename(file_path)}")

        if not os.path.exists(file_path):
            logger.warning(f"Skipping: File not found - {file_path}")
            continue

        try:
            loader = CSVLoader(file_path)
            pages = loader.load_and_split()
            # Use RecursiveCharacterTextSplitter for better control of chunk size
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len,
                is_separator_regex=False,
            )
            chunks = text_splitter.split_documents(pages)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Error processing CSV file {file_path}: {str(e)}")
            st.warning(f"Could not process CSV file {os.path.basename(file_path)}: {str(e)}")

    if not all_chunks:
        status_text.text("No valid documents were processed.")
        st.warning("No valid documents were processed.")
        return

    try:
        # Update status for embedding creation
        status_text.text("Creating embeddings...")
        embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=SecretStr(GOOGLE_API_KEY), model="models/embedding-001")

        # Create FAISS index and save locally
        status_text.text("Saving to vector database...")
        db = FAISS.from_documents(all_chunks, embedding_model)
        db.save_local(f"./{combined_db_name}")

        # Store in session state that this DB has been initialized
        st.session_state['db_initialized'][combined_db_name] = True

        # Update final status
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        st.success(f"Embedding and storage completed successfully for {len(file_paths)} CSV files.")

        return db
    except Exception as e:
        logger.error(f"Error creating vector database: {str(e)}")
        st.error(f"Error creating vector database: {str(e)}")
        return None



def get_loader(file_path, extension):
    """Returns the appropriate loader based on file extension."""
    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": UnstructuredFileLoader,
        ".ppt": UnstructuredPowerPointLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".csv": CSVLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".xml": UnstructuredXMLLoader,
        ".h": UnstructuredFileLoader,
        ".dat": UnstructuredFileLoader,
        ".md": UnstructuredFileLoader,
    }
    if extension == ".json":
        def extract_data(record):
            json_str = json.dumps(record, ensure_ascii=False)
            return json_str
        return JSONLoader(
            file_path=file_path,
            jq_schema=".",
            content_key=None,
            text_content=False,
            json_lines=False,
        )
    elif extension == ".yaml" or extension == ".yml":
        # Read YAML file and convert to string for embedding
        with open(file_path, 'r', encoding='utf-8') as f:
            yaml_content = yaml.safe_load(f)
            yaml_str = yaml.dump(yaml_content, default_flow_style=False, allow_unicode=True)
        from langchain_core.documents import Document
        return [Document(page_content=yaml_str, metadata={"source": file_path})]
    elif extension == ".dat":
        # Read DAT file as plain text for embedding
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            dat_content = f.read()
        from langchain_core.documents import Document
        return [Document(page_content=dat_content, metadata={"source": file_path})]
    loader_cls = loaders.get(extension, None)
    if loader_cls is None:
        return None
    return loader_cls(file_path)



def embed_and_store_files(uploaded_files, embedding_model, combined_db_name):
    """Processes uploaded files, extracts text chunks, embeds them, and stores in FAISS."""
    all_chunks = []

    # Debugging: Check type of uploaded_files
    if not isinstance(uploaded_files, list) or not all(hasattr(file, 'name') for file in uploaded_files):
        raise ValueError(f"Expected a list of UploadedFile objects, got: {type(uploaded_files)}")

    # Create progress tracking elements
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_files = len(uploaded_files)

    # Process files even if the directory exists - FAISS can be recreated
    for i, file in enumerate(uploaded_files):
        # Update progress
        progress = (i + 1) / total_files
        progress_bar.progress(progress)
        status_text.text(f"Processing file {i+1} of {total_files}: {file.name}")

        # Truncate filename for filesystem usage
        original_name = file.name  # Access name attribute safely
        # Always extract the extension regardless of filename length
        name, ext = os.path.splitext(original_name)
        if len(original_name) > 50:
            truncated_name = name[:50 - len(ext)] + ext
        else:
            truncated_name = original_name

        # Save uploaded file temporarily with truncated name
        with open(truncated_name, "wb") as f:
            f.write(file.read())

        # Load and split document
        loader = get_loader(truncated_name, ext)
        if loader:
            if isinstance(loader, list):  # For YAML loader
                chunks = loader
            else:
                pages = loader.load_and_split()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50,
                    length_function=len,
                    is_separator_regex=False,
                )
                chunks = text_splitter.split_documents(pages)
            all_chunks.extend(chunks)

        # Remove temporary file
        os.remove(truncated_name)

    if not all_chunks:
        status_text.text("No documents were processed.")
        st.warning("No documents were processed. Please check file formats.")
        return None

    BATCH_SIZE = 200      # lower = fewer 429s
    MAX_RETRIES = 5      # simple retry
    BASE_SLEEP = 15       # seconds; doubles on each retry
    def _rate_limited(err: Exception) -> bool:
        msg = str(err).lower()
        return "429" in msg or "rate limit" in msg or "too many requests" in msg
    try:
        status_text.text("Creating embeddings (batched)…")
        total = len(all_chunks)
        if total == 0:
            raise ValueError("No chunks to embed.")

        db = None
        processed = 0

        for start in range(0, total, BATCH_SIZE):
            batch = all_chunks[start:start + BATCH_SIZE]

            sleep_s = BASE_SLEEP
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    if db is None:
                        db = FAISS.from_documents(batch, embedding_model)
                    else:
                        db.add_documents(batch)
                    break  # success
                except Exception as e:
                    if attempt < MAX_RETRIES and _rate_limited(e):
                        time.sleep(sleep_s)
                        sleep_s *= 2  # simple backoff
                    else:
                        raise

            processed = start + len(batch)
            progress_bar.progress(min(0.9 * processed / total, 0.9))
            status_text.text(f"Indexing... ({processed}/{total})")

        status_text.text("Saving to vector database…")
        db.save_local(f"./{combined_db_name}")

        st.session_state.setdefault('db_initialized', {})
        st.session_state['db_initialized'][combined_db_name] = True

        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        st.success(f"Embedding and storage completed successfully for {len(uploaded_files)} files.")
        return db
    except Exception as e:
        logger.error(f"Error creating FAISS index: {e}")
        st.error(f"Error creating vector database: {str(e)}")
        return None

def fetch_webpage_content(url):
    """Fetch and extract text content from a webpage."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text content
        text = soup.get_text()

        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        return text
    except Exception as e:
        logger.error(f"Error fetching webpage content: {str(e)}")
        return None




def process_multiple_media_files(uploaded_files, model, temperature, top_p, max_tokens, progress_bar, status_text):
    """Process multiple types of media files and generate a combined response."""
    all_content_parts = []
    file_info = []  # Store file information separately
    pdf_buffers = []  # Collect raw PDFs for FAISS RAG like the PDF tab

    # Process files in smaller batches to prevent timeout
    batch_size = 4  # Reduced batch size
    total_files = len(uploaded_files)

    # Create progress tracking elements
    # progress_bar = st.progress(0)  <- This will be passed in
    # status_text = st.empty() <- This will be passed in

    # Group files by type for more efficient processing
    file_groups = {
        'text': [],  # For PDF, DOCX, TXT, etc.
        'media': []  # For images, videos, audio
    }

    for file in uploaded_files:
        ext = os.path.splitext(file.name)[1].lower()
        if ext in ['.pdf', '.docx', '.txt', '.pptx', '.xml', '.json', '.yaml', '.yml', '.h', '.csv', '.xlsx', '.xls', '.md', '.dat']:
            file_groups['text'].append(file)
        elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.mp4', '.wav']:
            file_groups['media'].append(file)

    # Process media files first (direct processing)
    for file in file_groups['media']:
        try:
            ext = os.path.splitext(file.name)[1].lower()
            file_info.append((file.name, ext[1:] if ext.startswith('.') else ext))

            # Read file content
            # Ensure file pointer is at the beginning before reading
            if hasattr(file, 'seek'):
                try:
                    file.seek(0)
                except Exception:
                    pass
            file_bytes = file.read()

            # Create content part with appropriate MIME type
            content_part = {
                "mime_type": file.type,
                "data": file_bytes
            }
            all_content_parts.append(content_part)

        except Exception as e:
            logger.error(f"Error processing media file {file.name}: {str(e)}")
            st.warning(f"Could not process media file {file.name}: {str(e)}")

    # Process text-based files in batches
    total_text_files = len(file_groups['text'])
    for batch_start in range(0, total_text_files, batch_size):
        batch_end = min(batch_start + batch_size, total_text_files)
        batch_files = file_groups['text'][batch_start:batch_end]

        # Update progress for this batch
        batch_progress = batch_start / total_text_files
        progress_bar.progress(batch_progress)
        status_text.text(f"Processing text files batch {batch_start//batch_size + 1} of {(total_text_files + batch_size - 1)//batch_size}...")

        for file in batch_files:
            try:
                ext = os.path.splitext(file.name)[1].lower()
                file_info.append((file.name, ext[1:] if ext.startswith('.') else ext))

                # Read file content
                # Ensure file pointer is at the beginning before reading
                if hasattr(file, 'seek'):
                    try:
                        file.seek(0)
                    except Exception:
                        pass
                file_bytes = file.read()

                if ext == '.pdf':
                    # Convert PDF to text excerpts to avoid page-limit errors
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                            temp_file.write(file_bytes)
                            temp_pdf_path = temp_file.name

                        loader = PyPDFLoader(temp_pdf_path)
                        pages = loader.load()

                        # Cap pages and total characters to keep within model limits
                        max_pages = 200  # hard cap pages extracted
                        max_chars = 100_000  # hard cap characters appended
                        selected_pages = pages[:max_pages]
                        combined_text_parts = []
                        current_len = 0
                        for p in selected_pages:
                            txt = p.page_content or ""
                            if not txt:
                                continue
                            remaining = max_chars - current_len
                            if remaining <= 0:
                                break
                            snippet = txt[:remaining]
                            combined_text_parts.append(snippet)
                            current_len += len(snippet)

                        pdf_text_excerpt = "\n\n".join(combined_text_parts).strip()
                        if not pdf_text_excerpt:
                            pdf_text_excerpt = "(No extractable text from PDF)"

                        # Add filename context header
                        header = f"PDF: {file.name} (up to {min(len(pages), max_pages)} pages, {min(current_len, max_chars)} chars)\n\n"
                        all_content_parts.append(header + pdf_text_excerpt)

                        # Also collect raw PDF for building FAISS DB akin to PDF tab
                        pdf_buffers.append((file.name, file_bytes))

                        try:
                            os.unlink(temp_pdf_path)
                        except Exception:
                            pass
                    except Exception as pdf_proc_err:
                        logger.error(f"Error extracting text from PDF {file.name}: {str(pdf_proc_err)}")
                        st.warning(f"Could not extract text from PDF {file.name}: {str(pdf_proc_err)}")

                elif ext == '.csv':
                    # Process CSV files
                    try:
                        # Create a temporary file to process the CSV
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
                            temp_file.write(file_bytes)
                            temp_file_path = temp_file.name

                        # Read CSV with pandas
                        df = pd.read_csv(temp_file_path)

                        # Convert DataFrame to string representation
                        csv_text = df.to_string()
                        all_content_parts.append(csv_text)

                        # Clean up temporary file
                        os.unlink(temp_file_path)

                    except Exception as csv_error:
                        logger.error(f"Error processing CSV file {file.name}: {str(csv_error)}")
                        st.warning(f"Could not process CSV file {file.name}: {str(csv_error)}")

                elif ext in ['.xlsx', '.xls']:
                    # Process Excel files
                    try:
                        # Create a temporary file to process the Excel file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                            temp_file.write(file_bytes)
                            temp_file_path = temp_file.name

                        # Read Excel with pandas
                        df = pd.read_excel(temp_file_path)

                        # Create a more structured representation of the Excel data
                        # Include sheet name, column headers, and preserve table structure
                        excel_text = f"Filename: {file.name}\n\n"

                        # Handle multiple sheets
                        if isinstance(df, dict):  # Multiple sheets
                            for sheet_name, sheet_df in df.items():
                                excel_text += f"\nSheet: {sheet_name}\n"
                                excel_text += f"Columns: {', '.join(sheet_df.columns.astype(str))}\n"
                                excel_text += f"Rows: {len(sheet_df)}\n\n"
                                excel_text += sheet_df.to_string(index=True)
                                excel_text += "\n\n"
                        else:  # Single sheet
                            excel_text += f"Sheet: Sheet1\n"
                            excel_text += f"Columns: {', '.join(df.columns.astype(str))}\n"
                            excel_text += f"Rows: {len(df)}\n\n"
                            excel_text += df.to_string(index=True)

                        all_content_parts.append(excel_text)

                        # Clean up temporary file
                        os.unlink(temp_file_path)
                    except Exception as excel_error:
                        logger.error(f"Error processing Excel file {file.name}: {str(excel_error)}")
                        st.warning(f"Could not process Excel file {file.name}: {str(excel_error)}")

                elif ext == '.docx':
                    try:
                        # Create a temporary file to process the DOCX
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
                            temp_file.write(file_bytes)
                            temp_file_path = temp_file.name

                        # Use UnstructuredWordDocumentLoader to extract text
                        loader = UnstructuredWordDocumentLoader(temp_file_path)
                        documents = loader.load()

                        # Extract text from documents
                        docx_text = "\n\n".join([doc.page_content for doc in documents])
                        all_content_parts.append(docx_text)

                        # Clean up temporary file
                        os.unlink(temp_file_path)

                    except Exception as docx_error:
                        logger.error(f"Error processing DOCX file {file.name}: {str(docx_error)}")
                        st.warning(f"Could not process DOCX file {file.name}: {str(docx_error)}")

                elif ext in ['.ppt', '.pptx']:
                    try:
                        # Create a temporary file to process the PPT/PPTX
                        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                            temp_file.write(file_bytes)
                            temp_file_path = temp_file.name

                        # Use UnstructuredPowerPointLoader to extract text
                        loader = UnstructuredPowerPointLoader(temp_file_path)
                        documents = loader.load()

                        # Extract text from documents
                        ppt_text = "\n\n".join([doc.page_content for doc in documents])
                        all_content_parts.append(ppt_text)

                        # Clean up temporary file
                        os.unlink(temp_file_path)

                    except Exception as ppt_error:
                        logger.error(f"Error processing PPT/PPTX file {file.name}: {str(ppt_error)}")
                        st.warning(f"Could not process PPT/PPTX file {file.name}: {str(ppt_error)}")

                elif ext == '.txt':
                    # Process text files directly without using UnstructuredFileLoader
                    try:
                        # Decode the bytes to text
                        text_content = file_bytes.decode('utf-8', errors='replace')
                        all_content_parts.append(text_content)
                    except Exception as txt_error:
                        logger.error(f"Error processing TXT file {file.name}: {str(txt_error)}")
                        st.warning(f"Could not process TXT file {file.name}: {str(txt_error)}")

                elif ext == '.md':
                    # Process Markdown files directly
                    try:
                        md_content = file_bytes.decode('utf-8', errors='replace')
                        all_content_parts.append(md_content)
                    except Exception as md_error:
                        logger.error(f"Error processing MD file {file.name}: {str(md_error)}")
                        st.warning(f"Could not process Markdown file {file.name}: {str(md_error)}")

                elif ext == '.xml':
                    # Process XML files directly without using UnstructuredXMLLoader
                    try:
                        # Decode the bytes to text
                        xml_content = file_bytes.decode('utf-8', errors='replace')
                        all_content_parts.append(xml_content)
                    except Exception as xml_error:
                        logger.error(f"Error processing XML file {file.name}: {str(xml_error)}")
                        st.warning(f"Could not process XML file {file.name}: {str(xml_error)}")

                elif ext == '.json':
                    # Process JSON files directly
                    try:
                        # Decode the bytes to text
                        json_content = file_bytes.decode('utf-8', errors='replace')
                        all_content_parts.append(json_content)
                    except Exception as json_error:
                        logger.error(f"Error processing JSON file {file.name}: {str(json_error)}")
                        st.warning(f"Could not process JSON file {file.name}: {str(json_error)}")

                elif ext in ['.yaml', '.yml']:
                    # Process YAML files directly
                    try:
                        # Decode the bytes to text
                        yaml_content = file_bytes.decode('utf-8', errors='replace')
                        all_content_parts.append(yaml_content)
                    except Exception as yaml_error:
                        logger.error(f"Error processing YAML file {file.name}: {str(yaml_error)}")
                        st.warning(f"Could not process YAML file {file.name}: {str(yaml_error)}")

                elif ext == '.h':
                    # Process header files directly
                    try:
                        # Decode the bytes to text
                        header_content = file_bytes.decode('utf-8', errors='replace')
                        all_content_parts.append(header_content)
                    except Exception as h_error:
                        logger.error(f"Error processing header file {file.name}: {str(h_error)}")
                        st.warning(f"Could not process header file {file.name}: {str(h_error)}")

                elif ext == '.dat':
                    # Process DAT files directly as text
                    try:
                        dat_content = file_bytes.decode('utf-8', errors='replace')
                        all_content_parts.append(dat_content)
                    except Exception as dat_error:
                        logger.error(f"Error processing DAT file {file.name}: {str(dat_error)}")
                        st.warning(f"Could not process DAT file {file.name}: {str(dat_error)}")

                else:
                    # For other text-based files, try to extract text content directly
                    try:
                        # Decode the bytes to text
                        text_content = file_bytes.decode('utf-8', errors='replace')
                        all_content_parts.append(text_content)
                    except Exception as e:
                        logger.error(f"Error processing file {file.name}: {str(e)}")
                        st.warning(f"Could not process file {file.name}: {str(e)}")

            except Exception as e:
                logger.error(f"Error processing file {file.name}: {str(e)}")
                st.warning(f"Could not process file {file.name}: {str(e)}")

        # Add a delay between batches to prevent rate limiting
        if batch_end < total_text_files:
            time.sleep(2)

    # If any PDFs were uploaded, mirror PDF tab: index into FAISS and store DB name
    try:
        if pdf_buffers:
            # Create a fingerprint for PDF caching
            pdf_fingerprint = tuple(sorted([name for name, _ in pdf_buffers]))

            # Check if we already have a cached FAISS DB for these PDFs
            if 'mm_pdf_cache' not in st.session_state:
                st.session_state['mm_pdf_cache'] = {}

            cached_db_name = st.session_state['mm_pdf_cache'].get(pdf_fingerprint)
            if cached_db_name and os.path.exists(f"./{cached_db_name}"):
                # Use cached DB
                st.session_state['multi_media_pdf_db'] = cached_db_name
                status_text.text("Using cached PDF index...")
                progress_bar.progress(0.9)
            else:
                # Create new FAISS DB
                status_text.text("Indexing PDFs for retrieval...")
                progress_bar.progress(0.9)

                db_name_list = []
                for name, _ in pdf_buffers:
                    if len(name) > 20:
                        base, ext = os.path.splitext(name)
                        truncated = base[:20 - len(ext)] + ext
                        db_name_list.append(os.path.splitext(truncated)[0])
                    else:
                        db_name_list.append(os.path.splitext(name)[0])
                combined_db_name = "_".join(db_name_list) if db_name_list else "mm_pdf_db"
                if len(combined_db_name) > 50:
                    combined_db_name = combined_db_name[:50]

                all_pdf_chunks = []
                for name, pdf_bytes in pdf_buffers:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tpdf:
                        tpdf.write(pdf_bytes)
                        temp_pdf_path = tpdf.name
                    try:
                        loader = PyPDFLoader(temp_pdf_path)
                        # Align with single-PDF tab: use load_and_split then 500/50 chunking
                        pages = loader.load_and_split()
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=500,
                            chunk_overlap=50,
                            length_function=len,
                            is_separator_regex=False,
                        )
                        chunks = text_splitter.split_documents(pages)
                        all_pdf_chunks.extend(chunks)
                    finally:
                        try:
                            os.unlink(temp_pdf_path)
                        except Exception:
                            pass

                if all_pdf_chunks:
                    embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=SecretStr(GOOGLE_API_KEY), model="models/embedding-001")
                    # Batch processing to avoid rate limits
                    batch_size = 1000 # Documents per batch
                    db = None
                    total_chunks = len(all_pdf_chunks)
                    for i in range(0, total_chunks, batch_size):
                        batch = all_pdf_chunks[i:i+batch_size]
                        if db is None:
                            db = FAISS.from_documents(batch, embedding_model)
                        else:
                            db.add_documents(batch)
                        # Update progress and add a delay
                        progress_val = 0.9 + (0.1 * ((i + len(batch))/total_chunks))
                        progress_bar.progress(progress_val)
                        status_text.text(f"Indexing files for retrieval... ({i+ len(batch)}/{total_chunks}")
                        time.sleep(10) # 60-second delay between batches

                    #db = FAISS.from_documents(all_pdf_chunks, embedding_model)
                    db.save_local(f"./{combined_db_name}")
                    if 'db_initialized' not in st.session_state:
                        st.session_state['db_initialized'] = {}
                    st.session_state['db_initialized'][combined_db_name] = True
                    st.session_state['multi_media_pdf_db'] = combined_db_name
                    # Cache the DB name for these PDFs
                    st.session_state['mm_pdf_cache'][pdf_fingerprint] = combined_db_name
    except Exception as e:
        logger.error(f"Error creating PDF vector database for Multiple Media: {e}")

    # Update final status
    progress_bar.progress(1.0)
    status_text.text("Processing complete!")

    # Show success message only if content was extracted
    if all_content_parts:
        st.success("Files processed successfully! You can now ask questions about the content.")

    return all_content_parts, file_info

def _compute_files_fingerprint(uploaded_files):
    """Create a lightweight fingerprint for a set of UploadedFile objects to avoid reprocessing on rerun."""
    try:
        items = []
        for f in uploaded_files:
            size = getattr(f, 'size', None)
            items.append((f.name, size))
        # Sort to get stable ordering independent of selection order
        items.sort(key=lambda x: x[0])
        return tuple(items)
    except Exception:
        # Fall back to names only
        try:
            names = sorted([f.name for f in uploaded_files])
            return tuple(names)
        except Exception:
            return None

def crewai_web_search(url, question):
    """Perform web search using CrewAI agents and extract references. Uses Gemini model via CrewAI LLM class. Only Gemini API key is supported; Vertex AI is not used."""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Initializing CrewAI web search...")
        progress_bar.progress(0.2)
        from crewai import Agent, Task, Crew, LLM
        from crewai_tools import WebsiteSearchTool
        import os
        # Use only environment variable for GEMINI_API_KEY (never Vertex AI)
        gemini_api_key = os.environ.get("GEMINI_API_KEY") or "AIzaSyAm62k_mpMqcdQn8q3IqCjaOP4bCtE-qzE"
        os.environ["GEMINI_API_KEY"] = gemini_api_key  # Ensure it's set for subprocesses
        # Only use Gemini model string, never Vertex AI
        model_str = "gemini/gemini-1.5-flash"  # or "gemini/gemini-2.0-flash" if you want
        researcher_llm = LLM(api_key=gemini_api_key, model=model_str)
        analyst_llm = LLM(api_key=gemini_api_key, model=model_str)
        if url:
            website_search_tool = WebsiteSearchTool(website_url=url)
            tools = [website_search_tool]  # type: ignore[list]
        else:
            tools = []
        status_text.text("Creating research agent...")
        progress_bar.progress(0.4)
        researcher = Agent(  # type: ignore[list]
            role="Web Researcher",
            goal="Extract and analyze information from websites",
            backstory="You are an expert at finding and extracting relevant information from web content.",
            verbose=True,
            allow_delegation=False,
            tools=tools,  # type: ignore[list]
            llm=researcher_llm
        )
        analyst = Agent(
            role="Content Analyst",
            goal="Analyze information and provide comprehensive answers",
            backstory="You are an expert at analyzing information and providing clear, concise answers.",
            verbose=True,
            allow_delegation=False,
            llm=analyst_llm
        )
        status_text.text("Creating research task...")
        progress_bar.progress(0.6)
        research_task = Task(
            description=f"Research the web to find information about: {question}",
            agent=researcher,
            expected_output="Detailed information extracted from the web relevant to the question."
        )
        analysis_task = Task(
            description=f"Analyze the research results and answer the question: {question}",
            agent=analyst,
            expected_output="A comprehensive answer to the question with reasoning.",
            context=[research_task]
        )
        status_text.text("Assembling and running the crew...")
        progress_bar.progress(0.8)
        crew = Crew(
            agents=[researcher, analyst],
            tasks=[research_task, analysis_task],
            verbose=True
        )
        result = crew.kickoff()
        progress_bar.progress(1.0)
        status_text.text("Web search complete!")
        import re
        url_pattern = r"https?://[\w\.-/\?&=%#]+"
        result_str = str(result)  # Ensure result is a string for regex
        references = re.findall(url_pattern, result_str)
        response_json = {
            "Answer": result_str,
            "References": references if references else None,
            "Used Chunks": "Web search results",
            "Similarity Scores": "N/A for web search",
            "Overall Similarity": "N/A for web search",
            "Reasoning": "Information extracted directly from the web using CrewAI agents."
        }
        return json.dumps(response_json)
    except Exception as e:
        logger.error(f"Error in CrewAI web search: {str(e)}")
        st.error(f"Error in CrewAI web search: {str(e)}")
        return None

def process_url_content(url, combined_db_name):
    """Process URL content and store in FAISS vector store."""
    try:
        # Ensure an event loop exists for this thread (Streamlit workaround)
        ensure_event_loop()  # <--- FIXED HERE
        # Create progress tracking elements
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Update status for fetching content
        status_text.text("Fetching webpage content...")
        progress_bar.progress(0.2)

        # Fetch webpage content
        webpage_content = fetch_webpage_content(url)
        if not webpage_content:
            status_text.text("Failed to fetch webpage content.")
            st.error("Failed to fetch webpage content.")
            return None

        # Update status for text splitting
        status_text.text("Processing webpage content...")
        progress_bar.progress(0.4)

        # Split content into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )

        # Create documents from chunks
        chunks = text_splitter.create_documents([webpage_content])

        # Update status for embedding creation
        status_text.text("Creating embeddings...")
        progress_bar.progress(0.6)

        # Initialize embedding model
        embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=SecretStr(GOOGLE_API_KEY), model="models/embedding-001")

        # Create FAISS index
        db = FAISS.from_documents(chunks, embedding_model)

        # Update status for saving
        status_text.text("Saving to vector database...")
        progress_bar.progress(0.8)

        # Save to disk
        db.save_local(f"./{combined_db_name}")

        # Update final status
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        st.success("Webpage content processed and stored successfully.")

        return db
    except Exception as e:
        logger.error(f"Error processing URL content: {str(e)}")
        st.error(f"Error processing URL content: {str(e)}")
        return None

def visualize_csv_data(csv_content, question, model, temperature, top_p, max_tokens):
    """Generate visualization suggestions and graphs using Gemini."""
    try:
        # Create a prompt for visualization
        visualization_prompt = f"""Based on the following CSV data and question, create visualizations to help answer the question.
For each visualization:
1. Describe what type of chart would be most appropriate
2. Explain why this visualization is helpful
3. Create the actual visualization using Python code with plotly
4. Return the visualization code in a format that can be directly executed

IMPORTANT: The code should use the DataFrame 'df' that is already loaded in the environment. DO NOT try to read the CSV file again.
DO NOT include code like 'df = pd.read_csv("data.csv")' as the DataFrame is already available.

CSV Content (first few rows):
{csv_content[:2000]}

Question: {question}

Please provide the response in the following format:
{{
    "visualizations": [
        {{
            "type": "chart_type",
            "description": "what this visualization shows",
            "reason": "why this visualization is appropriate",
            "code": "complete Python code to create the visualization using plotly. The code should create a figure named 'fig' and should not include any display or show commands. Use the DataFrame 'df' that is already loaded."
        }}
    ],
    "analysis": "analysis of the data",
    "key_insights": ["key insights from the data"]
}}

Return ONLY the JSON without any additional text or formatting. The code should only create the figure, not display it."""

        # Create the model with generation config
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
        )

        model = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config
        )

        # Generate visualization suggestions
        response = model.generate_content(visualization_prompt)

        # Parse the response - clean up any markdown formatting
        json_content = response.text
        if "```json" in json_content:
            json_content = json_content.split("```json")[1].split("```")[0].strip()
        elif "```" in json_content:
            json_content = json_content.split("```")[1].strip()

        visualization_data = json.loads(json_content)

        return visualization_data
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        return None

def create_visualization(viz_data, df):
    """Create a Plotly figure based on visualization data."""
    try:
        viz_type = viz_data['type'].lower()
        data = viz_data['data']

        if viz_type == "line":
            fig = go.Figure(data=go.Scatter(x=data['x'], y=data['y'], mode='lines+markers'))
        elif viz_type == "bar":
            fig = go.Figure(data=go.Bar(x=data['x'], y=data['y']))
        elif viz_type == "scatter":
            fig = go.Figure(data=go.Scatter(x=data['x'], y=data['y'], mode='markers'))
        elif viz_type == "pie":
            fig = go.Figure(data=go.Pie(labels=data['labels'], values=data['y']))
        elif viz_type == "histogram":
            fig = go.Figure(data=go.Histogram(x=data['x']))
        elif viz_type == "box":
            fig = go.Figure(data=go.Box(x=data['x'], y=data['y']))
        else:
            return None

        fig.update_layout(
            title=data['title'],
            xaxis_title=viz_data['columns'][0],
            yaxis_title=viz_data['columns'][1] if len(viz_data['columns']) > 1 else None,
            showlegend=True
        )

        return fig
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        return None

def visualize_pdf_data(pdf_text, question, model, temperature, top_p, max_tokens):
    """Generate visualization suggestions and graphs for PDF content using Gemini."""
    try:
        # Create a prompt for visualization
        visualization_prompt = f"""Based on the following text extracted from PDF documents and the question, create visualizations to help answer the question.
For each visualization:
1. Describe what type of chart would be most appropriate
2. Explain why this visualization is helpful
3. Create the actual visualization using Python code with plotly
4. Return the visualization code in a format that can be directly executed

Text Content (first 2000 characters):
{pdf_text[:2000]}

Question: {question}

IMPORTANT INSTRUCTIONS:
- If the text contains structured data (numbers, statistics, percentages, etc.), create visualizations based on that data.
- If the text doesn't contain structured data, create conceptual visualizations that represent the key concepts, relationships, or processes mentioned in the text.
- NEVER state that you cannot fulfill that request or unable to create graphs. Instead, always provide at least one visualization, even if it's conceptual.
- For conceptual visualizations, use sample data that illustrates the concepts discussed in the text.
- DO NOT include phrases like 'I cannot fulfill that request' or 'I am unable to create graphs' in your response.

Please provide the response in the following format:
{{
    "visualizations": [
        {{
            "type": "chart_type",
            "description": "what this visualization shows",
            "reason": "why this visualization is appropriate",
            "code": "complete Python code to create the visualization using plotly. The code should create a figure named 'fig' and should not include any display or show commands."
        }}
    ],
    "analysis": "analysis of the data",
    "key_insights": ["key insights from the data"]
}}

Return ONLY the JSON without any additional text or formatting. The code should only create the figure, not display it."""

        # Create the model with generation config
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
        )

        model = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config
        )

        # Generate visualization suggestions
        response = model.generate_content(visualization_prompt)

        # Parse the response - clean up any markdown formatting
        json_content = response.text
        if "```json" in json_content:
            json_content = json_content.split("```json")[1].split("```")[0].strip()
        elif "```" in json_content:
            json_content = json_content.split("```")[1].strip()

        # Check if the response contains phrases indicating the model refused to create visualizations
        refusal_phrases = [
            "cannot fulfill that request",
            "unable to create graphs",
            "unable to visualize data",
            "cannot visualize data",
            "I am sorry",
            "I apologize"
        ]

        for phrase in refusal_phrases:
            if phrase in json_content:
                # If the model refused, create a fallback visualization
                logger.warning(f"Model refused to create visualizations. Using fallback visualization.")
                return create_fallback_visualization(pdf_text, question)

        try:
            visualization_data = json.loads(json_content)
            return visualization_data
        except json.JSONDecodeError:
            # If JSON parsing fails, create a fallback visualization
            logger.warning(f"Failed to parse visualization JSON. Using fallback visualization.")
            return create_fallback_visualization(pdf_text, question)

    except Exception as e:
        logger.error(f"Error generating PDF visualizations: {str(e)}")
        return create_fallback_visualization(pdf_text, question)

def create_fallback_visualization(pdf_text, question):
    """Create a fallback visualization when the model refuses to create one."""
    # Extract key terms from the question
    key_terms = question.lower().split()
    key_terms = [term for term in key_terms if len(term) > 3]  # Filter out short words

    # Create a simple bar chart with sample data
    import plotly.graph_objects as go

    # Generate sample data based on key terms
    categories = key_terms[:5] if len(key_terms) >= 5 else key_terms + ["Term 1", "Term 2", "Term 3", "Term 4", "Term 5"][:5-len(key_terms)]
    values = [75, 60, 45, 30, 15][:len(categories)]

    # Create the figure
    fig = go.Figure(data=go.Bar(
        x=categories,
        y=values,
        marker_color='rgb(55, 83, 109)'
    ))

    # Update layout
    fig.update_layout(
        title=f"Conceptual Visualization for: {question[:50]}...",
        xaxis_title="Key Concepts",
        yaxis_title="Relevance Score",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    # Create the visualization data structure
    visualization_data = {
        "visualizations": [
            {
                "type": "bar",
                "description": "Conceptual visualization showing the relevance of key terms from your question",
                "reason": "This visualization helps understand the relative importance of different concepts related to your question",
                "code": f"""
import plotly.graph_objects as go
fig = go.Figure(data=go.Bar(
    x={categories},
    y={values},
    marker_color='rgb(55, 83, 109)'
))
fig.update_layout(
    title="{question[:50]}...",
    xaxis_title="Key Concepts",
    yaxis_title="Relevance Score",
    height=400,
    margin=dict(l=20, r=20, t=40, b=20)
)
"""
            }
        ],
        "analysis": "This is a conceptual visualization based on the key terms in your question. It shows the relative importance of different concepts related to your query.",
        "key_insights": [
            "The visualization represents key concepts from your question",
            "The height of each bar indicates the conceptual relevance",
            "This is a simplified representation to help understand the relationships between concepts"
        ]
    }

    return visualization_data

def download_video_from_gcs(gcs_uri):
    """Downloads a video from a GCS URI and returns its local path."""
    if not VERTEXAI_AVAILABLE:
        st.error("Vertex AI components are not available. Please install google-cloud-aiplatform.")
        return None
    try:
        # Parse the GCS URI
        if not gcs_uri.startswith("gs://"):
            raise ValueError("Invalid GCS URI. Must start with 'gs://'.")

        bucket_name, blob_name = gcs_uri[5:].split("/", 1)

        # Create a client
        storage_client = storage.Client()

        # Get the bucket and blob
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Create a temporary file to download the video
        temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

        # Download the video
        blob.download_to_filename(temp_video_file.name)

        return temp_video_file.name
    except Exception as e:
        logger.error(f"Error downloading video from GCS: {e}")
        st.error(f"Error downloading video from GCS: {e}")
        return None

def load_and_embed_excel(file_paths, combined_db_name):
    """
    Loads multiple Excel files, splits text into chunks, generates embeddings,
    and stores them in a FAISS index.
    """
    all_chunks = []

    # Create progress tracking elements
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_files = len(file_paths)

    for i, file_path in enumerate(file_paths):
        # Update progress
        progress = (i + 1) / total_files
        progress_bar.progress(progress)
        status_text.text(f"Processing file {i+1} of {total_files}: {os.path.basename(file_path)}")

        if not os.path.exists(file_path):
            logger.warning(f"Skipping: File not found - {file_path}")
            continue

        try:
            # Create a Document object for each Excel file
            documents = []

            # Read Excel file with pandas
            excel_data = pd.read_excel(file_path, sheet_name=None)  # None to get all sheets

            # Process each sheet
            if isinstance(excel_data, dict):
                for sheet_name, df in excel_data.items():
                    # Create metadata
                    metadata = {
                        "source": file_path,
                        "sheet": sheet_name,
                        "columns": list(df.columns),
                        "rows": len(df)
                    }

                    # Create a structured text representation
                    content = f"Sheet: {sheet_name}\n"
                    content += f"Columns: {', '.join(df.columns.astype(str))}\n"
                    content += f"Rows: {len(df)}\n\n"
                    content += df.to_string(index=True)

                    # Create a document with page_content and metadata
                    from langchain_core.documents import Document
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
            else:
                # Single sheet as DataFrame
                df = excel_data
                metadata = {
                    "source": file_path,
                    "sheet": "Sheet1",
                    "columns": list(df.columns),
                    "rows": len(df)
                }
                content = f"Sheet: Sheet1\n"
                content += f"Columns: {', '.join(df.columns.astype(str))}\n"
                content += f"Rows: {len(df)}\n\n"
                content += df.to_string(index=True)
                from langchain_core.documents import Document
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)

            # Use RecursiveCharacterTextSplitter for better control of chunk size
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len,
                is_separator_regex=False,
            )
            chunks = text_splitter.split_documents(documents)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Error processing Excel file {file_path}: {str(e)}")
            st.warning(f"Could not process Excel file {file_path}: {str(e)}")

    if not all_chunks:
        status_text.text("No valid documents were processed.")
        st.warning("No valid documents were processed.")
        return

    try:
        # Update status for embedding creation
        status_text.text("Creating embeddings...")
        embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=SecretStr(GOOGLE_API_KEY), model="models/embedding-001")

        # Create FAISS index and save locally
        status_text.text("Saving to vector database...")
        db = FAISS.from_documents(all_chunks, embedding_model)
        db.save_local(f"./{combined_db_name}")

        # Store in session state that this DB has been initialized
        st.session_state['db_initialized'][combined_db_name] = True

        # Update final status
        progress_bar.progress(1.0)
        status_text.text("Processing complete!")
        st.success(f"Embedding and storage completed successfully for {len(file_paths)} Excel files.")

        return db
    except Exception as e:
        logger.error(f"Error creating vector database: {str(e)}")
        st.error(f"Error creating vector database: {str(e)}")
        return None

def main():
    ensure_event_loop()  # <-- Add this line at the very top of main()
    page_setup()
    initialize_session_state()
    sidebar_accordion()

    # Only show file management sidebar for media types that support it
    typepdf = st.session_state.get('media_type', 'Text Prompt')
    excluded_types = [
        "TXT files", "PPT files", "DOCX files", "XML files", "JSON files",
        "YAML files", "H files", "DAT files", "Markdown files",
        "Voice to Voice", "Multiple Media"
    ]
    if typepdf not in excluded_types:
        render_file_management_sidebar()  # Add file management sidebar
    # ... rest of main() logic ...
    model = st.session_state.get('selected_model', 'gemini-2.0-flash')
    temperature = st.session_state.get('temperature', 1.0)
    top_p = st.session_state.get('top_p', 0.94)
    max_tokens = st.session_state.get('max_tokens', 7000)
    use_agent = st.session_state.get('agent_enabled', False)

    # Check if agent query is active and show response
    if use_agent and st.session_state.get('show_agent_response') and st.session_state.get('agent_query'):
        st.header("🤖 Agent Response", divider="blue")

        # Display the query
        st.subheader("Your Question:")
        st.info(st.session_state['agent_query'])

        # Process the agent query
        with st.spinner("🤖 Agent is searching the web and analyzing information..."):
            response_json = crewai_web_search(None, st.session_state['agent_query'])
            if response_json:
                parsed_data = json.loads(response_json)
                answer = parsed_data.get("Answer", "No answer found.")
                references = parsed_data.get("References")
                reasoning = parsed_data.get("Reasoning", "N/A")

                # Display the response
                st.subheader("Agent Answer:")
                st.success(answer)

                if references:
                    st.markdown("**📚 References:**")
                    for ref in references:
                        st.markdown(f"- [{ref}]({ref})")

                if reasoning:
                    st.markdown(f"**🤔 Reasoning:** {reasoning}")

                # Add to agent chat history if it exists
                if 'agent_chat_history' not in st.session_state:
                    st.session_state.agent_chat_history = []

                st.session_state.agent_chat_history.append({
                    'role': 'user',
                    'content': st.session_state['agent_query']
                })

                st.session_state.agent_chat_history.append({
                    'role': 'agent',
                    'content': answer,
                    'references': references,
                    'reasoning': reasoning
                })

                # Save to CSV log
                save_query_response_to_csv(
                    user_query=st.session_state['agent_query'],
                    bot_response=answer,
                    query_type="Agent Web Search",
                    model_used="CrewAI Agent",
                    additional_metadata={
                        'references': str(references) if references else None,
                        'reasoning': reasoning
                    }
                )

                # Clear the query after processing
                del st.session_state['agent_query']
                del st.session_state['show_agent_response']

                # Add a button to continue with agent
                if st.button("💬 Continue with Agent", type="primary", key="continue_agent_chat"):
                    st.session_state['media_type'] = "Agent Web Search"
                    st.rerun()
            else:
                st.error("❌ Agent failed to generate a response. Please try again.")
                # Clear the query after error
                del st.session_state['agent_query']
                del st.session_state['show_agent_response']

        st.markdown("---")

    # Show agent query interface in main area when agent is enabled
    elif st.session_state.get('agent_enabled') and not st.session_state.get('show_agent_response'):
        # --- CrewAI Agent Web Search UI ---
        # Show a blue header and intro message above the chat
        st.markdown('<span style="color:#1976d2;font-size:1.5rem;font-weight:700;">🤖 AI Agent Web Search</span>', unsafe_allow_html=True)
        st.markdown('<div style="color:#1976d2;font-size:1.05rem;margin-bottom:1.2rem;">Start a conversation by asking a research question below.</div>', unsafe_allow_html=True)
        chat_container = st.container()
        with chat_container:
            if st.session_state.agent_chat_history:
                for message in st.session_state.agent_chat_history:
                    if message['role'] == 'user':
                        st.markdown(f'<div class="agent-chat-bubble user">👤 {message["content"]}</div>', unsafe_allow_html=True)
                    elif message['role'] == 'agent':
                        st.markdown(f'<div class="agent-chat-bubble agent">🤖 {message["content"]}</div>', unsafe_allow_html=True)
                        if message.get('references'):
                            refs = ''.join([f'<div><a href="{ref}" target="_blank">🔗 {ref}</a></div>' for ref in message['references']])
                            st.markdown(f'<div class="agent-chat-bubble references"><b>📚 References:</b><br>{refs}</div>', unsafe_allow_html=True)
                        if message.get('reasoning'):
                            st.markdown(f'<div class="agent-chat-bubble reasoning"><b>🤔 Reasoning:</b> {message["reasoning"]}</div>', unsafe_allow_html=True)
            else:
                pass
        # Input row
        st.markdown('<hr style="margin:2.2rem 0 1.2rem 0;border:0;border-top:1.5px solid #e3e8f7;">', unsafe_allow_html=True)
        agent_question = st.text_area("Enter your research question:", "", key="agent_question_input", label_visibility="collapsed", placeholder="e.g. What are the latest developments in quantum computing?",)
        col_send, col_clear = st.columns([2,1])
        with col_send:
            if st.button("🚀 Send", key="agent_send_btn", use_container_width=True):
                if agent_question:
                    st.session_state.agent_chat_history.append({'role': 'user', 'content': agent_question})
                    with st.spinner("🔍 Conducting web research..."):
                        response_json = crewai_web_search(None, agent_question)
                        if response_json:
                            parsed_data = json.loads(response_json)
                            answer = parsed_data.get("Answer", "No answer found.")
                            references = parsed_data.get("References")
                            reasoning = parsed_data.get("Reasoning", "N/A")
                            st.session_state.agent_chat_history.append({
                                'role': 'agent',
                                'content': answer,
                                'references': references,
                                'reasoning': reasoning
                            })

                            # Save to CSV log
                            save_query_response_to_csv(
                                user_query=agent_question,
                                bot_response=answer,
                                query_type="Agent Web Search Chat",
                                model_used="CrewAI Agent",
                                additional_metadata={
                                    'references': str(references) if references else None,
                                    'reasoning': reasoning
                                }
                            )
                        st.rerun()
                else:
                    st.warning("⚠️ Please enter a question.")
        with col_clear:
            if st.button("🗑️ Clear Chat", key="agent_clear_btn", use_container_width=True):
                st.session_state.agent_chat_history = []
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    elif typepdf == "Text Prompt":
        st.header("💬 Text Prompt Engineering", divider="blue")

        # --- Interaction mode selection ---
        interaction_mode = st.radio(
            "Choose interaction mode:",
            ["Text", "Voice"],
            horizontal=True,
            key="prompt_interaction_mode"
        )

        if interaction_mode == "Voice":
            # --- Voice Input (Improved UI) ---
            with st.container(border=True):
                st.subheader("🎙️ Voice Recorder", anchor=False)
                st.caption("Tap the microphone, speak your question, then tap again to stop. The audio will be transcribed automatically.")
                audio_bytes_voice = audio_recorder(
                    text="🎤 Tap to Record / Stop",
                    icon_name="microphone",
                    icon_size="2x",
                    neutral_color="#6C757D",
                    recording_color="#FF4B4B",
                    key="voice_prompt_rec",
                )
            voice_text = ""
            transcription_error = False
            if audio_bytes_voice:
                # Playback preview
                st.audio(audio_bytes_voice, format='audio/wav', start_time=0)
                if sr is not None and AudioSegment is not None:
                    with st.spinner("🎧 Transcribing..."):
                        try:
                            audio_file_voice = io.BytesIO(audio_bytes_voice)
                            audio_file_voice.seek(0)
                            audio_segment_voice = AudioSegment.from_file(audio_file_voice, format="wav")
                            temp_wav_voice = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                            audio_segment_voice.export(temp_wav_voice.name, format="wav")
                            temp_wav_voice.close()
                            recognizer_voice = sr.Recognizer()
                            with sr.AudioFile(temp_wav_voice.name) as source:
                                audio_data_voice = recognizer_voice.record(source)
                                voice_text = recognizer_voice.recognize_google(audio_data_voice)
                            os.remove(temp_wav_voice.name)
                            st.success(f"Transcription: {voice_text}")
                        except Exception as e:
                            transcription_error = True
                            st.warning(f"Automatic transcription failed: {e}. Please type what you said.")
                else:
                    transcription_error = True
                    st.info("SpeechRecognition or pydub not installed. Please type what you said.")
                if (transcription_error or (audio_bytes_voice and not voice_text)):
                    manual_voice_text = st.text_area("Manual transcription (type what you said):", "", height=100)
                    if manual_voice_text:
                        voice_text = manual_voice_text
                # Prefill the text_area with transcribed text if available
                if voice_text:
                    st.session_state['prompt_question'] = voice_text

        # --- Prompt Engineering Section ---
        if 'prompt_chat_history' not in st.session_state:
            st.session_state.prompt_chat_history = []
        if 'prompt_cache' not in st.session_state:
            st.session_state.prompt_cache = {}
        # Display chat history
        for message in st.session_state.prompt_chat_history:
            if message['role'] == 'user':
                st.markdown("**User:**")
                st.info(message['content'])
            elif message['role'] == 'agent':
                st.markdown("**Agent:**")
                st.success(message['content'])
                # === Voice OUTPUT (TTS) ===
                if interaction_mode == "Voice" and gTTS is not None:
                    try:
                        # Remove any leading role prefix like 'agent:'
                        spoken_text = re.sub(r'^(agent|expert):\s*', '', message['content'], flags=re.I)
                        tts = gTTS(text=spoken_text, lang='en')
                        tts_fp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                        tts.save(tts_fp.name)
                        tts_fp.close()
                        with open(tts_fp.name, 'rb') as audio_file_out:
                            audio_bytes_out = audio_file_out.read()
                        st.audio(audio_bytes_out, format='audio/mp3')
                    except Exception as e:
                        st.warning(f"TTS generation failed: {e}")
            elif message['role'] == 'expert':
                st.markdown("**Expert:**")
                st.success(message['content'], icon="🎓")
                # === Voice OUTPUT (TTS) for Expert role ===
                if interaction_mode == "Voice" and gTTS is not None:
                    try:
                        spoken_text = re.sub(r'^(agent|expert):\s*', '', message['content'], flags=re.I)
                        tts = gTTS(text=spoken_text, lang='en')
                        tts_fp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                        tts.save(tts_fp.name)
                        tts_fp.close()
                        with open(tts_fp.name, 'rb') as audio_file_out:
                            audio_bytes_out = audio_file_out.read()
                        st.audio(audio_bytes_out, format='audio/mp3')
                    except Exception as e:
                        st.warning(f"TTS generation failed: {e}")
        st.markdown("---")
        prompt_template = st.selectbox(
            "Select a prompt template:",
            ["Basic Question", "Role-Based", "Chain of Thought", "Few-Shot", "Custom"]
        )
        # Role selection for role-based prompts
        role = ""
        if prompt_template == "Role-Based":
            role = st.selectbox(
                "Select a role:",
                ["Expert", "Teacher", "Friend", "Analyst", "Custom"]
            )
            if role == "Custom":
                role = st.text_input("Enter custom role:")
        # Few-shot examples
        examples = []
        if prompt_template == "Few-Shot":
            num_examples = st.slider("Number of examples:", 1, 5, 2)
            for i in range(num_examples):
                with st.expander(f"Example {i+1}"):
                    input_text = st.text_area(f"Input {i+1}:", key=f"fewshot_input_{i}")
                    output_text = st.text_area(f"Output {i+1}:", key=f"fewshot_output_{i}")
                    examples.append((input_text, output_text))
        # Custom prompt template
        custom_template = ""
        if prompt_template == "Custom":
            custom_template = st.text_area(
                "Enter your custom prompt template:",
                "I want you to act as a {role}. Please {task}."
            )
        # --- Question Input with Keyboard Shortcuts ---
        # Using a regular text area for multi-line input but adding custom JS to bind keyboard shortcuts:
        question = st.text_area("Enter your question or prompt:", key="prompt_question")

        # Inject JavaScript to make "Enter" trigger the "Generate Response" button
        # and keep "Shift + Enter" as the newline shortcut.
        # This script finds the textarea by its aria-label and the button by its visible text.
        # It then listens for keydown events to perform the shortcut behaviour.
        st.markdown(
            """
            <script>
            // Wait for the DOM to be fully loaded
            document.addEventListener('DOMContentLoaded', function() {
                // Function to attach the key listener
                function attachEnterShortcut() {
                    const textarea = document.querySelector('textarea[aria-label="Enter your question or prompt:"]');
                    if (!textarea) {
                        return;
                    }

                    // Prevent attaching multiple listeners
                    if (textarea.dataset.listenerAttached === 'true') {
                        return;
                    }
                    textarea.dataset.listenerAttached = 'true';

                    textarea.addEventListener('keydown', function(e) {
                        // If only Enter is pressed (no Shift), trigger the Generate Response button
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            // Find the button with the exact text "Generate Response"
                            const buttons = Array.from(document.querySelectorAll('button'));
                            const genBtn = buttons.find(btn => btn.innerText.trim() === 'Generate Response');
                            if (genBtn) {
                                genBtn.click();
                            }
                        }
                    });
                }

                // Initial attempt
                attachEnterShortcut();

                // Observe DOM changes (Streamlit re-renders) and re-attach if needed
                const observer = new MutationObserver(() => {
                    attachEnterShortcut();
                });
                observer.observe(document.body, { childList: true, subtree: true });
            });
            </script>
            """,
            unsafe_allow_html=True
        )
        # === Auto-generate response on voice recording stop ===
        if interaction_mode == "Voice":
            if 'last_voice_prompt' not in st.session_state:
                st.session_state['last_voice_prompt'] = ""
            if voice_text and voice_text != st.session_state['last_voice_prompt']:
                st.session_state['last_voice_prompt'] = voice_text
                question = voice_text  # ensure we use the latest spoken prompt
                st.session_state.prompt_chat_history.append({'role': 'user', 'content': question})
                cache_key = f"{prompt_template}_{question}"
                if cache_key in st.session_state.prompt_cache:
                    response_text = st.session_state.prompt_cache[cache_key]
                    st.session_state.prompt_chat_history.append({'role': 'agent', 'content': response_text})
                    st.rerun()
                else:
                    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.prompt_chat_history[:-1]])
                    if prompt_template == "Basic Question":
                        prompt_final = f"Previous conversation:\n{context}\n\nCurrent question: {question}"
                    elif prompt_template == "Role-Based":
                        prompt_final = f"Previous conversation:\n{context}\n\nI want you to act as a {role}. Please answer the following question: {question}"
                    elif prompt_template == "Chain of Thought":
                        prompt_final = f"Previous conversation:\n{context}\n\nLet's approach this step by step. {question}"
                    elif prompt_template == "Few-Shot":
                        prompt_final = "Here are some examples:\n\n"
                        for i, (input_text, output_text) in enumerate(examples):
                            prompt_final += f"Example {i+1}:\nInput: {input_text}\nOutput: {output_text}\n\n"
                        prompt_final += f"Previous conversation:\n{context}\n\nNow, please answer this question: {question}"
                    elif prompt_template == "Custom":
                        prompt_final = f"Previous conversation:\n{context}\n\n{custom_template.format(role='expert', task=question)}"
                    generation_config = GenerationConfig(
                        temperature=temperature,
                        top_p=top_p,
                        max_output_tokens=max_tokens,
                    )
                    try:
                        model_instance = genai.GenerativeModel(
                            model_name=model,
                            generation_config=generation_config
                        )  # type: ignore[attr-defined]
                        with st.spinner("Generating response..."):
                            response = model_instance.generate_content(prompt_final)
                            st.session_state.prompt_cache[cache_key] = response.text
                            st.session_state.prompt_chat_history.append({
                                'role': 'agent' if prompt_template != "Role-Based" else 'expert',
                                'content': response.text
                            })

                            # Save to CSV log
                            save_query_response_to_csv(
                                user_query=question,
                                bot_response=response.text,
                                query_type=f"Voice {prompt_template}" if interaction_mode == "Voice" else f"Text {prompt_template}",
                                model_used=model,
                                temperature=temperature,
                                additional_metadata={
                                    'interaction_mode': interaction_mode,
                                    'prompt_template': prompt_template,
                                    'role': role if prompt_template == "Role-Based" else None
                                }
                            )
                            st.rerun()
                    except Exception as e:
                        st.warning(f"Error generating response: {e}")

        # Generate button
        if st.button("Generate Response", key="prompt_generate_btn"):
            if question:
                st.session_state.prompt_chat_history.append({'role': 'user', 'content': question})
                cache_key = f"{prompt_template}_{question}"
                if cache_key in st.session_state.prompt_cache:
                    response_text = st.session_state.prompt_cache[cache_key]
                    st.session_state.prompt_chat_history.append({'role': 'agent', 'content': response_text})
                    st.rerun()
                else:
                    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.prompt_chat_history[:-1]])
                    if prompt_template == "Basic Question":
                        prompt = f"Previous conversation:\n{context}\n\nCurrent question: {question}"
                    elif prompt_template == "Role-Based":
                        prompt = f"Previous conversation:\n{context}\n\nI want you to act as a {role}. Please answer the following question: {question}"
                    elif prompt_template == "Chain of Thought":
                        prompt = f"Previous conversation:\n{context}\n\nLet's approach this step by step. {question}"
                    elif prompt_template == "Few-Shot":
                        prompt = "Here are some examples:\n\n"
                        for i, (input_text, output_text) in enumerate(examples):
                            prompt += f"Example {i+1}:\nInput: {input_text}\nOutput: {output_text}\n\n"
                        prompt += f"Previous conversation:\n{context}\n\nNow, please answer this question: {question}"
                    elif prompt_template == "Custom":
                        prompt = f"Previous conversation:\n{context}\n\n{custom_template.format(role='expert', task=question)}"
                    generation_config = GenerationConfig(
                        temperature=temperature,
                        top_p=top_p,
                        max_output_tokens=max_tokens,
                    )
                    try:
                        model_instance = genai.GenerativeModel(
                            model_name=model,
                            generation_config=generation_config
                        )  # type: ignore[attr-defined]
                        with st.spinner("Generating response..."):
                            response = model_instance.generate_content(prompt)
                            st.session_state.prompt_cache[cache_key] = response.text
                            st.session_state.prompt_chat_history.append({
                                'role': 'agent' if prompt_template != "Role-Based" else 'expert',
                                'content': response.text
                            })

                            # Save to CSV log
                            save_query_response_to_csv(
                                user_query=question,
                                bot_response=response.text,
                                query_type=f"Text {prompt_template}",
                                model_used=model,
                                temperature=temperature,
                                additional_metadata={
                                    'interaction_mode': 'Text',
                                    'prompt_template': prompt_template,
                                    'role': role if prompt_template == "Role-Based" else None
                                }
                            )
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a question or prompt.")
        if st.button("Clear Chat History", key="prompt_clear_btn"):
            st.session_state.prompt_chat_history = []
            st.session_state.prompt_cache = {}
            st.rerun()

    elif typepdf == "URL Search":
        st.header("🌐 URL Search", divider="blue")

        # Initialize processed flag
        if 'url_processed' not in st.session_state:
            st.session_state['url_processed'] = False

        # --- Interaction mode selection ---
        interaction_mode = render_voice_interaction_mode("url_search")

        # Minimal UI for URL Search
        url_input = st.text_area(
            "Website URL:",
            placeholder="https://example.com",
            key="url_input",
            height=70
        )
        # Process URL button
        if st.button("🔍 Process URL", type="primary", key="process_url_btn"):
            if url_input:
                if not url_input.startswith(('http://', 'https://')):
                    url_input = 'https://' + url_input
                with st.spinner("🌐 Fetching and processing web content..."):
                    try:
                        combined_db_name = f"url_{hash(url_input) % 10000}"
                        db = process_url_content(url_input, combined_db_name)
                        if db:
                            st.success(f"✅ Successfully processed: {url_input}")
                            st.session_state['current_url_db'] = combined_db_name
                            st.session_state['current_url'] = url_input
                            st.session_state['url_processed'] = True
                            st.rerun()
                        else:
                            st.error("❌ Failed to process URL. Please check the URL and try again.")
                    except Exception as e:
                        st.error(f"❌ Error processing URL: {str(e)}")
            else:
                st.warning("⚠️ Please enter a URL.")
        # Question section (only show if URL is processed in current session)
        if st.session_state.get('url_processed') and 'current_url_db' in st.session_state and 'current_url' in st.session_state:
            question = get_voice_or_text_input(interaction_mode, "Ask a question about the web content:", "url_search_question", prefill_key="url_search_prefill")

            if question:
                if use_agent:
                    with st.spinner("Agent is searching the web and aggregating references..."):
                        response_json = crewai_web_search(st.session_state['current_url'], question)
                        if response_json:
                            parsed_data = json.loads(response_json)
                            st.subheader("Agent Answer:")
                            answer_text = parsed_data.get("Answer", "No answer found.")
                            st.write(answer_text)
                            # Voice output for agent response
                            render_voice_output(answer_text, interaction_mode)
                            references = parsed_data.get("References")
                            if references:
                                st.write("References:")
                                for ref in references:
                                    st.markdown(f"- [{ref}]({ref})")
                            st.write("Search Information:")
                            st.write(f"Reasoning: {parsed_data.get('Reasoning', 'N/A')}")

                            # Save to CSV log
                            save_query_response_to_csv(
                                user_query=question,
                                bot_response=answer_text,
                                query_type="URL Search Agent",
                                model_used="CrewAI Agent",
                                additional_metadata={
                                    'interaction_mode': interaction_mode,
                                    'url': st.session_state['current_url'],
                                    'references': str(references) if references else None,
                                    'reasoning': parsed_data.get('Reasoning', 'N/A')
                                }
                            )
                else:
                    try:
                        with st.spinner("Processing your question... This may take a few moments."):
                            response = rag_pipeline(question, model, st.session_state['current_url_db'], temperature, top_p, max_tokens)
                            if response:
                                json_content = response.content.strip("```json\n").strip("```")
                                parsed_data = json.loads(json_content)

                                answer_text = parsed_data.get("Answer", "No answer found.")
                                st.subheader("Response:")
                                st.write(answer_text)
                                # Voice output for regular response
                                render_voice_output(answer_text, interaction_mode)
                                # st.write(f"Similarity Score: {parsed_data.get('Overall Similarity', 'N/A')}")
                                st.write(f"Used Chunks: {parsed_data.get('Used Chunks', 'N/A')}")
                                st.write(f"Reasoning: {parsed_data.get('Reasoning', 'N/A')}")

                                # Save to CSV log
                                save_query_response_to_csv(
                                    user_query=question,
                                    bot_response=answer_text,
                                    query_type="URL Search RAG",
                                    model_used=model,
                                    temperature=temperature,
                                    additional_metadata={
                                        'interaction_mode': interaction_mode,
                                        'url': st.session_state['current_url'],
                                        'similarity_score': parsed_data.get('Overall Similarity', 'N/A'),
                                        'used_chunks': parsed_data.get('Used Chunks', 'N/A'),
                                        'reasoning': parsed_data.get('Reasoning', 'N/A')
                                    }
                                )
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            if st.button("🗑️ Clear URL", key="clear_url_btn"):
                if 'current_url_db' in st.session_state:
                    del st.session_state['current_url_db']
                if 'current_url' in st.session_state:
                    del st.session_state['current_url']
                st.session_state['url_processed'] = False
                st.rerun()

    elif typepdf == "PDF files":
      st.header("📄 PDF Files", divider="blue")

      # --- Interaction mode selection ---
      interaction_mode = render_voice_interaction_mode("pdf_files")

      # --- Source selection: Upload new vs Use saved ---
      saved_pdf_groups = st.session_state.get('saved_files', {}).get('PDF files', {})
      use_saved_source = False
      selected_saved_group_ids = []
      if saved_pdf_groups:
          source_choice = st.radio(
              "Choose source:",
              ["Upload new", "Use saved"],
              horizontal=True,
              key="pdf_source_choice"
          )
          if source_choice == "Use saved":
              use_saved_source = True
              group_display_map = {f"{gid} - {meta['group_name']}": gid for gid, meta in saved_pdf_groups.items()}
              selected_saved_displays = st.multiselect(
                  "Select saved PDF vector database(s):",
                  list(group_display_map.keys()),
                  key="pdf_saved_select_multi"
              )
              if selected_saved_displays:
                  selected_saved_group_ids = [group_display_map[d] for d in selected_saved_displays]

      uploaded_files = None if use_saved_source else st.file_uploader("Choose maximum of 50 PDFs", type='pdf', accept_multiple_files=True)
      if uploaded_files and len(uploaded_files) > 50:
        st.error("You can upload a maximum of 50 PDF files")
        return
      # Create combined_db_name based on uploads when present; otherwise default
      db_name_list = []
      if uploaded_files:
          for file in uploaded_files:
              original_name = file.name
              if len(original_name) > 20:
                  name, ext = os.path.splitext(original_name)
                  truncated_name = name[:20 - len(ext)] + ext
                  db_name_list.append(os.path.splitext(truncated_name)[0])  # Base name without extension
              else:
                  db_name_list.append(os.path.splitext(original_name)[0])
      combined_db_name = "_".join(db_name_list) if db_name_list else "pdf_db"
      if len(combined_db_name) > 50:
          combined_db_name = combined_db_name[:50]

      # Check if saved file group(s) are selected (from sidebar or from in-page selector)
      selected_group_ids = selected_saved_group_ids or ([st.session_state.get('selected_file_group')] if st.session_state.get('selected_file_group') else [])
      selected_file_type = 'PDF files' if selected_group_ids else st.session_state.get('selected_file_type')

      if selected_group_ids and selected_file_type == 'PDF files':
          # Load selected file group(s)
          try:
              if len(selected_group_ids) == 1:
                  file_group = load_file_group(selected_group_ids[0], selected_file_type)
                  if file_group:
                      st.info(f"📁 Using saved file group: {file_group['group_name']}")
                      db = file_group.get('db')
                      # Ensure FAISS path is available for queries
                      combined_db_name = file_group.get('db_path', None) or "pdf_db"
                      if db:
                          st.success("✅ Loaded saved embeddings successfully!")
                      else:
                          st.error("❌ Could not load saved embeddings")
                          return
                  else:
                      st.error("❌ Could not load selected file group")
                      return
              else:
                  # Merge multiple saved FAISS indexes into one combined index

                  merged_db = None
                  merged_names = []
                  for gid in selected_group_ids:
                      fg = load_file_group(gid, selected_file_type)
                      if not fg or not fg.get('db'):
                          st.warning(f"Skipping group {gid}: could not load embeddings")
                          continue
                      merged_names.append(fg.get('group_name', gid))
                      if merged_db is None:
                          merged_db = fg['db']
                      else:
                          try:
                              merged_db.merge_from(fg['db'])
                          except Exception as _merge_err:
                              st.error(f"Failed to merge index from {gid}: {_merge_err}")
                              return

                  if not merged_db:
                      st.error("❌ Could not load any of the selected saved groups")
                      return

                  # Persist merged index to a combined directory
                  storage_dir = ensure_storage_directory()
                  merge_id = get_file_hash("_".join(selected_group_ids), "pdf_groups")
                  combined_dir = os.path.join(storage_dir, f"combined_{merge_id}_db")
                  try:
                      merged_db.save_local(combined_dir)
                  except Exception as _save_err:
                      st.error(f"Failed to save merged index: {_save_err}")
                      return

                  combined_db_name = combined_dir
                  st.success(f"✅ Loaded {len(selected_group_ids)} groups and built a combined index.")
                  st.info("Groups: " + ", ".join(merged_names[:3]) + (" ..." if len(merged_names) > 3 else ""))
          except Exception as e:
              st.error(f"❌ Error preparing saved embeddings: {e}")
              return
      elif uploaded_files:
          st.write(f"Processing {len(uploaded_files)} files...")

          # Initialize embedding model
          embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")

          # Check if we need to process files
          db = None
          process_files = True

          # If DB already exists in session and files haven't changed, skip processing
          if combined_db_name in st.session_state['db_initialized'] and st.session_state['db_initialized'][combined_db_name]:
              try:
                  # Try to load the existing DB
                  index_path = f"./{combined_db_name}/index.faiss"
                  if os.path.exists(index_path):
                      db = FAISS.load_local(
                          f"./{combined_db_name}",
                          embedding_model,
                          allow_dangerous_deserialization=True
                      )
                      st.success(f"Using existing vector database for {len(uploaded_files)} files.")
                      process_files = False
                  else:
                      st.warning("Vector database not found. Creating new one...")
              except Exception as e:
                  st.warning(f"Could not load existing database: {e}. Recreating...")

          if process_files:
              # Process and store embeddings with error handling
              try:
                  db = embed_and_store_files(uploaded_files, embedding_model, combined_db_name)
              except Exception as e:  # Use generic Exception
                  if "429" in str(e) and "Quota exceeded" in str(e):  # Check for quota exceeded error
                      st.error("Quota exceeded for Gemini API. Please change the API key from code.")
                      return  # Exit the block until the user retries with the new key
                  else:
                      st.error(f"An error occurred: {str(e)}")
                      return

          # Add save button for new uploads
          if db and uploaded_files:
              col_save, col_clear = st.columns([1, 1])
              with col_save:
                  if st.button("💾 Save Files", key="save_pdf_files", type="primary"):
                      # Prepare file group data
                      file_group_data = {
                          'files': [{'name': f.name, 'size': f.size} for f in uploaded_files],
                          'db': db
                      }

                      # Save the file group
                      group_id, metadata = save_file_group(file_group_data, 'PDF files')
                      st.success(f"✅ Files saved successfully! Group ID: {group_id}")
                      st.rerun()

              with col_clear:
                  if st.button("🗑️ Clear Files", key="clear_pdf_files"):
                      st.session_state['selected_file_group'] = None
                      st.session_state['selected_file_type'] = None
                      st.rerun()
      else:
          st.info("Please upload PDF files or select a saved file group from the sidebar.")
          return

      # Create two columns for question input and visualization toggle
      col1, col2 = st.columns([3, 1])

      with col1:
          question = get_voice_or_text_input(
              interaction_mode,
              "Enter your question and hit return:",
              "pdf_question",
              prefill_key=None,
              placeholder="e.g., What does this PDF say about compliance requirements?"
          )
      with col2:
          visualization_enabled = st.toggle("Visualize", value=False)

      if question:
          try:
              # Show loading spinner while processing the question
              with st.spinner("Processing your question... This may take a few moments."):
                  # Query Vector DB
                  response = rag_pipeline(question, model, combined_db_name, temperature, top_p, max_tokens)
                  if response:
                      json_content = response.content.strip("```json\n").strip("```")
                      parsed_data = json.loads(json_content)

                      answer = parsed_data.get("Answer", None)
                      overall_similarity = parsed_data.get("Overall Similarity", None)

                      st.subheader("Response:")
                      st.write(answer)
                      # Voice output for PDF response
                      render_voice_output(answer, interaction_mode)
                    #   st.write(f"Similarity Score: {overall_similarity}")

                      # Save to CSV log
                      save_query_response_to_csv(
                          user_query=question,
                          bot_response=answer,
                          query_type="PDF RAG Query",
                          model_used=model,
                          temperature=temperature,
                          additional_metadata={
                              'interaction_mode': interaction_mode,
                              'num_files': len(uploaded_files) if uploaded_files else 0,
                              'file_names': [f.name for f in uploaded_files] if uploaded_files else [],
                              'similarity_score': overall_similarity,
                              'visualization_enabled': visualization_enabled
                          }
                      )

                  # If visualization is enabled, generate and display visualizations
                  if visualization_enabled:
                      st.subheader("Data Analysis and Visualizations")

                      with st.spinner("Extracting data from PDFs for visualization..."):
                          # Extract text content from PDFs
                          pdf_texts = []
                          if uploaded_files:
                              for pdf_file in uploaded_files:
                                  try:
                                      # Create a temporary file to process the PDF
                                      with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                                          temp_file.write(pdf_file.read())
                                          temp_file_path = temp_file.name

                                      # Use PyPDFLoader to extract text
                                      loader = PyPDFLoader(temp_file_path)
                                      pages = loader.load()

                                      # Extract text from pages
                                      pdf_text = "\n\n".join([page.page_content for page in pages])
                                      pdf_texts.append(pdf_text)

                                      # Clean up temporary file
                                      os.unlink(temp_file_path)
                                  except Exception as pdf_error:
                                      logger.error(f"Error processing PDF file {pdf_file.name}: {str(pdf_error)}")
                                      st.warning(f"Could not process PDF file {pdf_file.name}: {str(pdf_error)}")

                          # Combine all PDF texts
                          combined_text = "\n\n".join(pdf_texts)

                      with st.spinner("Generating visualizations..."):
                          # Generate visualizations using the PDF-specific function
                          visualization_data = visualize_pdf_data(combined_text, question, model, temperature, top_p, max_tokens)

                          if visualization_data:
                              # Display overall analysis
                              st.write("### Data Analysis")
                              st.write(visualization_data.get("analysis", "No analysis available"))

                              # Display key insights
                              st.write("### Key Insights")
                              for insight in visualization_data.get("key_insights", []):
                                  st.write(f"• {insight}")

                              # Display visualization suggestions and plots
                              st.write("### Visualizations")
                              for viz in visualization_data.get("visualizations", []):
                                  with st.expander(f"{viz['type']} - {viz['description']}"):
                                      st.write("#### Description")
                                      st.write(viz['description'])

                                      st.write("#### Why This Visualization?")
                                      st.write(viz['reason'])

                                      # Execute the visualization code
                                      try:
                                          # Create a local environment with the necessary variables
                                          local_env = {'px': px, 'go': go}
                                          # Execute the code and get the figure
                                          exec(viz['code'], local_env)
                                          fig = local_env.get('fig')
                                          if fig:
                                              # Update layout to ensure inline display
                                              fig.update_layout(
                                                  height=400,
                                                  margin=dict(l=20, r=20, t=40, b=20)
                                              )
                                              # Display the figure inline
                                              st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                                          else:
                                              st.warning("Could not create visualization. The code did not produce a figure.")
                                      except Exception as e:
                                          st.error(f"Error creating visualization: {str(e)}")
                                          # Show the code that failed for debugging
                                          st.code(viz['code'], language="python")
                          else:
                              st.warning("Could not generate visualization suggestions. Please try again.")
          except Exception as e:  # Use generic Exception
              if "429" in str(e) and "Quota exceeded" in str(e):  # Check for quota exceeded error during query
                  st.error("Quota exceeded for Gemini API. Please change the API key from code.")
                  return
              else:
                  st.error(f"An error occurred during query: {str(e)}")
                  return


    elif typepdf == "CSV files":
        st.header("📊 CSV Files", divider="blue")

        # --- Interaction mode selection ---
        interaction_mode = render_voice_interaction_mode("csv_files")

        # Source selection like PDF/JSON
        use_saved, selected_saved_group = select_source_and_saved_group('CSV files', 'csv')
        # Respect in-page selection first, then fall back to sidebar selection
        selected_group_id = selected_saved_group or st.session_state.get('selected_file_group')
        selected_file_type = 'CSV files' if selected_saved_group else st.session_state.get('selected_file_type')

        # File upload (disabled when using saved)
        uploaded_csv_files = None if use_saved else st.file_uploader("Choose maximum of 50 CSV files.", type="csv", accept_multiple_files=True)
        if uploaded_csv_files and len(uploaded_csv_files) > 50:
            st.error("You can upload a maximum of 50 CSV files")
            return

        # Build DB name from uploads when uploading new
        combined_db_name = build_combined_db_name_from_uploads(uploaded_csv_files or [], "csv_db")
        if len(combined_db_name) > 50:
            combined_db_name = combined_db_name[:50]

        if selected_group_id and selected_file_type == 'CSV files':
            # Load selected file group
            file_group = load_file_group(selected_group_id, selected_file_type)
            if file_group:
                st.info(f"📁 Using saved file group: {file_group['group_name']}")
                db = file_group.get('db')
                if db:
                    st.success("✅ Loaded saved embeddings successfully!")
                    # Use saved FAISS directory for retrieval
                    combined_db_name = file_group.get('db_path', combined_db_name)
                else:
                    st.error("❌ Could not load saved embeddings")
                    return
            else:
                st.error("❌ Could not load selected file group")
                return
        elif uploaded_csv_files:
            # Check if we need to process files
            process_files = True
            embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")

            # If DB already exists in session and files haven't changed, skip processing
            if combined_db_name in st.session_state['db_initialized'] and st.session_state['db_initialized'][combined_db_name]:
                try:
                    # Try to load the existing DB
                    db = FAISS.load_local(
                        combined_db_name,
                        embedding_model,
                        allow_dangerous_deserialization=True
                    )
                    st.success(f"Using existing vector database for {len(uploaded_csv_files)} files.")
                    process_files = False
                except Exception as e:
                    st.warning(f"Could not load existing database: {e}. Recreating...")

            if process_files:
                with st.spinner("Processing CSV files and creating embeddings..."):
                    temp_file_paths = []
                    for csv_file in uploaded_csv_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
                            temp_file.write(csv_file.read())
                            temp_file_paths.append(temp_file.name)

                    # Will recreate database even if directory exists
                    db = load_and_embed_csv(temp_file_paths, combined_db_name)
                    st.success("CSV files processed and embeddings stored successfully.")

                    # Clean up temp files
                    for temp_file in temp_file_paths:
                        try:
                            os.remove(temp_file)
                        except:
                            pass

            # Add save button for new uploads
            if db and uploaded_csv_files:
                col_save, col_clear = st.columns([1, 1])
                with col_save:
                    if st.button("💾 Save Files", key="save_csv_files", type="primary"):
                        # Prepare file group data
                        file_group_data = {
                            'files': [{'name': f.name, 'size': f.size} for f in uploaded_csv_files],
                            'db': db
                        }

                        # Save the file group
                        group_id, metadata = save_file_group(file_group_data, 'CSV files')
                        st.success(f"✅ Files saved successfully! Group ID: {group_id}")
                        st.rerun()

                with col_clear:
                    if st.button("🗑️ Clear Files", key="clear_csv_files"):
                        st.session_state['selected_file_group'] = None
                        st.session_state['selected_file_type'] = None
                        st.rerun()
        else:
            st.info("Please upload CSV files or select a saved file group from the sidebar.")
            return

        # Create two columns for question input and visualization toggle
        col1, col2 = st.columns([3, 1])

        with col1:
            question = get_voice_or_text_input(
                interaction_mode,
                "Enter your question and hit return:",
                "csv_question",
                prefill_key=None,
                placeholder="e.g., Summarize total sales by region for 2024"
            )
        with col2:
            visualization_enabled = st.toggle("Visualize", value=False)

        if question:
            with st.spinner("Processing your question... This may take a few moments."):
                # Query Vector DB
                response = rag_pipeline(question, model, combined_db_name)
                json_content = response.content.strip("```json\n").strip("```")
                parsed_data = json.loads(json_content)

                answer = parsed_data.get("Answer", None)
                overall_similarity = parsed_data.get("Overall Similarity", None)

                st.subheader("Response:")
                st.write(answer)
                # Voice output for CSV response
                render_voice_output(answer, interaction_mode)
                # st.write(f"Similarity Score: {overall_similarity}")

                # Save to CSV log
                save_query_response_to_csv(
                    user_query=question,
                    bot_response=answer,
                    query_type="CSV RAG Query",
                    model_used=model,
                    temperature=temperature,
                    additional_metadata={
                        'interaction_mode': interaction_mode,
                        'num_files': len(uploaded_csv_files) if uploaded_csv_files else 0,
                        'file_names': [f.name for f in uploaded_csv_files] if uploaded_csv_files else [],
                        'similarity_score': overall_similarity,
                        'visualization_enabled': visualization_enabled
                        }
                    )

            # If visualization is enabled, generate and display visualizations
            if visualization_enabled:
                st.subheader("Data Analysis and Visualizations")

                with st.spinner("Loading and processing CSV data..."):
                    # Read and combine CSV content into a DataFrame
                    dfs = []
                    if uploaded_csv_files:
                        for csv_file in uploaded_csv_files:
                            df = pd.read_csv(csv_file)
                            dfs.append(df)

                    if len(dfs) > 1:
                        df = pd.concat(dfs, ignore_index=True)
                    elif len(dfs) == 1:
                        df = dfs[0]
                    else:
                        st.warning("No CSV data available for visualization")
                        return

                with st.spinner("Generating visualizations..."):
                    # Generate visualizations
                    visualization_data = visualize_csv_data(df.to_string(), question, model, temperature, top_p, max_tokens)

                    if visualization_data:
                        # Display overall analysis
                        st.write("### Data Analysis")
                        st.write(visualization_data.get("analysis", "No analysis available"))

                        # Display key insights
                        st.write("### Key Insights")
                        for insight in visualization_data.get("key_insights", []):
                            st.write(f"• {insight}")

                        # Display visualization suggestions and plots
                        st.write("### Visualizations")
                        for viz in visualization_data.get("visualizations", []):
                            with st.expander(f"{viz['type']} - {viz['description']}"):
                                st.write("#### Description")
                                st.write(viz['description'])

                                st.write("#### Why This Visualization?")
                                st.write(viz['reason'])

                                # Execute the visualization code
                                try:
                                    # Create a local environment with the DataFrame
                                    local_env = {'df': df, 'px': px, 'go': go}
                                    # Execute the code and get the figure
                                    exec(viz['code'], local_env)
                                    fig = local_env.get('fig')
                                    if fig:
                                        # Update layout to ensure inline display
                                        fig.update_layout(
                                            height=400,
                                            margin=dict(l=20, r=20, t=40, b=20)
                                        )
                                        # Display the figure inline
                                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                                    else:
                                        st.warning("Could not create visualization. The code did not produce a figure.")
                                except Exception as e:
                                    st.error(f"Error creating visualization: {str(e)}")
                    else:
                        st.warning("Could not generate visualization suggestions. Please try again.")

    elif typepdf == "Excel files":
        st.header("📊 Excel Files", divider="blue")

        # --- Interaction mode selection ---
        interaction_mode = render_voice_interaction_mode("excel_files")

        # Source selection like CSV/JSON
        use_saved, selected_saved_group = select_source_and_saved_group('Excel files', 'excel')
        # Prefer in-page saved selection over sidebar
        selected_group_id = selected_saved_group or st.session_state.get('selected_file_group')
        selected_file_type = 'Excel files' if selected_saved_group else st.session_state.get('selected_file_type')

        # File uploader disabled when using saved
        uploaded_excel_files = None if use_saved else st.file_uploader("Choose maximum of 50 Excel files.", type=["xlsx", "xls"], accept_multiple_files=True)
        if uploaded_excel_files and len(uploaded_excel_files) > 50:
            st.error("You can upload a maximum of 50 Excel files")
            return

        # Build DB name from uploads when uploading new
        combined_db_name = build_combined_db_name_from_uploads(uploaded_excel_files or [], "excel_db")
        if len(combined_db_name) > 50:
            combined_db_name = combined_db_name[:50]

        if selected_group_id and selected_file_type == 'Excel files':
            # Load selected file group
            file_group = load_file_group(selected_group_id, selected_file_type)
            if file_group:
                st.info(f"📁 Using saved file group: {file_group['group_name']}")
                db = file_group.get('db')
                if db:
                    st.success("✅ Loaded saved embeddings successfully!")
                    # Use saved FAISS directory for retrieval
                    combined_db_name = file_group.get('db_path', combined_db_name)
                else:
                    st.error("❌ Could not load saved embeddings")
                    return
            else:
                st.error("❌ Could not load selected file group")
                return
        elif uploaded_excel_files:
            # Check if we need to process files
            process_files = True
            embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY, model="models/embedding-001")

            # If DB already exists in session and files haven't changed, skip processing
            if combined_db_name in st.session_state['db_initialized'] and st.session_state['db_initialized'][combined_db_name]:
                try:
                    # Try to load the existing DB
                    db = FAISS.load_local(
                        combined_db_name,
                        embedding_model,
                        allow_dangerous_deserialization=True
                    )
                    st.success(f"Using existing vector database for {len(uploaded_excel_files)} files.")
                    process_files = False
                except Exception as e:
                    st.warning(f"Could not load existing database: {e}. Recreating...")

            if process_files:
                with st.spinner("Processing Excel files and creating embeddings..."):
                    temp_file_paths = []
                    for excel_file in uploaded_excel_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(excel_file.name)[1]) as temp_file:
                            temp_file.write(excel_file.read())
                            temp_file_paths.append(temp_file.name)

                    # Will recreate database even if directory exists
                    db = load_and_embed_excel(temp_file_paths, combined_db_name)
                    st.success("Excel files processed and embeddings stored successfully.")

                    # Clean up temp files
                    for temp_file in temp_file_paths:
                        try:
                            os.remove(temp_file)
                        except:
                            pass

            # Add save button for new uploads
            if db and uploaded_excel_files:
                col_save, col_clear = st.columns([1, 1])
                with col_save:
                    if st.button("💾 Save Files", key="save_excel_files", type="primary"):
                        # Prepare file group data
                        file_group_data = {
                            'files': [{'name': f.name, 'size': f.size} for f in uploaded_excel_files],
                            'db': db
                        }

                        # Save the file group
                        group_id, metadata = save_file_group(file_group_data, 'Excel files')
                        st.success(f"✅ Files saved successfully! Group ID: {group_id}")
                        st.rerun()

                with col_clear:
                    if st.button("🗑️ Clear Files", key="clear_excel_files"):
                        st.session_state['selected_file_group'] = None
                        st.session_state['selected_file_type'] = None
                        st.rerun()
        else:
            st.info("Please upload Excel files or select a saved file group from the sidebar.")
            return

        # Create two columns for question input and visualization toggle
        col1, col2 = st.columns([3, 1])

        with col1:
            question = get_voice_or_text_input(
                interaction_mode,
                "Enter your question and hit return:",
                "excel_question",
                prefill_key=None,
                placeholder="e.g., What are the top 5 products by revenue?"
            )
        with col2:
            visualization_enabled = st.toggle("Visualize", value=False)

        if question:
            with st.spinner("Processing your question... This may take a few moments."):
                # Query Vector DB
                response = rag_pipeline(question, model, combined_db_name)
                json_content = response.content.strip("```json\n").strip("```")
                parsed_data = json.loads(json_content)

                answer = parsed_data.get("Answer", None)
                overall_similarity = parsed_data.get("Overall Similarity", None)

                st.subheader("Response:")
                st.write(answer)
                render_voice_output(answer, interaction_mode)
                # st.write(f"Similarity Score: {overall_similarity}")

                # Save to CSV log
                save_query_response_to_csv(
                    user_query=question,
                    bot_response=answer,
                    query_type="Excel RAG Query",
                    model_used=model,
                    temperature=temperature,
                    additional_metadata={
                        'interaction_mode': interaction_mode,
                        'num_files': len(uploaded_excel_files) if uploaded_excel_files else 0,
                        'file_names': [f.name for f in uploaded_excel_files] if uploaded_excel_files else [],
                        'similarity_score': overall_similarity,
                        'visualization_enabled': visualization_enabled
                    }
                )

            # If visualization is enabled, generate and display visualizations
            if visualization_enabled:
                st.subheader("Data Analysis and Visualizations")

                with st.spinner("Loading and processing Excel data..."):
                    # Read and combine Excel content into a DataFrame
                    dfs = []
                    if uploaded_excel_files:
                        for excel_file in uploaded_excel_files:
                            df = pd.read_excel(excel_file)
                            dfs.append(df)

                    if len(dfs) > 1:
                        df = pd.concat(dfs, ignore_index=True)
                    elif len(dfs) == 1:
                        df = dfs[0]
                    else:
                        st.warning("No Excel data available for visualization")
                        return

                with st.spinner("Generating visualizations..."):
                    # Generate visualizations
                    visualization_data = visualize_csv_data(df.to_string(), question, model, temperature, top_p, max_tokens)

                    if visualization_data:
                        # Display overall analysis
                        st.write("### Data Analysis")
                        st.write(visualization_data.get("analysis", "No analysis available"))

                        # Display key insights
                        st.write("### Key Insights")
                        for insight in visualization_data.get("key_insights", []):
                            st.write(f"• {insight}")

                        # Display visualization suggestions and plots
                        st.write("### Visualizations")
                        for viz in visualization_data.get("visualizations", []):
                            with st.expander(f"{viz['type']} - {viz['description']}"):
                                st.write("#### Description")
                                st.write(viz['description'])

                                st.write("#### Why This Visualization?")
                                st.write(viz['reason'])

                                # Execute the visualization code
                                try:
                                    # Create a local environment with the DataFrame
                                    local_env = {'df': df, 'px': px, 'go': go}
                                    # Execute the code and get the figure
                                    exec(viz['code'], local_env)
                                    fig = local_env.get('fig')
                                    if fig:
                                        # Update layout to ensure inline display
                                        fig.update_layout(
                                            height=400,
                                            margin=dict(l=20, r=20, t=40, b=20)
                                        )
                                        # Display the figure inline
                                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                                    else:
                                        st.warning("Could not create visualization. The code did not produce a figure.")
                                except Exception as e:
                                    st.error(f"Error creating visualization: {str(e)}")
                    else:
                        st.warning("Could not generate visualization suggestions. Please try again.")


    elif typepdf == "TXT files":
        st.header("📄 TXT Files", divider="blue")
        interaction_mode = render_voice_interaction_mode("txt_files")

        # Upload only (no saved groups UI)
        use_saved = False
        selected_saved_group = None
        selected_group_id = None
        selected_file_type = None
        uploaded_files = st.file_uploader("Choose maximum of 50 TXT", type='txt', accept_multiple_files=True)
        if uploaded_files and len(uploaded_files) > 50:
            st.error("You can upload a maximum of 50 TXT files")
            return

        # Build DB name from uploads when uploading new
        combined_db_name = build_combined_db_name_from_uploads(uploaded_files or [], "txt_db")
        if len(combined_db_name) > 50:
            combined_db_name = combined_db_name[:50]

        if uploaded_files:
            st.write(f"Processing {len(uploaded_files)} files...")

            # Initialize embedding model
            embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=SecretStr(GOOGLE_API_KEY), model="models/embedding-001")

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Process files using unified function
            db, success = save_media_embeddings(uploaded_files, 'TXT files', embedding_model, combined_db_name, progress_bar, status_text)

            if success and db:
                # Question Input (uploads only)
                placeholder_label = "Enter your question about the TXT files:"
                question = get_voice_or_text_input(
                    interaction_mode,
                    placeholder_label,
                    "txt_question",
                    prefill_key=None,
                    placeholder="e.g., Extract key points from the uploaded TXT files"
                )

                if question and db:
                    try:
                        # Show loading spinner while processing the question
                        with st.spinner("Processing your question... This may take a few moments."):
                            # Query Vector DB
                            response = rag_pipeline(question, model, combined_db_name, temperature, top_p, max_tokens)
                            if response:
                                json_content = response.content
                                if isinstance(json_content, str):
                                    json_content = json_content.strip("```json\n").strip("```")
                                    try:
                                        parsed_data = json.loads(json_content)
                                    except Exception:
                                        st.warning("Could not parse model response as JSON.")
                                        parsed_data = {}
                                else:
                                    st.warning("Model response was not a string and could not be parsed as JSON.")
                                    parsed_data = {}
                                st.subheader("Response:")
                                answer_text = parsed_data.get("Answer", "No answer found.")
                                st.write(answer_text)
                                render_voice_output(answer_text, interaction_mode)
                                st.write(f"Used Chunks: {parsed_data.get('Used Chunks', 'N/A')}")
                                st.write(f"Reasoning: {parsed_data.get('Reasoning', 'N/A')}")

                                # Save to CSV log
                                save_query_response_to_csv(
                                    user_query=question,
                                    bot_response=answer_text,
                                    query_type="TXT RAG Query",
                                    model_used=model,
                                    temperature=temperature,
                                    additional_metadata={
                                        'interaction_mode': interaction_mode,
                                        'num_files': len(uploaded_files),
                                        'file_names': [f.name for f in uploaded_files],
                                        'similarity_score': parsed_data.get('Overall Similarity', 'N/A'),
                                        'used_chunks': parsed_data.get('Used Chunks', 'N/A'),
                                        'reasoning': parsed_data.get('Reasoning', 'N/A')
                                    }
                                )
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.error("Failed to process TXT files. Please try again.")

    elif typepdf == "PPT files":
        st.header("📄 PPT Files", divider="blue")
        interaction_mode = render_voice_interaction_mode("ppt_files")

        # Upload only (no saved groups UI)
        selected_group_id = None
        selected_file_type = None
        uploaded_files = st.file_uploader("Choose maximum of 50 PPT", type=['ppt','pptx'], accept_multiple_files=True)
        if uploaded_files and len(uploaded_files) > 50:
            st.error("You can upload a maximum of 50 PPT files")
            return

        # Generate database name
        db_name_list = []
        for file in uploaded_files:
            original_name = file.name
            if len(original_name) > 20:
                name, ext = os.path.splitext(original_name)
                truncated_name = name[:20 - len(ext)] + ext
                db_name_list.append(os.path.splitext(truncated_name)[0])
            else:
                db_name_list.append(os.path.splitext(original_name)[0])

        combined_db_name = "_".join(db_name_list) if db_name_list else "ppt_db"
        if len(combined_db_name) > 50:
            combined_db_name = combined_db_name[:50]

        if uploaded_files:
            st.write(f"Processing {len(uploaded_files)} files...")

            # Initialize embedding model
            embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=SecretStr(GOOGLE_API_KEY), model="models/embedding-001")

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Process files using unified function
            db, success = save_media_embeddings(uploaded_files, 'PPT files', embedding_model, combined_db_name, progress_bar, status_text)

            if success and db:
                # Question Input (uploads only)
                question = get_voice_or_text_input(
                    interaction_mode,
                    "Enter your question about the PPT files:",
                    "ppt_question",
                    prefill_key=None,
                    placeholder="e.g., What are the main takeaways from this presentation?"
                )

                if question and db:
                    try:
                        # Show loading spinner while processing the question
                        with st.spinner("Processing your question... This may take a few moments."):
                            # Query Vector DB
                            response = rag_pipeline(question, model, combined_db_name, temperature, top_p, max_tokens)
                            if response:
                                json_content = response.content
                                if isinstance(json_content, str):
                                    json_content = json_content.strip("```json\n").strip("```")
                                    try:
                                        parsed_data = json.loads(json_content)
                                    except Exception:
                                        st.warning("Could not parse model response as JSON.")
                                        parsed_data = {}
                                else:
                                    st.warning("Model response was not a string and could not be parsed as JSON.")
                                    parsed_data = {}
                                st.subheader("Response:")
                                answer_text = parsed_data.get("Answer", "No answer found.")
                                st.write(answer_text)
                                render_voice_output(answer_text, interaction_mode)
                                st.write(f"Used Chunks: {parsed_data.get('Used Chunks', 'N/A')}")

                                # Save to CSV log
                                save_query_response_to_csv(
                                    user_query=question,
                                    bot_response=answer_text,
                                    query_type="PPT RAG Query",
                                    model_used=model,
                                    temperature=temperature,
                                    additional_metadata={
                                        'interaction_mode': interaction_mode,
                                        'num_files': len(uploaded_files),
                                        'file_names': [f.name for f in uploaded_files],
                                        'similarity_score': parsed_data.get('Overall Similarity', 'N/A'),
                                        'used_chunks': parsed_data.get('Used Chunks', 'N/A')
                                    }
                                )
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.error("Failed to process PPT files. Please try again.")

    elif typepdf == "DOCX files":
        st.header("📄 DOCX Files", divider="blue")
        interaction_mode = render_voice_interaction_mode("docx_files")

        # Upload only (no saved groups UI)
        selected_group_id = None
        selected_file_type = None
        uploaded_files = st.file_uploader("Choose maximum of 50 DOCX", type=['docx'], accept_multiple_files=True)
        if uploaded_files and len(uploaded_files) > 50:
            st.error("You can upload a maximum of 50 DOCX files")
            return

        # Generate database name
        db_name_list = []
        for file in uploaded_files:
            original_name = file.name
            if len(original_name) > 20:
                name, ext = os.path.splitext(original_name)
                truncated_name = name[:20 - len(ext)] + ext
                db_name_list.append(os.path.splitext(truncated_name)[0])
            else:
                db_name_list.append(os.path.splitext(original_name)[0])

        combined_db_name = "_".join(db_name_list) if db_name_list else "docx_db"
        if len(combined_db_name) > 50:
            combined_db_name = combined_db_name[:50]

        if uploaded_files:
            st.write(f"Processing {len(uploaded_files)} files...")

            # Initialize embedding model
            embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=SecretStr(GOOGLE_API_KEY), model="models/embedding-001")

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Process files using unified function
            db, success = save_media_embeddings(uploaded_files, 'DOCX files', embedding_model, combined_db_name, progress_bar, status_text)

            if success and db:
                # Question Input (uploads only)
                question = get_voice_or_text_input(interaction_mode, "Enter your question about the DOCX files:", "docx_question")

                if question and db:
                    try:
                        # Show loading spinner while processing the question
                        with st.spinner("Processing your question... This may take a few moments."):
                            # Query Vector DB
                            response = rag_pipeline(question, model, combined_db_name, temperature, top_p, max_tokens)
                            if response:
                                json_content = response.content
                                if isinstance(json_content, str):
                                    json_content = json_content.strip("```json\n").strip("```")
                                    try:
                                        parsed_data = json.loads(json_content)
                                    except Exception:
                                        st.warning("Could not parse model response as JSON.")
                                        parsed_data = {}
                                else:
                                    st.warning("Model response was not a string and could not be parsed as JSON.")
                                    parsed_data = {}
                                st.subheader("Response:")
                                answer_text = parsed_data.get("Answer", "No answer found.")
                                st.write(answer_text)
                                render_voice_output(answer_text, interaction_mode)
                                st.write(f"Used Chunks: {parsed_data.get('Used Chunks', 'N/A')}")
                                st.write(f"Reasoning: {parsed_data.get('Reasoning', 'N/A')}")

                                # Save to CSV log
                                save_query_response_to_csv(
                                    user_query=question,
                                    bot_response=answer_text,
                                    query_type="DOCX RAG Query",
                                    model_used=model,
                                    temperature=temperature,
                                    additional_metadata={
                                        'interaction_mode': interaction_mode,
                                        'num_files': len(uploaded_files),
                                        'file_names': [f.name for f in uploaded_files],
                                        'similarity_score': parsed_data.get('Overall Similarity', 'N/A'),
                                        'used_chunks': parsed_data.get('Used Chunks', 'N/A'),
                                        'reasoning': parsed_data.get('Reasoning', 'N/A')
                                    }
                                )
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.error("Failed to process DOCX files. Please try again.")

    elif typepdf == "JSON files":
        st.header("📄 JSON Files", divider="blue")
        interaction_mode = render_voice_interaction_mode("json_files")

        # Upload only (no saved groups UI)
        selected_group_id = None
        selected_file_type = None
        uploaded_files = st.file_uploader("Choose maximum of 50 JSON", type='json', accept_multiple_files=True)
        if uploaded_files and len(uploaded_files) > 50:
            st.error("You can upload a maximum of 50 JSON files")
            return

        # Generate database name
        combined_db_name = build_combined_db_name_from_uploads(uploaded_files or [], "json_db")

        if uploaded_files:
            st.write(f"Processing {len(uploaded_files)} files...")

            # Initialize embedding model
            embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=SecretStr(GOOGLE_API_KEY), model="models/embedding-001")

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Process files using unified function
            db, success = save_media_embeddings(uploaded_files, 'JSON files', embedding_model, combined_db_name, progress_bar, status_text)

            if success and db:
                # Question Input (uploads only)
                question = get_voice_or_text_input(interaction_mode, "Enter your question about the JSON files:", "json_question")

                if question and db:
                    try:
                        # Show loading spinner while processing the question
                        with st.spinner("Processing your question... This may take a few moments."):
                            # Query Vector DB
                            response = rag_pipeline(question, model, combined_db_name, temperature, top_p, max_tokens)
                            if response:
                                json_content = response.content
                                if isinstance(json_content, str):
                                    json_content = json_content.strip("```json\n").strip("```")
                                    try:
                                        parsed_data = json.loads(json_content)
                                    except Exception:
                                        st.warning("Could not parse model response as JSON.")
                                        parsed_data = {}
                                else:
                                    st.warning("Model response was not a string and could not be parsed as JSON.")
                                    parsed_data = {}
                                st.subheader("Response:")
                                answer_text = parsed_data.get("Answer", "No answer found.")
                                st.write(answer_text)
                                render_voice_output(answer_text, interaction_mode)
                                st.write(f"Used Chunks: {parsed_data.get('Used Chunks', 'N/A')}")
                                st.write(f"Reasoning: {parsed_data.get('Reasoning', 'N/A')}")

                                # Save to CSV log
                                save_query_response_to_csv(
                                    user_query=question,
                                    bot_response=answer_text,
                                    query_type="JSON RAG Query",
                                    model_used=model,
                                    temperature=temperature,
                                    additional_metadata={
                                        'interaction_mode': interaction_mode,
                                        'num_files': len(uploaded_files),
                                        'file_names': [f.name for f in uploaded_files],
                                        'similarity_score': parsed_data.get('Overall Similarity', 'N/A'),
                                        'used_chunks': parsed_data.get('Used Chunks', 'N/A'),
                                        'reasoning': parsed_data.get('Reasoning', 'N/A')
                                    }
                                )
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.error("Failed to process JSON files. Please try again.")

    elif typepdf == "H files":
        st.header("📄 H Files", divider="blue")
        interaction_mode = render_voice_interaction_mode("h_files")

        # Upload only (no saved groups UI)
        selected_group_id = None
        selected_file_type = None
        uploaded_files = st.file_uploader("Choose maximum of 50 H files", type='h', accept_multiple_files=True)
        if uploaded_files and len(uploaded_files) > 50:
            st.error("You can upload a maximum of 50 H files")
            return

        # Generate database name
        db_name_list = []
        for file in uploaded_files:
            original_name = file.name
            if len(original_name) > 20:
                name, ext = os.path.splitext(original_name)
                truncated_name = name[:20 - len(ext)] + ext
                db_name_list.append(os.path.splitext(truncated_name)[0])
            else:
                db_name_list.append(os.path.splitext(original_name)[0])

        combined_db_name = "_".join(db_name_list) if db_name_list else "h_db"
        if len(combined_db_name) > 50:
            combined_db_name = combined_db_name[:50]

        if uploaded_files:
            st.write(f"Processing {len(uploaded_files)} files...")

            # Initialize embedding model
            embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=SecretStr(GOOGLE_API_KEY), model="models/embedding-001")

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Process files using unified function
            db, success = save_media_embeddings(uploaded_files, 'H files', embedding_model, combined_db_name, progress_bar, status_text)

            if success and db:
                # Question Input (uploads only)
                question = get_voice_or_text_input(interaction_mode, "Enter your question about the H files:", "h_question")

                if question and db:
                    try:
                        # Show loading spinner while processing the question
                        with st.spinner("Processing your question... This may take a few moments."):
                            # Query Vector DB
                            response = rag_pipeline(question, model, combined_db_name, temperature, top_p, max_tokens)
                            if response:
                                json_content = response.content
                                if isinstance(json_content, str):
                                    json_content = json_content.strip("```json\n").strip("```")
                                    try:
                                        parsed_data = json.loads(json_content)
                                    except Exception:
                                        st.warning("Could not parse model response as JSON.")
                                        parsed_data = {}
                                else:
                                    st.warning("Model response was not a string and could not be parsed as JSON.")
                                    parsed_data = {}
                                st.subheader("Response:")
                                answer_text = parsed_data.get("Answer", "No answer found.")
                                st.write(answer_text)
                                render_voice_output(answer_text, interaction_mode)
                                st.write(f"Used Chunks: {parsed_data.get('Used Chunks', 'N/A')}")
                                st.write(f"Reasoning: {parsed_data.get('Reasoning', 'N/A')}")

                                # Save to CSV log
                                save_query_response_to_csv(
                                    user_query=question,
                                    bot_response=answer_text,
                                    query_type="H Files RAG Query",
                                    model_used=model,
                                    temperature=temperature,
                                    additional_metadata={
                                        'interaction_mode': interaction_mode,
                                        'num_files': len(uploaded_files),
                                        'file_names': [f.name for f in uploaded_files],
                                        'similarity_score': parsed_data.get('Overall Similarity', 'N/A'),
                                        'used_chunks': parsed_data.get('Used Chunks', 'N/A'),
                                        'reasoning': parsed_data.get('Reasoning', 'N/A')
                                    }
                                )
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.error("Failed to process H files. Please try again.")

    elif typepdf == "XML files":
        st.header("📄 XML Files", divider="blue")
        interaction_mode = render_voice_interaction_mode("xml_files")

        # Upload only (no saved groups UI)
        selected_group_id = None
        selected_file_type = None
        uploaded_files = st.file_uploader("Choose maximum of 50 XML", type='xml', accept_multiple_files=True)
        if uploaded_files and len(uploaded_files) > 50:
            st.error("You can upload a maximum of 50 XML files")
            return

        # Generate database name
        db_name_list = []
        for file in uploaded_files:
            original_name = file.name
            if len(original_name) > 20:
                name, ext = os.path.splitext(original_name)
                truncated_name = name[:20 - len(ext)] + ext
                db_name_list.append(os.path.splitext(truncated_name)[0])
            else:
                db_name_list.append(os.path.splitext(original_name)[0])

        combined_db_name = "_".join(db_name_list) if db_name_list else "xml_db"
        if len(combined_db_name) > 50:
            combined_db_name = combined_db_name[:50]

        if uploaded_files:
            st.write(f"Processing {len(uploaded_files)} files...")

            # Initialize embedding model
            embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=SecretStr(GOOGLE_API_KEY), model="models/embedding-001")

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Process files using unified function
            db, success = save_media_embeddings(uploaded_files, 'XML files', embedding_model, combined_db_name, progress_bar, status_text)

            if success and db:
                # Question Input (uploads only)
                question = get_voice_or_text_input(interaction_mode, "Enter your question about the XML files:", "xml_question")

                if question and db:
                    try:
                        # Show loading spinner while processing the question
                        with st.spinner("Processing your question... This may take a few moments."):
                            # Query Vector DB
                            response = rag_pipeline(question, model, combined_db_name, temperature, top_p, max_tokens)
                            if response:
                                json_content = response.content
                                if isinstance(json_content, str):
                                    json_content = json_content.strip("```json\n").strip("```")
                                    try:
                                        parsed_data = json.loads(json_content)
                                    except Exception:
                                        st.warning("Could not parse model response as JSON.")
                                        parsed_data = {}
                                else:
                                    st.warning("Model response was not a string and could not be parsed as JSON.")
                                    parsed_data = {}
                                st.subheader("Response:")
                                answer_text = parsed_data.get("Answer", "No answer found.")
                                st.write(answer_text)
                                render_voice_output(answer_text, interaction_mode)
                                st.write(f"Used Chunks: {parsed_data.get('Used Chunks', 'N/A')}")
                                st.write(f"Reasoning: {parsed_data.get('Reasoning', 'N/A')}")

                                # Save to CSV log
                                save_query_response_to_csv(
                                    user_query=question,
                                    bot_response=answer_text,
                                    query_type="XML RAG Query",
                                    model_used=model,
                                    temperature=temperature,
                                    additional_metadata={
                                        'interaction_mode': interaction_mode,
                                        'num_files': len(uploaded_files),
                                        'file_names': [f.name for f in uploaded_files],
                                        'similarity_score': parsed_data.get('Overall Similarity', 'N/A'),
                                        'used_chunks': parsed_data.get('Used Chunks', 'N/A'),
                                        'reasoning': parsed_data.get('Reasoning', 'N/A')
                                    }
                                )
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.error("Failed to process XML files. Please try again.")

    elif typepdf == "Images":
        st.header("🖼️ Image Analysis", divider="blue")
        interaction_mode = render_voice_interaction_mode("image_analysis")
        upload_status = st.empty()
        progress_bar = st.empty()
        uploaded_images = st.file_uploader("Upload your image files.", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        if uploaded_images:
            upload_status.text("Uploading images...")
            progress = progress_bar.progress(0)
            for i in range(10):
                time.sleep(0.05)
                progress.progress((i + 1) * 10)
            upload_status.success(f"Successfully uploaded {len(uploaded_images)} images!")
            col_count = min(3, len(uploaded_images))
            cols = st.columns(col_count)
            for i, image_file in enumerate(uploaded_images):
                cols[i % col_count].image(image_file, caption=image_file.name, use_container_width=True)
            analysis_type = st.selectbox(
                "Select analysis type:",
                ["Custom Prompt", "Describe and Analyze", "Compare Images"],
                key="image_analysis_type"
            )
            if analysis_type == "Custom Prompt":
                prompt2 = get_voice_or_text_input(interaction_mode, "Enter your prompt for all uploaded images:", "image_custom_prompt")
                if prompt2 and st.button("Analyze with Custom Prompt", key="image_custom_btn"):
                    image_parts = []
                    for image_file in uploaded_images:
                        image_bytes = image_file.read()
                        image_part = {
                            "mime_type": image_file.type,
                            "data": image_bytes
                        }
                        image_parts.append(image_part)
                    generation_config = GenerationConfig(
                        temperature=temperature,
                        top_p=top_p,
                        max_output_tokens=max_tokens,
                    )
                    try:
                        model_instance = genai.GenerativeModel(
                            model_name=model,
                            generation_config=generation_config
                        )  # type: ignore[attr-defined]
                        with st.spinner("Processing your prompt... This may take a few moments."):
                            content_parts = image_parts + [prompt2]
                            response = model_instance.generate_content(content_parts)  # type: ignore[arg-type]
                            st.markdown(response.text)
                            render_voice_output(response.text, interaction_mode)
                            # Save to CSV log
                            save_query_response_to_csv(
                                user_query=prompt2,
                                bot_response=response.text,
                                query_type="Image Analysis",
                                model_used=model,
                                temperature=temperature,
                                additional_metadata={
                                    'interaction_mode': interaction_mode,
                                    'num_images': len(uploaded_images),
                                    'image_names': [img.name for img in uploaded_images]
                                }
                            )

                    except Exception as e:
                        st.error(f"Error generating content: {str(e)}")
            elif analysis_type == "Describe and Analyze":
                if st.button("Describe and Analyze Images", key="image_describe_btn"):
                    with st.spinner("Analyzing images... This may take a few moments."):
                        for i, image_file in enumerate(uploaded_images):
                            st.write(f"### Analysis for {image_file.name}")
                            image_bytes = image_file.read()
                            image_part = {
                                "mime_type": image_file.type,
                                "data": image_bytes
                            }
                            analysis_prompt = """
                            Describe and analyze this image in detail. Please include:
                            1. A general description of what's in the image
                            2. Key objects, people, or elements present
                            3. Notable visual characteristics (colors, composition, lighting)
                            4. Any text visible in the image
                            5. Context or setting of the image
                            6. Mood or atmosphere conveyed
                            Format your response with clear sections.
                            """
                            try:
                                generation_config = GenerationConfig(
                                    temperature=temperature,
                                    top_p=top_p,
                                    max_output_tokens=max_tokens,
                                )
                                model_instance = genai.GenerativeModel(
                                    model_name=model,
                                    generation_config=generation_config
                                )  # type: ignore[attr-defined]
                                response = model_instance.generate_content([image_part, analysis_prompt])
                                st.markdown(response.text)
                                render_voice_output(response.text, interaction_mode)
                                # Save to CSV log
                                save_query_response_to_csv(
                                    user_query=analysis_prompt,
                                    bot_response=response.text,
                                    query_type="Image Describe and Analyze",
                                    model_used=model,
                                    temperature=temperature,
                                    additional_metadata={
                                        'interaction_mode': interaction_mode,
                                        'image_name': image_file.name
                                    }
                                )
                                if i < len(uploaded_images) - 1:
                                    st.markdown("---")
                            except Exception as e:
                                st.error(f"Error analyzing {image_file.name}: {str(e)}")
            elif analysis_type == "Compare Images" and len(uploaded_images) > 1:
                if st.button("Compare Images", key="image_compare_btn"):
                    with st.spinner("Comparing images... This may take a few moments."):
                        image_parts = []
                        for image_file in uploaded_images:
                            image_bytes = image_file.read()
                            image_part = {
                                "mime_type": image_file.type,
                                "data": image_bytes
                            }
                            image_parts.append(image_part)
                        comparison_prompt = f"""
                        Compare these {len(uploaded_images)} images in detail. Please include:
                        1. Similarities between the images
                        2. Key differences between the images
                        3. Unique elements in each image
                        4. Which image might be more effective for different purposes
                        5. Overall comparative analysis
                        Format your response with clear sections for easy reading.
                        """
                        try:
                            generation_config = GenerationConfig(
                                temperature=temperature,
                                top_p=top_p,
                                max_output_tokens=max_tokens,
                            )
                            model_instance = genai.GenerativeModel(
                                model_name=model,
                                generation_config=generation_config
                            )  # type: ignore[attr-defined]
                            content_parts = image_parts + [comparison_prompt]
                            response = model_instance.generate_content(content_parts)
                            st.subheader("Image Comparison Results:")
                            st.markdown(response.text)
                            render_voice_output(response.text, interaction_mode)
                            # Save to CSV log
                            save_query_response_to_csv(
                                user_query=comparison_prompt,
                                bot_response=response.text,
                                query_type="Image Compare",
                                model_used=model,
                                temperature=temperature,
                                additional_metadata={
                                    'interaction_mode': interaction_mode,
                                    'num_images': len(uploaded_images),
                                    'image_names': [img.name for img in uploaded_images]
                                }
                            )
                        except Exception as e:
                            st.error(f"Error comparing images: {str(e)}")
            elif analysis_type == "Compare Images" and len(uploaded_images) <= 1:
                st.warning("Please upload at least 2 images for comparison.")

    elif typepdf == "Image editing":
        st.header("🛠️ Image Editing", divider="blue")
        interaction_mode = render_voice_interaction_mode("image_editing")

        # Base image input
        col_left, col_right = st.columns([2, 1])
        with col_left:
            uploaded_image = st.file_uploader("Upload a base image", type=["png", "jpg", "jpeg"], accept_multiple_files=False, key="image_edit_upload")

        edit_prompt = get_voice_or_text_input(
            interaction_mode,
            "Describe the edits to apply:",
            "image_edit_prompt",
            placeholder="e.g., Add a small knitted wizard hat and a Hawaii shirt"
        )

        # Preview uploaded image
        base_image_bytes = None
        base_image_name = None
        if uploaded_image is not None:
            base_image_bytes = uploaded_image.read()
            base_image_name = uploaded_image.name
            st.image(BytesIO(base_image_bytes), caption=f"Base: {uploaded_image.name}", use_container_width=True)

        col_gen1, col_clear1 = st.columns([1, 1])
        with col_gen1:
            generate_btn = st.button("Apply edits", type="primary", use_container_width=True, key="apply_image_edits")
        with col_clear1:
            if st.button("Clear", use_container_width=True, key="clear_image_edits"):
                st.session_state.pop("image_edit_upload", None)
                st.session_state.pop("image_edit_prompt", None)
                st.rerun()

        if generate_btn:
            if not base_image_bytes:
                st.warning("Please upload an image.")
            elif not edit_prompt:
                st.warning("Please enter an edit description.")
            else:
                try:
                    # Use new Google GenAI client for image generation/editing
                    ensure_genai_config()
                    client = genai_new.Client(api_key=GOOGLE_API_KEY)
                    image_model_id = "gemini-2.5-flash-image-preview"

                    # Open base image with PIL for compatibility
                    base_pil = Image.open(BytesIO(base_image_bytes))

                    with st.spinner("Generating edited image..."):
                        response = client.models.generate_content(
                            model=image_model_id,
                            contents=[edit_prompt, base_pil],
                        )

                    # Extract image bytes from response
                    image_parts = []
                    try:
                        candidates = getattr(response, "candidates", []) or []
                        if candidates:
                            parts = getattr(candidates[0].content, "parts", []) or []
                            for part in parts:
                                inline_data = getattr(part, "inline_data", None)
                                if inline_data and getattr(inline_data, "data", None):
                                    image_parts.append({
                                        "bytes": inline_data.data,
                                        "mime": getattr(inline_data, "mime_type", "image/png")
                                    })
                        elif getattr(response, "parts", None):
                            for part in response.parts:
                                inline_data = getattr(part, "inline_data", None)
                                if inline_data and getattr(inline_data, "data", None):
                                    image_parts.append({
                                        "bytes": inline_data.data,
                                        "mime": getattr(inline_data, "mime_type", "image/png")
                                    })
                    except Exception as _e:
                        st.error(f"Failed to parse model response: {_e}")

                    if image_parts:
                        img0 = Image.open(BytesIO(image_parts[0]["bytes"]))
                        st.image(img0, caption="Edited image", use_container_width=True)

                        # Offer download
                        mime = image_parts[0]["mime"] or "image/png"
                        ext = mimetypes.guess_extension(mime) or ".png"
                        out_name = f"edited_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
                        st.download_button("Download image", data=image_parts[0]["bytes"], file_name=out_name, mime=mime, use_container_width=True)

                        # Log interaction
                        save_query_response_to_csv(
                            user_query=edit_prompt,
                            bot_response=f"Edited image generated: {out_name}",
                            query_type="Image editing",
                            model_used=image_model_id,
                            additional_metadata={
                                "base_image": base_image_name,
                                "mime": mime
                            }
                        )
                    else:
                        st.warning("No image returned by the model. Try refining your prompt.")
                except Exception as e:
                    st.error(f"Error during image editing: {e}")

    elif typepdf == "Video, mp4 file":
        st.header("🎬 Video Analysis", divider="blue")
        interaction_mode = render_voice_interaction_mode("video_analysis")
        uploaded_videos = st.file_uploader("Upload your video files.", type="mp4", accept_multiple_files=True)
        if uploaded_videos:
            video_parts = []
            for video_file in uploaded_videos:
                # Read the video file
                video_bytes = video_file.read()

                # Create video part for Gemini
                video_part = {
                    "mime_type": video_file.type,
                    "data": video_bytes
                }
                video_parts.append(video_part)

            prompt3 = get_voice_or_text_input(interaction_mode, "Enter your prompt for all uploaded videos:", "video_prompt")
            if prompt3:
                try:
                    generation_config = GenerationConfig(
                        temperature=temperature,
                        top_p=top_p,
                        max_output_tokens=max_tokens,
                    )
                    # Store original model name for CSV logging
                    original_model_name = model
                    model = genai.GenerativeModel(
                        model_name=model,
                        generation_config=generation_config
                    )  # type: ignore[attr-defined]

                    # Create content parts list with videos and prompt
                    content_parts = video_parts + [prompt3]

                    # Generate content
                    response = model.generate_content(content_parts)

                    # Display the response
                    st.markdown(response.text)
                    render_voice_output(response.text, interaction_mode)


                    # Save to CSV log
                    save_query_response_to_csv(
                        user_query=prompt3,
                        bot_response=response.text,
                        query_type="Video Analysis",
                        model_used=original_model_name,
                        temperature=temperature,
                        additional_metadata={
                            'interaction_mode': interaction_mode,
                            'num_files': len(uploaded_videos),
                            'file_names': [f.name for f in uploaded_videos]
                        }
                    )

                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error generating content: {error_msg}")
                    st.error(f"Error: {error_msg}")

    elif typepdf == "Audio files":
        st.header("🎵 Audio Analysis", divider="blue")
        interaction_mode = render_voice_interaction_mode("audio_analysis")
        uploaded_audios = st.file_uploader("Upload your audio files.", type="wav", accept_multiple_files=True)
        if uploaded_audios:
            audio_parts = []
            for audio_file in uploaded_audios:
                # Read the audio file
                audio_bytes = audio_file.read()

                # Create audio part for Gemini
                audio_part = {
                    "mime_type": audio_file.type,
                    "data": audio_bytes
                }
                audio_parts.append(audio_part)

            prompt4 = get_voice_or_text_input(interaction_mode, "Enter your prompt for all uploaded audio files:", "audio_prompt")
            if prompt4:
                try:
                    generation_config = GenerationConfig(
                        temperature=temperature,
                        top_p=top_p,
                        max_output_tokens=max_tokens,
                    )
                    # Store original model name for CSV logging
                    original_model_name = model
                    model = genai.GenerativeModel(
                        model_name=model,
                        generation_config=generation_config
                    )  # type: ignore[attr-defined]

                    # Create content parts list with audios and prompt
                    content_parts = audio_parts + [prompt4]

                    # Generate content
                    response = model.generate_content(content_parts)

                    # Display the response
                    st.markdown(response.text)
                    render_voice_output(response.text, interaction_mode)

                    # Get confidence scores if available
                    try:
                        # confidence_scores = get_confidence_scores(response)
                        # st.write(f"Confidence score: {confidence_scores.get('average_score')}")

                        # Save to CSV log
                        save_query_response_to_csv(
                            user_query=prompt4,
                            bot_response=response.text,
                            query_type="Audio Analysis",
                            model_used=original_model_name,
                            temperature=temperature,
                            additional_metadata={
                                'interaction_mode': interaction_mode,
                                'num_files': len(uploaded_audios),
                                'file_names': [f.name for f in uploaded_audios]
                            }
                        )
                    except Exception as score_err:
                        logger.warning(f"Could not get confidence scores: {score_err}")

                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error generating content: {error_msg}")
                    st.error(f"Error: {error_msg}")

    elif typepdf == "DAT files":
        st.header("📄 DAT Files", divider="blue")
        interaction_mode = render_voice_interaction_mode("dat_files")

        # Upload only (no saved groups UI)
        uploaded_files = st.file_uploader("Choose maximum of 50 DAT files", type='dat', accept_multiple_files=True)
        if uploaded_files and len(uploaded_files) > 50:
            st.error("You can upload a maximum of 50 DAT files")
            return

        # Generate database name
        db_name_list = []
        for file in uploaded_files:
            original_name = file.name
            if len(original_name) > 20:
                name, ext = os.path.splitext(original_name)
                truncated_name = name[:20 - len(ext)] + ext
                db_name_list.append(os.path.splitext(truncated_name)[0])
            else:
                db_name_list.append(os.path.splitext(original_name)[0])

        combined_db_name = "_".join(db_name_list) if db_name_list else "dat_db"
        if len(combined_db_name) > 50:
            combined_db_name = combined_db_name[:50]

        if uploaded_files:
            st.write(f"Processing {len(uploaded_files)} files...")

            # Initialize embedding model
            embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=SecretStr(GOOGLE_API_KEY), model="models/embedding-001")

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Process files using unified function
            db, success = save_media_embeddings(uploaded_files, 'DAT files', embedding_model, combined_db_name, progress_bar, status_text)

            if success and db:
                # Question Input (uploads only)
                question = get_voice_or_text_input(interaction_mode, "Enter your question about the DAT files:", "dat_question")

                if question and db:
                    try:
                        # Show loading spinner while processing the question
                        with st.spinner("Processing your question... This may take a few moments."):
                            # Query Vector DB
                            response = rag_pipeline(question, model, combined_db_name, temperature, top_p, max_tokens)
                            if response:
                                json_content = response.content
                                if isinstance(json_content, str):
                                    json_content = json_content.strip("```json\n").strip("```")
                                    try:
                                        parsed_data = json.loads(json_content)
                                    except Exception:
                                        st.warning("Could not parse model response as JSON.")
                                        parsed_data = {}
                                else:
                                    st.warning("Model response was not a string and could not be parsed as JSON.")
                                    parsed_data = {}
                                st.subheader("Response:")
                                answer_text = parsed_data.get("Answer", "No answer found.")
                                st.write(answer_text)
                                render_voice_output(answer_text, interaction_mode)
                                st.write(f"Used Chunks: {parsed_data.get('Used Chunks', 'N/A')}")
                                st.write(f"Reasoning: {parsed_data.get('Reasoning', 'N/A')}")

                                # Save to CSV log
                                save_query_response_to_csv(
                                    user_query=question,
                                    bot_response=answer_text,
                                    query_type="DAT RAG Query",
                                    model_used=model,
                                    temperature=temperature,
                                    additional_metadata={
                                        'interaction_mode': interaction_mode,
                                        'num_files': len(uploaded_files),
                                        'file_names': [f.name for f in uploaded_files],
                                        'similarity_score': parsed_data.get('Overall Similarity', 'N/A'),
                                        'used_chunks': parsed_data.get('Used Chunks', 'N/A'),
                                        'reasoning': parsed_data.get('Reasoning', 'N/A')
                                    }
                                )
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.error("Failed to process DAT files. Please try again.")

    elif typepdf == "Markdown files":
        st.header("📄 Markdown Files", divider="blue")
        interaction_mode = render_voice_interaction_mode("md_files")

        # File upload (full width, vertical layout like PDF)
        uploaded_files = st.file_uploader("Choose maximum of 50 Markdown files", type='md', accept_multiple_files=True)
        if uploaded_files and len(uploaded_files) > 50:
            st.error("You can upload a maximum of 50 Markdown files")
            return

        # Generate database name
        db_name_list = []
        for file in uploaded_files:
            original_name = file.name
            if len(original_name) > 20:
                name, ext = os.path.splitext(original_name)
                truncated_name = name[:20 - len(ext)] + ext
                db_name_list.append(os.path.splitext(truncated_name)[0])
            else:
                db_name_list.append(os.path.splitext(original_name)[0])

        combined_db_name = "_".join(db_name_list) if db_name_list else "md_db"
        if len(combined_db_name) > 50:
            combined_db_name = combined_db_name[:50]

        if uploaded_files:
            st.write(f"Processing {len(uploaded_files)} files...")

            # Initialize embedding model
            embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=SecretStr(GOOGLE_API_KEY), model="models/embedding-001")

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Process files using unified function
            db, success = save_media_embeddings(uploaded_files, 'Markdown files', embedding_model, combined_db_name, progress_bar, status_text)

            if success and db:

                # Question Input
                question = get_voice_or_text_input(interaction_mode, "Enter your question about the Markdown files:", "md_question")

                if question and db:
                    try:
                        # Show loading spinner while processing the question
                        with st.spinner("Processing your question... This may take a few moments."):
                            # Query Vector DB
                            response = rag_pipeline(question, model, combined_db_name, temperature, top_p, max_tokens)
                            if response:
                                json_content = response.content
                                if isinstance(json_content, str):
                                    json_content = json_content.strip("```json\n").strip("```")
                                    try:
                                        parsed_data = json.loads(json_content)
                                    except Exception:
                                        st.warning("Could not parse model response as JSON.")
                                        parsed_data = {}
                                else:
                                    st.warning("Model response was not a string and could not be parsed as JSON.")
                                    parsed_data = {}
                                st.subheader("Response:")
                                answer_text = parsed_data.get("Answer", "No answer found.")
                                st.write(answer_text)
                                render_voice_output(answer_text, interaction_mode)
                                st.write(f"Used Chunks: {parsed_data.get('Used Chunks', 'N/A')}")
                                st.write(f"Reasoning: {parsed_data.get('Reasoning', 'N/A')}")

                                # Save to CSV log
                                save_query_response_to_csv(
                                    user_query=question,
                                    bot_response=answer_text,
                                    query_type="Markdown RAG Query",
                                    model_used=model,
                                    temperature=temperature,
                                    additional_metadata={
                                        'interaction_mode': interaction_mode,
                                        'num_files': len(uploaded_files),
                                        'file_names': [f.name for f in uploaded_files],
                                        'similarity_score': parsed_data.get('Overall Similarity', 'N/A'),
                                        'used_chunks': parsed_data.get('Used Chunks', 'N/A'),
                                        'reasoning': parsed_data.get('Reasoning', 'N/A')
                                    }
                                )
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.error("Failed to process Markdown files. Please try again.")

    elif typepdf == "Multiple Media":
        st.header("🔀 Multiple Media", divider="blue")
        interaction_mode = render_voice_interaction_mode("multi_media")

        # Upload only (no saved groups UI)
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "png", "jpg", "jpeg", "mp4", "wav", "txt", "docx", "pptx", "csv", "xml", "json", "yaml", "yml", "h", "xlsx", "xls", "dat", "md"],
            accept_multiple_files=True
        )

        use_pdf_rag = False

        if uploaded_files:
            # Always process files fresh to ensure all file types work correctly
            progress_bar = st.progress(0)
            status_text = st.empty()
            all_content_parts, file_info = process_multiple_media_files(uploaded_files, model, temperature, top_p, max_tokens, progress_bar, status_text)

            if all_content_parts:
                st.success("Files processed successfully! You can now ask questions about the content.")
                prompt = get_voice_or_text_input(interaction_mode, "Enter your question about the uploaded files:", "multi_media_question")

                if prompt:
                    # Check if only PDFs were uploaded
                    pdf_files = [f for f in uploaded_files if f.name.lower().endswith('.pdf')]
                    non_pdf_files = [f for f in uploaded_files if not f.name.lower().endswith('.pdf')]
                    is_pdf_only = len(pdf_files) > 0 and len(non_pdf_files) == 0

                    mm_pdf_db = st.session_state.get('multi_media_pdf_db')
                    use_pdf_rag = False

                    # Prefer PDF RAG whenever any PDFs are uploaded
                    if mm_pdf_db and len(pdf_files) > 0:
                        use_pdf_rag = True

                    # Use PDF RAG for PDF-only uploads or PDF-specific questions
                    if use_pdf_rag:
                        if is_pdf_only:
                            st.info("📄 PDF-only upload detected - using optimized PDF RAG pipeline for best accuracy")
                        else:
                            st.info("📄 PDFs detected - using PDF RAG pipeline for improved accuracy")
                        try:
                            response = rag_pipeline(prompt, model, mm_pdf_db, temperature, top_p, max_tokens)
                            if response:
                                json_content = response.content.strip("```json\n").strip("```")
                                parsed_data = json.loads(json_content)
                                answer_text = parsed_data.get("Answer", "No answer found.")
                                st.subheader("Response:")
                                st.markdown(answer_text)
                                render_voice_output(answer_text, interaction_mode)
                                save_query_response_to_csv(
                                    user_query=prompt,
                                    bot_response=answer_text,
                                    query_type="Multiple Media PDF RAG",
                                    model_used=model,
                                    temperature=temperature,
                                    additional_metadata={
                                        'interaction_mode': interaction_mode,
                                        'num_files': len(uploaded_files),
                                    'file_names': [f.name for f in uploaded_files],
                                    'similarity_score': parsed_data.get('Overall Similarity', 'N/A'),
                                    'used_chunks': parsed_data.get('Used Chunks', 'N/A'),
                                    'reasoning': parsed_data.get('Reasoning', 'N/A')
                                }
                            )
                            return
                        except Exception as e:
                            st.warning(f"RAG PDF processing failed, falling back to multimodal analysis: {e}")
                            use_pdf_rag = False

                # Use multimodal analysis for general questions or when PDF RAG fails
                if prompt and not use_pdf_rag:
                    # Show loading spinner while processing (generic multimodal)
                    with st.spinner("Processing your question... This may take a few moments."):
                        try:
                            # Create the model with generation config
                            generation_config = GenerationConfig(
                                temperature=temperature,
                                top_p=top_p,
                                max_output_tokens=max_tokens,
                            )

                            # Store original model name for CSV logging
                            original_model_name = model
                            ensure_genai_config()
                            ensure_genai_config()
                            ensure_genai_config()
                            ensure_genai_config()
                            model = genai.GenerativeModel(
                                model_name=model,
                                generation_config=generation_config
                            )  # type: ignore[attr-defined]

                            # Create the final prompt with context about the files
                            context_prompt = f"""I have uploaded the following files:
{', '.join(f'{name} ({type})' for name, type in file_info)}

Please analyze all these files together and answer the following question: {prompt}

Provide a comprehensive answer that takes into account information from all the uploaded files."""

                            # Process content in smaller chunks to prevent timeout
                            try:
                                # If there are too many content parts, process them in batches
                                if len(all_content_parts) > 10:
                                    st.write("Processing a large number of files. This may take some time...")

                                    # Split content parts into batches
                                    batch_size = 10
                                    all_responses = []

                                    for i in range(0, len(all_content_parts), batch_size):
                                        batch = all_content_parts[i:i+batch_size]
                                        st.write(f"Processing batch {i//batch_size + 1} of {(len(all_content_parts) + batch_size - 1)//batch_size}...")

                                        # Add a small delay between batches
                                        if i > 0:
                                            time.sleep(1)

                                        # Generate content for this batch
                                        batch_response = model.generate_content(batch)
                                        all_responses.append(batch_response.text)

                                    # Combine all responses
                                    combined_response = "\n\n".join(all_responses)

                                    # Display the combined response
                                    st.subheader("Response:")
                                    st.markdown(combined_response)
                                    render_voice_output(combined_response, interaction_mode)

                                    # Use the last response for confidence scores
                                    response = batch_response
                                else:
                                    # Generate content for all parts at once
                                    response = model.generate_content(all_content_parts + [context_prompt])  # type: ignore[arg-type]

                                    # Display the response
                                    st.subheader("Response:")
                                    st.markdown(response.text)
                                    render_voice_output(response.text, interaction_mode)

                                # Save to CSV log
                                save_query_response_to_csv(
                                    user_query=prompt,
                                    bot_response=response.text,
                                    query_type="Multiple Media Analysis",
                                    model_used=original_model_name,
                                    temperature=temperature,
                                    additional_metadata={
                                        'interaction_mode': interaction_mode,
                                        'num_files': len(uploaded_files),
                                        'file_names': [f.name for f in uploaded_files],
                                        'file_types': [f.type for f in uploaded_files]
                                    }
                                )
                            except Exception as e:
                                error_msg = str(e)
                                logger.error(f"Error generating content: {error_msg}")
                                st.error(f"Error: {error_msg}")

                                # Try a fallback approach with just the context prompt
                                try:
                                    st.write("Trying a simplified approach...")
                                    fallback_response = model.generate_content(context_prompt)
                                    st.subheader("Response (Simplified):")
                                    st.markdown(fallback_response.text)
                                    response = fallback_response
                                except Exception as fallback_error:
                                    st.error(f"Fallback approach also failed: {str(fallback_error)}")
                                    return
                        except Exception as e:
                            error_msg = str(e)
                            logger.error(f"Error in multiple media processing: {error_msg}")
                            st.error(f"Error: {error_msg}")
            else:
                st.error("No files were successfully processed. Please check your file formats and try again.")
        else:
            st.info("Please upload multiple media files or select a saved file group from the sidebar.")

    elif typepdf == "YAML files":
        st.header("📄 YAML Files", divider="blue")
        interaction_mode = render_voice_interaction_mode("yaml_files")

        # Upload only (no saved groups UI)
        selected_group_id = None
        selected_file_type = None
        uploaded_files = st.file_uploader("Choose maximum of 50 YAML files", type=['yaml', 'yml'], accept_multiple_files=True)
        if uploaded_files and len(uploaded_files) > 50:
            st.error("You can upload a maximum of 50 YAML files")
            return

        # Generate database name
        combined_db_name = build_combined_db_name_from_uploads(uploaded_files or [], "yaml_db")

        if uploaded_files:
            st.write(f"Processing {len(uploaded_files)} files...")

            # Initialize embedding model
            embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=SecretStr(GOOGLE_API_KEY), model="models/embedding-001")

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Process files using unified function
            db, success = save_media_embeddings(uploaded_files, 'YAML files', embedding_model, combined_db_name, progress_bar, status_text)

            if success and db:
                # Question Input (uploads only)
                question = get_voice_or_text_input(interaction_mode, "Enter your question about the YAML files:", "yaml_question")

                if question and db:
                    try:
                        # Show loading spinner while processing the question
                        with st.spinner("Processing your question... This may take a few moments."):
                            # Query Vector DB
                            response = rag_pipeline(question, model, combined_db_name, temperature, top_p, max_tokens)
                            if response:
                                json_content = response.content
                                if isinstance(json_content, str):
                                    json_content = json_content.strip("```json\n").strip("```")
                                    try:
                                        parsed_data = json.loads(json_content)
                                    except Exception:
                                        st.warning("Could not parse model response as JSON.")
                                        parsed_data = {}
                                else:
                                    st.warning("Model response was not a string and could not be parsed as JSON.")
                                    parsed_data = {}
                                st.subheader("Response:")
                                answer_text = parsed_data.get("Answer", "No answer found.")
                                st.write(answer_text)
                                render_voice_output(answer_text, interaction_mode)
                                st.write(f"Used Chunks: {parsed_data.get('Used Chunks', 'N/A')}")
                                st.write(f"Reasoning: {parsed_data.get('Reasoning', 'N/A')}")

                                # Save to CSV log
                                save_query_response_to_csv(
                                    user_query=question,
                                    bot_response=answer_text,
                                    query_type="YAML RAG Query",
                                    model_used=model,
                                    temperature=temperature,
                                    additional_metadata={
                                        'interaction_mode': interaction_mode,
                                        'num_files': len(uploaded_files),
                                        'file_names': [f.name for f in uploaded_files],
                                        'similarity_score': parsed_data.get('Overall Similarity', 'N/A'),
                                        'used_chunks': parsed_data.get('Used Chunks', 'N/A'),
                                        'reasoning': parsed_data.get('Reasoning', 'N/A')
                                    }
                                )
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.error("Failed to process YAML files. Please try again.")

    elif typepdf == "H files":
        uploaded_files = st.file_uploader("Choose maximum of 50 H files", type='h', accept_multiple_files=True)
        if uploaded_files and len(uploaded_files) > 50:
          st.error("You can upload a maximum of 50 H files")
          return
        db_name_list = []
        for file in uploaded_files:
            original_name = file.name
            if len(original_name) > 20:
                name, ext = os.path.splitext(original_name)
                truncated_name = name[:20 - len(ext)] + ext
                db_name_list.append(os.path.splitext(truncated_name)[0])  # Base name without extension
            else:
                db_name_list.append(os.path.splitext(original_name)[0])

        # Create combined_db_name and truncate to 100 characters if necessary
        combined_db_name = "_".join(db_name_list) if db_name_list else "h_db"
        if len(combined_db_name) > 50:
          combined_db_name = combined_db_name[:50]

        if uploaded_files:
            st.write(f"Processing {len(uploaded_files)} files...")

            # Initialize embedding model
            embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=SecretStr(GOOGLE_API_KEY), model="models/embedding-001")

            # Process and store embeddings
            db = embed_and_store_files(uploaded_files, embedding_model, combined_db_name)

            # Question Input
            question = get_voice_or_text_input(interaction_mode, "Enter your question about the H files:", "h_question")

            if question and db:
                # Show loading spinner while processing the question
                with st.spinner("Processing your question... This may take a few moments."):
                    # Query Vector DB (modify as needed)
                    response = rag_pipeline(question, model, combined_db_name, temperature, top_p, max_tokens)
                    if response:
                        json_content = response.content
                        if isinstance(json_content, str):
                            json_content = json_content.strip("```json\n").strip("```")
                            try:
                                parsed_data = json.loads(json_content)
                            except Exception:
                                st.warning("Could not parse model response as JSON.")
                                parsed_data = {}
                        else:
                            st.warning("Model response was not a string and could not be parsed as JSON.")
                            parsed_data = {}
                        st.subheader("Response:")
                        answer_text = parsed_data.get("Answer", "No answer found.")
                        st.write(answer_text)
                        render_voice_output(answer_text, interaction_mode)
                        # st.write(f"Similarity Score: {parsed_data.get('Overall Similarity', 'N/A')}")
                        st.write(f"Used Chunks: {parsed_data.get('Used Chunks', 'N/A')}")
                        st.write(f"Reasoning: {parsed_data.get('Reasoning', 'N/A')}")

                        # Save to CSV log
                        save_query_response_to_csv(
                            user_query=question,
                            bot_response=answer_text,
                            query_type="H Files RAG Query",
                            model_used=model,
                            temperature=temperature,
                            additional_metadata={
                                'interaction_mode': interaction_mode,
                                'num_files': len(uploaded_files),
                                'file_names': [f.name for f in uploaded_files],
                                'similarity_score': parsed_data.get('Overall Similarity', 'N/A'),
                                'used_chunks': parsed_data.get('Used Chunks', 'N/A'),
                                'reasoning': parsed_data.get('Reasoning', 'N/A')
                            }
                        )

    elif typepdf == "XML files":
        uploaded_files = st.file_uploader("Choose maximum of 50 XML", type='xml', accept_multiple_files=True)
        if uploaded_files and len(uploaded_files) > 50:
          st.error("You can upload a maximum of 50 XML files")
          return
        db_name_list = []
        for file in uploaded_files:
            original_name = file.name
            if len(original_name) > 20:
                name, ext = os.path.splitext(original_name)
                truncated_name = name[:20 - len(ext)] + ext
                db_name_list.append(os.path.splitext(truncated_name)[0])  # Base name without extension
            else:
                db_name_list.append(os.path.splitext(original_name)[0])

        # Create combined_db_name and truncate to 100 characters if necessary
        combined_db_name = "_".join(db_name_list) if db_name_list else "xml_db"
        if len(combined_db_name) > 50:
          combined_db_name = combined_db_name[:50]

        if uploaded_files:
            st.write(f"Processing {len(uploaded_files)} files...")

            # Initialize embedding model
            embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=SecretStr(GOOGLE_API_KEY), model="models/embedding-001")

            # Process and store embeddings
            db = embed_and_store_files(uploaded_files, embedding_model, combined_db_name)

            # Question Input
            question = get_voice_or_text_input(interaction_mode, "Enter your question about the XML files:", "xml_question")

            if question and db:
                # Show loading spinner while processing the question
                with st.spinner("Processing your question... This may take a few moments."):
                    # Query Vector DB (modify as needed)
                    response = rag_pipeline(question, model, combined_db_name, temperature, top_p, max_tokens)
                    if response:
                        json_content = response.content
                        if isinstance(json_content, str):
                            json_content = json_content.strip("```json\n").strip("```")
                            try:
                                parsed_data = json.loads(json_content)
                            except Exception:
                                st.warning("Could not parse model response as JSON.")
                                parsed_data = {}
                        else:
                            st.warning("Model response was not a string and could not be parsed as JSON.")
                            parsed_data = {}
                        st.subheader("Response:")
                        answer_text = parsed_data.get("Answer", "No answer found.")
                        st.write(answer_text)
                        render_voice_output(answer_text, interaction_mode)
                        # st.write(f"Similarity Score: {parsed_data.get('Overall Similarity', 'N/A')}")
                        st.write(f"Used Chunks: {parsed_data.get('Used Chunks', 'N/A')}")
                        st.write(f"Reasoning: {parsed_data.get('Reasoning', 'N/A')}")

                        # Save to CSV log
                        save_query_response_to_csv(
                            user_query=question,
                            bot_response=answer_text,
                            query_type="XML RAG Query",
                            model_used=model,
                            temperature=temperature,
                            additional_metadata={
                                'interaction_mode': interaction_mode,
                                'num_files': len(uploaded_files),
                                'file_names': [f.name for f in uploaded_files],
                                'similarity_score': parsed_data.get('Overall Similarity', 'N/A'),
                                'used_chunks': parsed_data.get('Used Chunks', 'N/A'),
                                'reasoning': parsed_data.get('Reasoning', 'N/A')
                            }
                        )

    elif typepdf == "Images":
        upload_status = st.empty()
        progress_bar = st.empty()
        uploaded_images = st.file_uploader("Upload your image files.", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        if uploaded_images:
            upload_status.text("Uploading images...")
            progress = progress_bar.progress(0)
            for i in range(10):
                time.sleep(0.05)
                progress.progress((i + 1) * 10)
            upload_status.success(f"Successfully uploaded {len(uploaded_images)} images!")
            col_count = min(3, len(uploaded_images))
            cols = st.columns(col_count)
            for i, image_file in enumerate(uploaded_images):
                cols[i % col_count].image(image_file, caption=image_file.name, use_container_width=True)
            analysis_type = st.selectbox(
                "Select analysis type:",
                ["Custom Prompt", "Describe and Analyze", "Compare Images"],
                key="image_analysis_type"
            )
            if analysis_type == "Custom Prompt":
                prompt2 = get_voice_or_text_input(interaction_mode, "Enter your prompt for all uploaded images:", "image_custom_prompt")
                if prompt2 and st.button("Analyze with Custom Prompt", key="image_custom_btn"):
                    image_parts = []
                    for image_file in uploaded_images:
                        image_bytes = image_file.read()
                        image_part = {
                            "mime_type": image_file.type,
                            "data": image_bytes
                        }
                        image_parts.append(image_part)
                    generation_config = GenerationConfig(
                        temperature=temperature,
                        top_p=top_p,
                        max_output_tokens=max_tokens,
                    )
                    try:
                        model_instance = genai.GenerativeModel(
                            model_name=model,
                            generation_config=generation_config
                        )  # type: ignore[attr-defined]
                        with st.spinner("Processing your prompt... This may take a few moments."):
                            content_parts = image_parts + [prompt2]
                            response = model_instance.generate_content(content_parts)  # type: ignore[arg-type]
                            st.markdown(response.text)
                            render_voice_output(response.text, interaction_mode)
                            # Save to CSV log
                            save_query_response_to_csv(
                                user_query=prompt2,
                                bot_response=response.text,
                                query_type="Image Analysis",
                                model_used=model,
                                temperature=temperature,
                                additional_metadata={
                                    'interaction_mode': interaction_mode,
                                    'num_images': len(uploaded_images),
                                    'image_names': [img.name for img in uploaded_images]
                                }
                            )
                    except Exception as e:
                        st.error(f"Error generating content: {str(e)}")
            elif analysis_type == "Describe and Analyze":
                if st.button("Describe and Analyze Images", key="image_describe_btn"):
                    with st.spinner("Analyzing images... This may take a few moments."):
                        for i, image_file in enumerate(uploaded_images):
                            st.write(f"### Analysis for {image_file.name}")
                            image_bytes = image_file.read()
                            image_part = {
                                "mime_type": image_file.type,
                                "data": image_bytes
                            }
                            analysis_prompt = """
                            Describe and analyze this image in detail. Please include:
                            1. A general description of what's in the image
                            2. Key objects, people, or elements present
                            3. Notable visual characteristics (colors, composition, lighting)
                            4. Any text visible in the image
                            5. Context or setting of the image
                            6. Mood or atmosphere conveyed
                            Format your response with clear sections.
                            """
                            try:
                                generation_config = GenerationConfig(
                                    temperature=temperature,
                                    top_p=top_p,
                                    max_output_tokens=max_tokens,
                                )
                                model_instance = genai.GenerativeModel(
                                    model_name=model,
                                    generation_config=generation_config
                                )  # type: ignore[attr-defined]
                                response = model_instance.generate_content([image_part, analysis_prompt])
                                st.markdown(response.text)
                                render_voice_output(response.text, interaction_mode)
                                # Save to CSV log
                                save_query_response_to_csv(
                                    user_query=analysis_prompt,
                                    bot_response=response.text,
                                    query_type="Image Describe and Analyze",
                                    model_used=model,
                                    temperature=temperature,
                                    additional_metadata={
                                        'interaction_mode': interaction_mode,
                                        'image_name': image_file.name
                                    }
                                )
                                if i < len(uploaded_images) - 1:
                                    st.markdown("---")
                            except Exception as e:
                                st.error(f"Error analyzing {image_file.name}: {str(e)}")
            elif analysis_type == "Compare Images" and len(uploaded_images) > 1:
                if st.button("Compare Images", key="image_compare_btn"):
                    with st.spinner("Comparing images... This may take a few moments."):
                        image_parts = []
                        for image_file in uploaded_images:
                            image_bytes = image_file.read()
                            image_part = {
                                "mime_type": image_file.type,
                                "data": image_bytes
                            }
                            image_parts.append(image_part)
                        comparison_prompt = f"""
                        Compare these {len(uploaded_images)} images in detail. Please include:
                        1. Similarities between the images
                        2. Key differences between the images
                        3. Unique elements in each image
                        4. Which image might be more effective for different purposes
                        5. Overall comparative analysis
                        Format your response with clear sections for easy reading.
                        """
                        try:
                            generation_config = GenerationConfig(
                                temperature=temperature,
                                top_p=top_p,
                                max_output_tokens=max_tokens,
                            )
                            model_instance = genai.GenerativeModel(
                                model_name=model,
                                generation_config=generation_config
                            )  # type: ignore[attr-defined]
                            content_parts = image_parts + [comparison_prompt]
                            response = model_instance.generate_content(content_parts)
                            st.subheader("Image Comparison Results:")
                            st.markdown(response.text)
                            render_voice_output(response.text, interaction_mode)
                            # Save to CSV log
                            save_query_response_to_csv(
                                user_query=comparison_prompt,
                                bot_response=response.text,
                                query_type="Image Compare",
                                model_used=model,
                                temperature=temperature,
                                additional_metadata={
                                    'interaction_mode': interaction_mode,
                                    'num_images': len(uploaded_images),
                                    'image_names': [img.name for img in uploaded_images]
                                }
                            )
                        except Exception as e:
                            st.error(f"Error comparing images: {str(e)}")
            elif analysis_type == "Compare Images" and len(uploaded_images) <= 1:
                st.warning("Please upload at least 2 images for comparison.")

    elif typepdf == "Video, mp4 file":
        uploaded_videos = st.file_uploader("Upload your video files.", type="mp4", accept_multiple_files=True)
        if uploaded_videos:
            video_parts = []
            for video_file in uploaded_videos:
                # Read the video file
                video_bytes = video_file.read()

                # Create video part for Gemini
                video_part = {
                    "mime_type": video_file.type,
                    "data": video_bytes
                }
                video_parts.append(video_part)

            prompt3 = get_voice_or_text_input(interaction_mode, "Enter your prompt for all uploaded videos:", "video_prompt")
            if prompt3:
                try:
                    generation_config = GenerationConfig(
                        temperature=temperature,
                        top_p=top_p,
                        max_output_tokens=max_tokens,
                    )
                    model = genai.GenerativeModel(
                        model_name=model,
                        generation_config=generation_config
                    )  # type: ignore[attr-defined]

                    # Create content parts list with videos and prompt
                    content_parts = video_parts + [prompt3]

                    # Generate content
                    response = model.generate_content(content_parts)

                    # Display the response
                    st.markdown(response.text)
                    render_voice_output(response.text, interaction_mode)

                    # Get confidence scores if available
                    try:
                        # confidence_scores = get_confidence_scores(response)
                        # st.write(f"Confidence score: {confidence_scores.get('average_score')}")

                        # Save to CSV log
                        save_query_response_to_csv(
                            user_query=prompt3,
                            bot_response=response.text,
                            query_type="Video Analysis",
                            model_used=model,
                            temperature=temperature,
                            additional_metadata={
                                'interaction_mode': interaction_mode,
                                'num_files': len(uploaded_videos),
                                'file_names': [f.name for f in uploaded_videos]
                            }
                        )
                    except Exception as score_err:
                        logger.warning(f"Could not get confidence scores: {score_err}")

                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error generating content: {error_msg}")
                    st.error(f"Error: {error_msg}")

    elif typepdf == "Audio files":
        st.header("🎵 Audio Analysis", divider="blue")
        interaction_mode = render_voice_interaction_mode("audio_analysis")
        uploaded_audios = st.file_uploader("Upload your audio files.", type="wav", accept_multiple_files=True)
        if uploaded_audios:
            audio_parts = []
            for audio_file in uploaded_audios:
                # Read the audio file
                audio_bytes = audio_file.read()

                # Create audio part for Gemini
                audio_part = {
                    "mime_type": audio_file.type,
                    "data": audio_bytes
                }
                audio_parts.append(audio_part)

            prompt4 = get_voice_or_text_input(interaction_mode, "Enter your prompt for all uploaded audio files:", "audio_prompt")
            if prompt4:
                try:
                    generation_config = GenerationConfig(
                        temperature=temperature,
                        top_p=top_p,
                        max_output_tokens=max_tokens,
                    )
                    model = genai.GenerativeModel(
                        model_name=model,
                        generation_config=generation_config
                    )  # type: ignore[attr-defined]

                    # Create content parts list with audios and prompt
                    content_parts = audio_parts + [prompt4]

                    # Generate content
                    response = model.generate_content(content_parts)

                    # Display the response
                    st.markdown(response.text)
                    render_voice_output(response.text, interaction_mode)

                    # Get confidence scores if available
                    try:
                        # confidence_scores = get_confidence_scores(response)
                        # st.write(f"Confidence score: {confidence_scores.get('average_score')}")

                        # Save to CSV log
                        save_query_response_to_csv(
                            user_query=prompt4,
                            bot_response=response.text,
                            query_type="Audio Analysis",
                            model_used=model,
                            temperature=temperature,
                            additional_metadata={
                                'interaction_mode': interaction_mode,
                                'num_files': len(uploaded_audios),
                                'file_names': [f.name for f in uploaded_audios]
                            }
                        )
                    except Exception as score_err:
                        logger.warning(f"Could not get confidence scores: {score_err}")

                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error generating content: {error_msg}")
                    st.error(f"Error: {error_msg}")

    elif typepdf == "DAT files":
        uploaded_files = st.file_uploader("Choose maximum of 50 DAT files", type='dat', accept_multiple_files=True)
        if uploaded_files and len(uploaded_files) > 50:
            st.error("You can upload a maximum of 50 DAT files")
            return
        db_name_list = []
        for file in uploaded_files:
            original_name = file.name
            if len(original_name) > 20:
                name, ext = os.path.splitext(original_name)
                truncated_name = name[:20 - len(ext)] + ext
                db_name_list.append(os.path.splitext(truncated_name)[0])
            else:
                db_name_list.append(os.path.splitext(original_name)[0])
        combined_db_name = "_".join(db_name_list) if db_name_list else "dat_db"
        if len(combined_db_name) > 50:
            combined_db_name = combined_db_name[:50]
        if uploaded_files:
            st.write(f"Processing {len(uploaded_files)} files...")
            embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=SecretStr(GOOGLE_API_KEY), model="models/embedding-001")
            db = embed_and_store_files(uploaded_files, embedding_model, combined_db_name)
            question = get_voice_or_text_input(interaction_mode, "Enter your question about the DAT files:", "dat_question")
            if question and db:
                try:
                    # Show loading spinner while processing the question
                    with st.spinner("Processing your question... This may take a few moments."):
                        # Query Vector DB (modify as needed)
                        response = rag_pipeline(question, model, combined_db_name, temperature, top_p, max_tokens)
                        if response:
                            json_content = response.content
                            if isinstance(json_content, str):
                                json_content = json_content.strip("```json\n").strip("```")
                                try:
                                    parsed_data = json.loads(json_content)
                                except Exception:
                                    st.warning("Could not parse model response as JSON.")
                                    parsed_data = {}
                            else:
                                st.warning("Model response was not a string and could not be parsed as JSON.")
                                parsed_data = {}
                            st.subheader("Response:")
                            answer_text = parsed_data.get("Answer", "No answer found.")
                            st.write(answer_text)
                            render_voice_output(answer_text, interaction_mode)
                            # st.write(f"Similarity Score: {parsed_data.get('Overall Similarity', 'N/A')}")
                            st.write(f"Used Chunks: {parsed_data.get('Used Chunks', 'N/A')}")
                            st.write(f"Reasoning: {parsed_data.get('Reasoning', 'N/A')}")

                        # Save to CSV log
                        save_query_response_to_csv(
                            user_query=question,
                            bot_response=answer_text,
                            query_type="DAT RAG Query",
                            model_used=model,
                            temperature=temperature,
                            additional_metadata={
                                'interaction_mode': interaction_mode,
                                'num_files': len(uploaded_files),
                                'file_names': [f.name for f in uploaded_files],
                                'similarity_score': parsed_data.get('Overall Similarity', 'N/A'),
                                'used_chunks': parsed_data.get('Used Chunks', 'N/A'),
                                'reasoning': parsed_data.get('Reasoning', 'N/A')
                            }
                        )
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    elif typepdf == "Markdown files":
        st.header("📄 Markdown Files", divider="blue")
        interaction_mode = render_voice_interaction_mode("md_files")
        uploaded_files = st.file_uploader("Choose maximum of 50 Markdown files", type='md', accept_multiple_files=True)
        if uploaded_files and len(uploaded_files) > 50:
            st.error("You can upload a maximum of 50 Markdown files")
            return
        db_name_list = []
        for file in uploaded_files:
            original_name = file.name
            if len(original_name) > 20:
                name, ext = os.path.splitext(original_name)
                truncated_name = name[:20 - len(ext)] + ext
                db_name_list.append(os.path.splitext(truncated_name)[0])
            else:
                db_name_list.append(os.path.splitext(original_name)[0])
        combined_db_name = "_".join(db_name_list) if db_name_list else "md_db"
        if len(combined_db_name) > 50:
            combined_db_name = combined_db_name[:50]
        if uploaded_files:
            st.write(f"Processing {len(uploaded_files)} files...")
            embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=SecretStr(GOOGLE_API_KEY), model="models/embedding-001")
            db = embed_and_store_files(uploaded_files, embedding_model, combined_db_name)
            question = get_voice_or_text_input(interaction_mode, "Enter your question about the Markdown files:", "md_question")
            if question and db:
                try:
                    # Show loading spinner while processing the question
                    with st.spinner("Processing your question... This may take a few moments."):
                        # Query Vector DB (modify as needed)
                        response = rag_pipeline(question, model, combined_db_name, temperature, top_p, max_tokens)
                        if response:
                            json_content = response.content
                            if isinstance(json_content, str):
                                json_content = json_content.strip("```json\n").strip("```")
                                try:
                                    parsed_data = json.loads(json_content)
                                except Exception:
                                    st.warning("Could not parse model response as JSON.")
                                    parsed_data = {}
                            else:
                                st.warning("Model response was not a string and could not be parsed as JSON.")
                                parsed_data = {}
                            st.subheader("Response:")
                            answer_text = parsed_data.get("Answer", "No answer found.")
                            st.write(answer_text)
                            render_voice_output(answer_text, interaction_mode)
                            # st.write(f"Similarity Score: {parsed_data.get('Overall Similarity', 'N/A')}")
                            st.write(f"Used Chunks: {parsed_data.get('Used Chunks', 'N/A')}")
                            st.write(f"Reasoning: {parsed_data.get('Reasoning', 'N/A')}")

                        # Save to CSV log
                        save_query_response_to_csv(
                            user_query=question,
                            bot_response=answer_text,
                            query_type="Markdown RAG Query",
                            model_used=model,
                            temperature=temperature,
                            additional_metadata={
                                'interaction_mode': interaction_mode,
                                'num_files': len(uploaded_files),
                                'file_names': [f.name for f in uploaded_files],
                                'similarity_score': parsed_data.get('Overall Similarity', 'N/A'),
                                'used_chunks': parsed_data.get('Used Chunks', 'N/A'),
                                'reasoning': parsed_data.get('Reasoning', 'N/A')
                            }
                        )
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    elif typepdf == "Multiple Media":
        st.write("Upload multiple types of media files (PDF, Images, Videos, Audio, etc.)")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "png", "jpg", "jpeg", "mp4", "wav", "txt", "docx", "pptx", "csv", "xml", "json", "yaml", "yml", "h", "xlsx", "xls", "dat", "md"],
            accept_multiple_files=True
        )
        if uploaded_files:
            progress_bar = st.progress(0)
            status_text = st.empty()
            all_content_parts, file_info = process_multiple_media_files(uploaded_files, model, temperature, top_p, max_tokens, progress_bar, status_text)

            if uploaded_files:
                st.success("Files processed successfully! You can now ask questions about the content.")
                prompt = get_voice_or_text_input(interaction_mode, "Enter your question about the uploaded files:", "multi_media_question")

                if prompt:
                    # Show loading spinner while processing
                    with st.spinner("Processing your question... This may take a few moments."):
                        try:
                            # Create the model with generation config
                            generation_config = GenerationConfig(
                                temperature=temperature,
                                top_p=top_p,
                                max_output_tokens=max_tokens,
                            )

                            model = genai.GenerativeModel(
                                model_name=model,
                                generation_config=generation_config
                            )  # type: ignore[attr-defined]

                            # Create the final prompt with context about the files
                            context_prompt = f"""I have uploaded the following files:
{', '.join(f'{name} ({type})' for name, type in file_info)}

Please analyze all these files together and answer the following question: {prompt}

Provide a comprehensive answer that takes into account information from all the uploaded files."""

                            # Process content in smaller chunks to prevent timeout
                            try:
                                # If there are too many content parts, process them in batches
                                if len(all_content_parts) > 10:
                                    st.write("Processing a large number of files. This may take some time...")

                                    # Split content parts into batches
                                    batch_size = 10
                                    all_responses = []

                                    for i in range(0, len(all_content_parts), batch_size):
                                        batch = all_content_parts[i:i+batch_size]
                                        st.write(f"Processing batch {i//batch_size + 1} of {(len(all_content_parts) + batch_size - 1)//batch_size}...")

                                        # Add a small delay between batches
                                        if i > 0:
                                            time.sleep(1)

                                        # Generate content for this batch
                                        batch_response = model.generate_content(batch)
                                        all_responses.append(batch_response.text)

                                    # Combine all responses
                                    combined_response = "\n\n".join(all_responses)

                                    # Display the combined response
                                    st.subheader("Response:")
                                    st.markdown(combined_response)
                                    render_voice_output(combined_response, interaction_mode)

                                    # Use the last response for confidence scores
                                    response = batch_response
                                else:
                                    # Generate content for all parts at once
                                    response = model.generate_content(all_content_parts + [context_prompt])  # type: ignore[arg-type]

                                    # Display the response
                                    st.subheader("Response:")
                                    st.markdown(response.text)
                                    render_voice_output(response.text, interaction_mode)
                            except Exception as e:
                                error_msg = str(e)
                                logger.error(f"Error generating content: {error_msg}")
                                st.error(f"Error: {error_msg}")

                                # Try a fallback approach with just the context prompt
                                try:
                                    st.write("Trying a simplified approach...")
                                    fallback_response = model.generate_content(context_prompt)
                                    st.subheader("Response (Simplified):")
                                    st.markdown(fallback_response.text)
                                    response = fallback_response
                                except Exception as fallback_error:
                                    st.error(f"Fallback approach also failed: {str(fallback_error)}")
                                    return
                        except Exception as e:
                            error_msg = str(e)
                            logger.error(f"Error in multiple media processing: {error_msg}")
                            st.error(f"Error: {error_msg}")
            else:
                st.error("No files were successfully processed. Please check your file formats and try again.")

    elif typepdf == "Voice to Voice":
        st.header("🎙️ Voice to Voice", divider="blue")
        st.write("Speak your question and get an audio response back!")

        # Initialize chat state for Voice to Voice
        if 'v2v_history' not in st.session_state:
            st.session_state['v2v_history'] = []
        if 'v2v_input' not in st.session_state:
            st.session_state['v2v_input'] = ""

        # Voice input section
        with st.container(border=True):
            st.subheader("🎤 Voice Input", anchor=False)
            st.caption("Tap the microphone, speak your question, then tap again to stop. The audio will be transcribed and processed.")

            audio_bytes = audio_recorder(
                text="🎤 Tap to Record / Stop",
                icon_name="microphone",
                icon_size="2x",
                neutral_color="#6C757D",
                recording_color="#FF4B4B",
                key="voice_to_voice_recorder",
            )

        voice_text = ""
        transcription_error = False

        if audio_bytes:
            # Playback preview
            st.audio(audio_bytes, format='audio/wav', start_time=0)

            # Transcribe audio to text
            if sr is not None and AudioSegment is not None:
                with st.spinner("🎧 Transcribing your voice..."):
                    try:
                        audio_file = io.BytesIO(audio_bytes)
                        audio_file.seek(0)
                        audio_segment = AudioSegment.from_file(audio_file, format="wav")
                        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                        audio_segment.export(temp_wav.name, format="wav")
                        temp_wav.close()

                        recognizer = sr.Recognizer()
                        with sr.AudioFile(temp_wav.name) as source:
                            audio_data = recognizer.record(source)
                            voice_text = recognizer.recognize_google(audio_data)

                        os.remove(temp_wav.name)
                        st.success(f"✅ Transcription: {voice_text}")

                    except Exception as e:
                        transcription_error = True
                        st.warning(f"⚠️ Automatic transcription failed: {e}. Please type what you said.")
            else:
                transcription_error = True
                st.info("ℹ️ SpeechRecognition or pydub not installed. Please type what you said.")

            # Manual transcription fallback
            if transcription_error or (audio_bytes and not voice_text):
                manual_text = st.text_area("Manual transcription (type what you said):", "", height=100)
                if manual_text:
                    voice_text = manual_text

            # Prefill the text input from transcription without auto-sending
            if voice_text:
                st.session_state['v2v_input'] = voice_text
                st.info("Transcription added to the text box below. You can edit it and click Generate Response.")

        # Display chat history
        if st.session_state['v2v_history']:
            st.subheader("💬 Chat History")
            for i, turn in enumerate(st.session_state['v2v_history']):
                with st.container(border=True):
                    col1, col2 = st.columns([1, 20])
                    with col1:
                        st.markdown("👤")
                    with col2:
                        st.markdown(f"**You:** {turn.get('user', '')}")

                    col1, col2 = st.columns([1, 20])
                    with col1:
                        st.markdown("🤖")
                    with col2:
                        st.markdown(f"**Assistant:** {turn.get('assistant', '')}")
            st.markdown("---")

        # Always-visible text input and action buttons
        user_input = st.text_area(
            "Type your message (or use the mic above):",
            value=st.session_state.get('v2v_input', ''),
            key="v2v_text_area",
            placeholder="Type here, or use the mic, then click Generate Response",
            height=100
        )
        # Keep session state in sync
        st.session_state['v2v_input'] = user_input

        col_gen, col_clear = st.columns([2, 1])
        with col_gen:
            gen_clicked = st.button("Generate Response", key="v2v_generate_btn", use_container_width=True)
        with col_clear:
            clear_clicked = st.button("Clear Chat History", key="v2v_clear_btn", use_container_width=True)

        if clear_clicked:
            st.session_state['v2v_history'] = []
            st.session_state['v2v_input'] = ""
            st.rerun()

        if gen_clicked:
            if not st.session_state['v2v_input'].strip():
                st.warning("Please enter a message or use the microphone.")
            else:
                question_text = st.session_state['v2v_input'].strip()
                with st.spinner("🤖 Generating response..."):
                    try:
                        generation_config = GenerationConfig(
                            temperature=temperature,
                            top_p=top_p,
                            max_output_tokens=max_tokens,
                        )
                        model_instance = genai.GenerativeModel(
                            model_name=model,
                            generation_config=generation_config
                        )
                        prompt = f"You are having a friendly conversation with someone. They just said: '{question_text}'. Respond naturally and conversationally as if you're talking to a friend. Keep your response brief and natural."
                        response = model_instance.generate_content(prompt)
                        response_text = response.text

                        # Show text response
                        st.subheader("📝 Text Response:")
                        st.success(response_text)

                        # TTS output
                        st.subheader("🔊 Audio Response:")
                        if gTTS is not None:
                            try:
                                with st.spinner("🎵 Converting response to speech..."):
                                    clean_response = re.sub(r'^(agent|expert):\s*', '', response_text, flags=re.I)
                                    tts = gTTS(text=clean_response, lang='en')
                                    tts_fp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                                    tts.save(tts_fp.name)
                                    tts_fp.close()
                                    with open(tts_fp.name, 'rb') as audio_file:
                                        audio_bytes_out = audio_file.read()
                                    st.audio(audio_bytes_out, format='audio/mp3')
                                    os.remove(tts_fp.name)
                            except Exception as tts_error:
                                st.error(f"❌ Text-to-speech generation failed: {tts_error}")
                        else:
                            st.warning("⚠️ gTTS not available. Install gTTS for voice output: `pip install gtts`")

                        # Append to chat history
                        st.session_state['v2v_history'].append({
                            'user': question_text,
                            'assistant': response_text
                        })

                        # Save to CSV log
                        save_query_response_to_csv(
                            user_query=question_text,
                            bot_response=response_text,
                            query_type="Voice to Voice",
                            model_used=model,
                            temperature=temperature,
                            additional_metadata={
                                'interaction_mode': 'Voice',
                                'transcription_method': 'automatic' if not transcription_error else 'manual',
                                'tts_enabled': gTTS is not None
                            }
                        )
                    except Exception as e:
                        st.error(f"❌ Error processing your question: {str(e)}")

        # Instructions for users
        with st.expander("ℹ️ How to use Voice to Voice"):
            st.markdown("""
            **Steps:**
            1. Click the microphone button to start recording
            2. Speak your question clearly
            3. Click the microphone button again to stop recording
            4. Wait for automatic transcription (or type manually if needed)
            5. Get both text and audio responses

            **Tips:**
            - Speak clearly and at a moderate pace
            - Ensure you have a good microphone
            - The system works best in quiet environments
            - You can type manually if transcription fails
            """)

    elif typepdf == "Multiple Media":
        st.write("Upload multiple types of media files (PDF, Images, Videos, Audio, etc.)")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "png", "jpg", "jpeg", "mp4", "wav", "txt", "docx", "pptx", "csv", "xml", "json", "yaml", "yml", "h", "xlsx", "xls", "dat", "md"],
            accept_multiple_files=True
        )
        if uploaded_files:
            progress_bar = st.progress(0)
            status_text = st.empty()
            all_content_parts, file_info = process_multiple_media_files(uploaded_files, model, temperature, top_p, max_tokens, progress_bar, status_text)

            if uploaded_files:
                st.success("Files processed successfully! You can now ask questions about the content.")
                prompt = get_voice_or_text_input(interaction_mode, "Enter your question about the uploaded files:", "multi_media_question")

                if prompt:
                    # Show loading spinner while processing
                    with st.spinner("Processing your question... This may take a few moments."):
                        try:
                            # Create the model with generation config
                            generation_config = GenerationConfig(
                                temperature=temperature,
                                top_p=top_p,
                                max_output_tokens=max_tokens,
                            )

                            model = genai.GenerativeModel(
                                model_name=model,
                                generation_config=generation_config
                            )  # type: ignore[attr-defined]

                            # Create the final prompt with context about the files
                            context_prompt = f"""I have uploaded the following files:
{', '.join(f'{name} ({type})' for name, type in file_info)}

Please analyze all these files together and answer the following question: {prompt}

Provide a comprehensive answer that takes into account information from all the uploaded files."""

                            # Process content in smaller chunks to prevent timeout
                            try:
                                # If there are too many content parts, process them in batches
                                if len(all_content_parts) > 10:
                                    st.write("Processing a large number of files. This may take some time...")

                                    # Split content parts into batches
                                    batch_size = 10
                                    all_responses = []

                                    for i in range(0, len(all_content_parts), batch_size):
                                        batch = all_content_parts[i:i+batch_size]
                                        st.write(f"Processing batch {i//batch_size + 1} of {(len(all_content_parts) + batch_size - 1)//batch_size}...")

                                        # Add a small delay between batches
                                        if i > 0:
                                            time.sleep(1)

                                        # Generate content for this batch
                                        batch_response = model.generate_content(batch)
                                        all_responses.append(batch_response.text)

                                    # Combine all responses
                                    combined_response = "\n\n".join(all_responses)

                                    # Display the combined response
                                    st.subheader("Response:")
                                    st.markdown(combined_response)
                                    render_voice_output(combined_response, interaction_mode)

                                    # Use the last response for confidence scores
                                    response = batch_response
                                else:
                                    # Generate content for all parts at once
                                    response = model.generate_content(all_content_parts + [context_prompt])  # type: ignore[arg-type]

                                    # Display the response
                                    st.subheader("Response:")
                                    st.markdown(response.text)
                                    render_voice_output(response.text, interaction_mode)
                            except Exception as e:
                                error_msg = str(e)
                                logger.error(f"Error generating content: {error_msg}")
                                st.error(f"Error: {error_msg}")

                                # Try a fallback approach with just the context prompt
                                try:
                                    st.write("Trying a simplified approach...")
                                    fallback_response = model.generate_content(context_prompt)
                                    st.subheader("Response (Simplified):")
                                    st.markdown(fallback_response.text)
                                    response = fallback_response
                                except Exception as fallback_error:
                                    st.error(f"Fallback approach also failed: {str(fallback_error)}")
                                    return
                        except Exception as e:
                            error_msg = str(e)
                            logger.error(f"Error in multiple media processing: {error_msg}")
                            st.error(f"Error: {error_msg}")
            else:
                st.error("No files were successfully processed. Please check your file formats and try again.")

    elif typepdf == "Generate Images":
        st.header("🎨 Image Generation", divider="blue")
        interaction_mode = render_voice_interaction_mode("generate_image")
        prompt_gen = get_voice_or_text_input(interaction_mode, "Enter your image prompt:", "image_generate_prompt")
        with st.expander("Generation Options", expanded=False):
            model_option = st.selectbox("Choose generation model", ["Imagen 4", "Imagen 4 Fast", "Imagen 4 Ultra"], key="img_gen_model")
            aspect_ratio = st.selectbox("Aspect ratio", ["1:1","9:16","16:9","3:4","4:3"], key="img_gen_aspect")
            num_images = st.slider("Number of images", 1, 4, 1, key="img_gen_num")
            safety_filter = st.selectbox("Safety filter level", ["BLOCK_LOW_AND_ABOVE","BLOCK_MEDIUM_AND_ABOVE","BLOCK_ONLY_HIGH","BLOCK_NONE"], key="img_gen_sf")
            person_generation = st.selectbox("Person generation", ["DONT_ALLOW","ALLOW_ADULT","ALLOW_ALL"], key="img_gen_pg")

        if st.button("Generate Image(s)", key="img_gen_btn") and prompt_gen:
            model_map = {
                "Imagen 4": "imagen-4.0-generate-preview-06-06",
                "Imagen 4 Fast": "imagen-4.0-fast-generate-preview-06-06",
                "Imagen 4 Ultra": "imagen-4.0-ultra-generate-preview-06-06"
            }
            generation_model_id = model_map[model_option]
            # Ultra model supports only one image
            if generation_model_id.endswith("ultra-generate-preview-06-06"):
                num_images = 1
            try:
                # Use API key for image generation (no project/location needed)
                VERTEX_API_KEY = "AIzaSyAm62k_mpMqcdQn8q3IqCjaOP4bCtE-qzE"

                client = genai_new.Client(api_key=VERTEX_API_KEY)
                image_response = client.models.generate_images(
                    model=generation_model_id,
                    prompt=prompt_gen,
                    config=types.GenerateImagesConfig(
                        aspect_ratio=aspect_ratio,
                        number_of_images=num_images,
                        safety_filter_level=safety_filter,
                        person_generation=person_generation
                    ),
                )
                st.subheader("Generated Images")
                for idx, img_obj in enumerate(image_response.generated_images, start=1):
                    # Handle the new image format from google.genai
                    if hasattr(img_obj, 'image') and hasattr(img_obj.image, '_pil_image'):
                        pil_img = img_obj.image._pil_image
                        st.image(pil_img, caption=f"Image {idx}", use_container_width=True)

                        # Add download button for the image
                        img_buffer = BytesIO()
                        pil_img.save(img_buffer, format='PNG')
                        img_buffer.seek(0)
                        st.download_button(
                            label=f"Download Image {idx}",
                            data=img_buffer.getvalue(),
                            file_name=f"generated_image_{idx}.png",
                            mime="image/png",
                            key=f"download_btn_{idx}"
                        )
                    elif hasattr(img_obj, 'image'):
                        st.image(img_obj.image, caption=f"Image {idx}", use_container_width=True)

                        # Add download button for the image (fallback)
                        try:
                            img_buffer = BytesIO()
                            img_obj.image.save(img_buffer, format='PNG')
                            img_buffer.seek(0)
                            st.download_button(
                                label=f"Download Image {idx}",
                                data=img_buffer.getvalue(),
                                file_name=f"generated_image_{idx}.png",
                                mime="image/png",
                                key=f"download_btn_fallback_{idx}"
                            )
                        except Exception as download_error:
                            st.warning(f"Could not create download button for Image {idx}: {download_error}")
                    else:
                        st.error(f"Could not display image {idx}")
                render_voice_output("Here are your generated images.", interaction_mode)
                # Save to CSV log
                save_query_response_to_csv(
                    user_query=prompt_gen,
                    bot_response=f"Generated {len(image_response.generated_images)} images.",
                    query_type="Image Generation",
                    model_used=generation_model_id,
                    additional_metadata={
                        'interaction_mode': interaction_mode,
                        'num_images': len(image_response.generated_images),
                        'aspect_ratio': aspect_ratio,
                        'model_option': model_option,
                        'safety_filter': safety_filter,
                        'person_generation': person_generation
                    }
                )
            except Exception as e:
                st.error(f"Error generating images: {e}")



    elif typepdf == "Images":
        upload_status = st.empty()
        progress_bar = st.empty()
        uploaded_images = st.file_uploader("Upload your image files.", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        if uploaded_images:
            upload_status.text("Uploading images...")
            progress = progress_bar.progress(0)
            for i in range(10):
                time.sleep(0.05)
                progress.progress((i + 1) * 10)
            upload_status.success(f"Successfully uploaded {len(uploaded_images)} images!")
            col_count = min(3, len(uploaded_images))
            cols = st.columns(col_count)
            for i, image_file in enumerate(uploaded_images):
                cols[i % col_count].image(image_file, caption=image_file.name, use_container_width=True)
            analysis_type = st.selectbox(
                "Select analysis type:",
                ["Custom Prompt", "Describe and Analyze", "Compare Images"],
                key="image_analysis_type"
            )
            if analysis_type == "Custom Prompt":
                prompt2 = get_voice_or_text_input(interaction_mode, "Enter your prompt for all uploaded images:", "image_custom_prompt")
                if prompt2 and st.button("Analyze with Custom Prompt", key="image_custom_btn"):
                    image_parts = []
                    for image_file in uploaded_images:
                        image_bytes = image_file.read()
                        image_part = {
                            "mime_type": image_file.type,
                            "data": image_bytes
                        }
                        image_parts.append(image_part)
                    generation_config = GenerationConfig(
                        temperature=temperature,
                        top_p=top_p,
                        max_output_tokens=max_tokens,
                    )
                    try:
                        model_instance = genai.GenerativeModel(
                            model_name=model,
                            generation_config=generation_config
                        )  # type: ignore[attr-defined]
                        with st.spinner("Processing your prompt... This may take a few moments."):
                            content_parts = image_parts + [prompt2]
                            response = model_instance.generate_content(content_parts)  # type: ignore[arg-type]
                            st.markdown(response.text)
                            render_voice_output(response.text, interaction_mode)
                            # Save to CSV log
                            save_query_response_to_csv(
                                user_query=prompt2,
                                bot_response=response.text,
                                query_type="Image Analysis",
                                model_used=model,
                                temperature=temperature,
                                additional_metadata={
                                    'interaction_mode': interaction_mode,
                                    'num_images': len(uploaded_images),
                                    'image_names': [img.name for img in uploaded_images]
                                }
                            )
                    except Exception as e:
                        st.error(f"Error generating content: {str(e)}")
            elif analysis_type == "Describe and Analyze":
                if st.button("Describe and Analyze Images", key="image_describe_btn"):
                    with st.spinner("Analyzing images... This may take a few moments."):
                        for i, image_file in enumerate(uploaded_images):
                            st.write(f"### Analysis for {image_file.name}")
                            image_bytes = image_file.read()
                            image_part = {
                                "mime_type": image_file.type,
                                "data": image_bytes
                            }
                            analysis_prompt = """
                            Describe and analyze this image in detail. Please include:
                            1. A general description of what's in the image
                            2. Key objects, people, or elements present
                            3. Notable visual characteristics (colors, composition, lighting)
                            4. Any text visible in the image
                            5. Context or setting of the image
                            6. Mood or atmosphere conveyed
                            Format your response with clear sections.
                            """
                            try:
                                generation_config = GenerationConfig(
                                    temperature=temperature,
                                    top_p=top_p,
                                    max_output_tokens=max_tokens,
                                )
                                model_instance = genai.GenerativeModel(
                                    model_name=model,
                                    generation_config=generation_config
                                )  # type: ignore[attr-defined]
                                response = model_instance.generate_content([image_part, analysis_prompt])
                                st.markdown(response.text)
                                render_voice_output(response.text, interaction_mode)
                                # Save to CSV log
                                save_query_response_to_csv(
                                    user_query=analysis_prompt,
                                    bot_response=response.text,
                                    query_type="Image Describe and Analyze",
                                    model_used=model,
                                    temperature=temperature,
                                    additional_metadata={
                                        'interaction_mode': interaction_mode,
                                        'image_name': image_file.name
                                    }
                                )
                                if i < len(uploaded_images) - 1:
                                    st.markdown("---")
                            except Exception as e:
                                st.error(f"Error analyzing {image_file.name}: {str(e)}")
            elif analysis_type == "Compare Images" and len(uploaded_images) > 1:
                if st.button("Compare Images", key="image_compare_btn"):
                    with st.spinner("Comparing images... This may take a few moments."):
                        image_parts = []
                        for image_file in uploaded_images:
                            image_bytes = image_file.read()
                            image_part = {
                                "mime_type": image_file.type,
                                "data": image_bytes
                            }
                            image_parts.append(image_part)
                        comparison_prompt = f"""
                        Compare these {len(uploaded_images)} images in detail. Please include:
                        1. Similarities between the images
                        2. Key differences between the images
                        3. Unique elements in each image
                        4. Which image might be more effective for different purposes
                        5. Overall comparative analysis
                        Format your response with clear sections for easy reading.
                        """
                        try:
                            generation_config = GenerationConfig(
                                temperature=temperature,
                                top_p=top_p,
                                max_output_tokens=max_tokens,
                            )
                            model_instance = genai.GenerativeModel(
                                model_name=model,
                                generation_config=generation_config
                            )  # type: ignore[attr-defined]
                            content_parts = image_parts + [comparison_prompt]
                            response = model_instance.generate_content(content_parts)
                            st.subheader("Image Comparison Results:")
                            st.markdown(response.text)
                            render_voice_output(response.text, interaction_mode)
                            # Save to CSV log
                            save_query_response_to_csv(
                                user_query=comparison_prompt,
                                bot_response=response.text,
                                query_type="Image Compare",
                                model_used=model,
                                temperature=temperature,
                                additional_metadata={
                                    'interaction_mode': interaction_mode,
                                    'num_images': len(uploaded_images),
                                    'image_names': [img.name for img in uploaded_images]
                                }
                            )
                        except Exception as e:
                            st.error(f"Error comparing images: {str(e)}")
            elif analysis_type == "Compare Images" and len(uploaded_images) <= 1:
                st.warning("Please upload at least 2 images for comparison.")

    elif typepdf == "Video, mp4 file":
        uploaded_videos = st.file_uploader("Upload your video files.", type="mp4", accept_multiple_files=True)
        if uploaded_videos:
            video_parts = []
            for video_file in uploaded_videos:
                # Read the video file
                video_bytes = video_file.read()

                # Create video part for Gemini
                video_part = {
                    "mime_type": video_file.type,
                    "data": video_bytes
                }
                video_parts.append(video_part)

            prompt3 = get_voice_or_text_input(interaction_mode, "Enter your prompt for all uploaded videos:", "video_prompt")
            if prompt3:
                try:
                    generation_config = GenerationConfig(
                        temperature=temperature,
                        top_p=top_p,
                        max_output_tokens=max_tokens,
                    )
                    model = genai.GenerativeModel(
                        model_name=model,
                        generation_config=generation_config
                    )  # type: ignore[attr-defined]

                    # Create content parts list with videos and prompt
                    content_parts = video_parts + [prompt3]

                    # Generate content
                    response = model.generate_content(content_parts)

                    # Display the response
                    st.markdown(response.text)
                    render_voice_output(response.text, interaction_mode)

                    # Get confidence scores if available
                    try:
                        # confidence_scores = get_confidence_scores(response)
                        # st.write(f"Confidence score: {confidence_scores.get('average_score')}")

                        # Save to CSV log
                        save_query_response_to_csv(
                            user_query=prompt3,
                            bot_response=response.text,
                            query_type="Video Analysis",
                            model_used=model,
                            temperature=temperature,
                            additional_metadata={
                                'interaction_mode': interaction_mode,
                                'num_files': len(uploaded_videos),
                                'file_names': [f.name for f in uploaded_videos]
                            }
                        )
                    except Exception as score_err:
                        logger.warning(f"Could not get confidence scores: {score_err}")

                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error generating content: {error_msg}")
                    st.error(f"Error: {error_msg}")

    elif typepdf == "Audio files":
        st.header("🎵 Audio Analysis", divider="blue")
        interaction_mode = render_voice_interaction_mode("audio_analysis")
        uploaded_audios = st.file_uploader("Upload your audio files.", type="wav", accept_multiple_files=True)
        if uploaded_audios:
            audio_parts = []
            for audio_file in uploaded_audios:
                # Read the audio file
                audio_bytes = audio_file.read()

                # Create audio part for Gemini
                audio_part = {
                    "mime_type": audio_file.type,
                    "data": audio_bytes
                }
                audio_parts.append(audio_part)

            prompt4 = get_voice_or_text_input(interaction_mode, "Enter your prompt for all uploaded audio files:", "audio_prompt")
            if prompt4:
                try:
                    generation_config = GenerationConfig(
                        temperature=temperature,
                        top_p=top_p,
                        max_output_tokens=max_tokens,
                    )
                    model = genai.GenerativeModel(
                        model_name=model,
                        generation_config=generation_config
                    )  # type: ignore[attr-defined]

                    # Create content parts list with audios and prompt
                    content_parts = audio_parts + [prompt4]

                    # Generate content
                    response = model.generate_content(content_parts)

                    # Display the response
                    st.markdown(response.text)
                    render_voice_output(response.text, interaction_mode)

                    # Get confidence scores if available
                    try:
                        # confidence_scores = get_confidence_scores(response)
                        # st.write(f"Confidence score: {confidence_scores.get('average_score')}")

                        # Save to CSV log
                        save_query_response_to_csv(
                            user_query=prompt4,
                            bot_response=response.text,
                            query_type="Audio Analysis",
                            model_used=model,
                            temperature=temperature,
                            additional_metadata={
                                'interaction_mode': interaction_mode,
                                'num_files': len(uploaded_audios),
                                'file_names': [f.name for f in uploaded_audios]
                            }
                        )
                    except Exception as score_err:
                        logger.warning(f"Could not get confidence scores: {score_err}")

                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error generating content: {error_msg}")
                    st.error(f"Error: {error_msg}")

    elif typepdf == "DAT files":
        uploaded_files = st.file_uploader("Choose maximum of 50 DAT files", type='dat', accept_multiple_files=True)
        if uploaded_files and len(uploaded_files) > 50:
            st.error("You can upload a maximum of 50 DAT files")
            return
        db_name_list = []
        for file in uploaded_files:
            original_name = file.name
            if len(original_name) > 20:
                name, ext = os.path.splitext(original_name)
                truncated_name = name[:20 - len(ext)] + ext
                db_name_list.append(os.path.splitext(truncated_name)[0])
            else:
                db_name_list.append(os.path.splitext(original_name)[0])
        combined_db_name = "_".join(db_name_list) if db_name_list else "dat_db"
        if len(combined_db_name) > 50:
            combined_db_name = combined_db_name[:50]
        if uploaded_files:
            st.write(f"Processing {len(uploaded_files)} files...")
            embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=SecretStr(GOOGLE_API_KEY), model="models/embedding-001")
            db = embed_and_store_files(uploaded_files, embedding_model, combined_db_name)
            question = get_voice_or_text_input(interaction_mode, "Enter your question about the DAT files:", "dat_question")
            if question and db:
                try:
                    # Show loading spinner while processing the question
                    with st.spinner("Processing your question... This may take a few moments."):
                        # Query Vector DB (modify as needed)
                        response = rag_pipeline(question, model, combined_db_name, temperature, top_p, max_tokens)
                        if response:
                            json_content = response.content
                            if isinstance(json_content, str):
                                json_content = json_content.strip("```json\n").strip("```")
                                try:
                                    parsed_data = json.loads(json_content)
                                except Exception:
                                    st.warning("Could not parse model response as JSON.")
                                    parsed_data = {}
                            else:
                                st.warning("Model response was not a string and could not be parsed as JSON.")
                                parsed_data = {}
                            st.subheader("Response:")
                            answer_text = parsed_data.get("Answer", "No answer found.")
                            st.write(answer_text)
                            render_voice_output(answer_text, interaction_mode)
                            # st.write(f"Similarity Score: {parsed_data.get('Overall Similarity', 'N/A')}")
                            st.write(f"Used Chunks: {parsed_data.get('Used Chunks', 'N/A')}")
                            st.write(f"Reasoning: {parsed_data.get('Reasoning', 'N/A')}")

                        # Save to CSV log
                        save_query_response_to_csv(
                            user_query=question,
                            bot_response=answer_text,
                            query_type="DAT RAG Query",
                            model_used=model,
                            temperature=temperature,
                            additional_metadata={
                                'interaction_mode': interaction_mode,
                                'num_files': len(uploaded_files),
                                'file_names': [f.name for f in uploaded_files],
                                'similarity_score': parsed_data.get('Overall Similarity', 'N/A'),
                                'used_chunks': parsed_data.get('Used Chunks', 'N/A'),
                                'reasoning': parsed_data.get('Reasoning', 'N/A')
                            }
                        )
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    elif typepdf == "Markdown files":
        st.header("📄 Markdown Files", divider="blue")
        interaction_mode = render_voice_interaction_mode("md_files")
        uploaded_files = st.file_uploader("Choose maximum of 50 Markdown files", type='md', accept_multiple_files=True)
        if uploaded_files and len(uploaded_files) > 50:
            st.error("You can upload a maximum of 50 Markdown files")
            return
        db_name_list = []
        for file in uploaded_files:
            original_name = file.name
            if len(original_name) > 20:
                name, ext = os.path.splitext(original_name)
                truncated_name = name[:20 - len(ext)] + ext
                db_name_list.append(os.path.splitext(truncated_name)[0])
            else:
                db_name_list.append(os.path.splitext(original_name)[0])
        combined_db_name = "_".join(db_name_list) if db_name_list else "md_db"
        if len(combined_db_name) > 50:
            combined_db_name = combined_db_name[:50]
        if uploaded_files:
            st.write(f"Processing {len(uploaded_files)} files...")
            embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=SecretStr(GOOGLE_API_KEY), model="models/embedding-001")
            db = embed_and_store_files(uploaded_files, embedding_model, combined_db_name)
            question = get_voice_or_text_input(interaction_mode, "Enter your question about the Markdown files:", "md_question")
            if question and db:
                try:
                    # Show loading spinner while processing the question
                    with st.spinner("Processing your question... This may take a few moments."):
                        # Query Vector DB (modify as needed)
                        response = rag_pipeline(question, model, combined_db_name, temperature, top_p, max_tokens)
                        if response:
                            json_content = response.content
                            if isinstance(json_content, str):
                                json_content = json_content.strip("```json\n").strip("```")
                                try:
                                    parsed_data = json.loads(json_content)
                                except Exception:
                                    st.warning("Could not parse model response as JSON.")
                                    parsed_data = {}
                            else:
                                st.warning("Model response was not a string and could not be parsed as JSON.")
                                parsed_data = {}
                            st.subheader("Response:")
                            answer_text = parsed_data.get("Answer", "No answer found.")
                            st.write(answer_text)
                            render_voice_output(answer_text, interaction_mode)
                            # st.write(f"Similarity Score: {parsed_data.get('Overall Similarity', 'N/A')}")
                            st.write(f"Used Chunks: {parsed_data.get('Used Chunks', 'N/A')}")
                            st.write(f"Reasoning: {parsed_data.get('Reasoning', 'N/A')}")

                        # Save to CSV log
                        save_query_response_to_csv(
                            user_query=question,
                            bot_response=answer_text,
                            query_type="Markdown RAG Query",
                            model_used=model,
                            temperature=temperature,
                            additional_metadata={
                                'interaction_mode': interaction_mode,
                                'num_files': len(uploaded_files),
                                'file_names': [f.name for f in uploaded_files],
                                'similarity_score': parsed_data.get('Overall Similarity', 'N/A'),
                                'used_chunks': parsed_data.get('Used Chunks', 'N/A'),
                                'reasoning': parsed_data.get('Reasoning', 'N/A')
                            }
                        )
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    elif typepdf == "Multiple Media":
        st.write("Upload multiple types of media files (PDF, Images, Videos, Audio, etc.)")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "png", "jpg", "jpeg", "mp4", "wav", "txt", "docx", "pptx", "csv", "xml", "json", "yaml", "yml", "h", "xlsx", "xls", "dat", "md"],
            accept_multiple_files=True
        )
        if uploaded_files:
            progress_bar = st.progress(0)
            status_text = st.empty()
            all_content_parts, file_info = process_multiple_media_files(uploaded_files, model, temperature, top_p, max_tokens, progress_bar, status_text)

            if uploaded_files:
                st.success("Files processed successfully! You can now ask questions about the content.")
                prompt = get_voice_or_text_input(interaction_mode, "Enter your question about the uploaded files:", "multi_media_question")

                if prompt:
                    # Show loading spinner while processing
                    with st.spinner("Processing your question... This may take a few moments."):
                        try:
                            # Create the model with generation config
                            generation_config = GenerationConfig(
                                temperature=temperature,
                                top_p=top_p,
                                max_output_tokens=max_tokens,
                            )

                            model = genai.GenerativeModel(
                                model_name=model,
                                generation_config=generation_config
                            )  # type: ignore[attr-defined]

                            # Create the final prompt with context about the files
                            context_prompt = f"""I have uploaded the following files:
{', '.join(f'{name} ({type})' for name, type in file_info)}

Please analyze all these files together and answer the following question: {prompt}

Provide a comprehensive answer that takes into account information from all the uploaded files."""

                            # Process content in smaller chunks to prevent timeout
                            try:
                                # If there are too many content parts, process them in batches
                                if len(all_content_parts) > 10:
                                    st.write("Processing a large number of files. This may take some time...")

                                    # Split content parts into batches
                                    batch_size = 10
                                    all_responses = []

                                    for i in range(0, len(all_content_parts), batch_size):
                                        batch = all_content_parts[i:i+batch_size]
                                        st.write(f"Processing batch {i//batch_size + 1} of {(len(all_content_parts) + batch_size - 1)//batch_size}...")

                                        # Add a small delay between batches
                                        if i > 0:
                                            time.sleep(1)

                                        # Generate content for this batch
                                        batch_response = model.generate_content(batch)
                                        all_responses.append(batch_response.text)

                                    # Combine all responses
                                    combined_response = "\n\n".join(all_responses)

                                    # Display the combined response
                                    st.subheader("Response:")
                                    st.markdown(combined_response)
                                    render_voice_output(combined_response, interaction_mode)

                                    # Use the last response for confidence scores
                                    response = batch_response
                                else:
                                    # Generate content for all parts at once
                                    response = model.generate_content(all_content_parts + [context_prompt])  # type: ignore[arg-type]

                                    # Display the response
                                    st.subheader("Response:")
                                    st.markdown(response.text)
                                    render_voice_output(response.text, interaction_mode)
                            except Exception as e:
                                error_msg = str(e)
                                logger.error(f"Error generating content: {error_msg}")
                                st.error(f"Error: {error_msg}")

                                # Try a fallback approach with just the context prompt
                                try:
                                    st.write("Trying a simplified approach...")
                                    fallback_response = model.generate_content(context_prompt)
                                    st.subheader("Response (Simplified):")
                                    st.markdown(fallback_response.text)
                                    response = fallback_response
                                except Exception as fallback_error:
                                    st.error(f"Fallback approach also failed: {str(fallback_error)}")
                                    return
                        except Exception as e:
                            error_msg = str(e)
                            logger.error(f"Error in multiple media processing: {error_msg}")
                            st.error(f"Error: {error_msg}")
            else:
                st.error("No files were successfully processed. Please check your file formats and try again.")

    elif typepdf == "Generate Images":
        st.header("🎨 Image Generation", divider="blue")
        interaction_mode = render_voice_interaction_mode("generate_image2")
        prompt_gen = get_voice_or_text_input(interaction_mode, "Enter your image prompt:", "image_generate_prompt_2")
        with st.expander("Generation Options", expanded=False):
            model_option = st.selectbox("Choose generation model", ["Imagen 4", "Imagen 4 Fast", "Imagen 4 Ultra"], key="img_gen_model2")
            aspect_ratio = st.selectbox("Aspect ratio", ["1:1","9:16","16:9","3:4","4:3"], key="img_gen_aspect2")
            num_images = st.slider("Number of images", 1, 4, 1, key="img_gen_num2")
            safety_filter = st.selectbox("Safety filter level", ["BLOCK_LOW_AND_ABOVE","BLOCK_MEDIUM_AND_ABOVE","BLOCK_ONLY_HIGH","BLOCK_NONE"], key="img_gen_sf2")
            person_generation = st.selectbox("Person generation", ["DONT_ALLOW","ALLOW_ADULT","ALLOW_ALL"], key="img_gen_pg2")

        if st.button("Generate Image(s)", key="img_gen_btn2") and prompt_gen:
            model_map = {
                "Imagen 4": "imagen-4.0-generate-preview-06-06",
                "Imagen 4 Fast": "imagen-4.0-fast-generate-preview-06-06",
                "Imagen 4 Ultra": "imagen-4.0-ultra-generate-preview-06-06"
            }
            generation_model_id = model_map[model_option]
            # Ultra model supports only one image
            if generation_model_id.endswith("ultra-generate-preview-06-06"):
                num_images = 1
            try:
                # Use API key for image generation (no project/location needed)
                VERTEX_API_KEY = "AIzaSyBIUiyk-Ksb0j3UxOzJqrTxqnNffzG_Y2M"

                client = genai_new.Client(api_key=VERTEX_API_KEY)
                image_response = client.models.generate_images(
                    model=generation_model_id,
                    prompt=prompt_gen,
                    config=types.GenerateImagesConfig(
                        aspect_ratio=aspect_ratio,
                        number_of_images=num_images,
                        safety_filter_level=safety_filter,
                        person_generation=person_generation
                    ),
                )
                st.subheader("Generated Images")
                for idx, img_obj in enumerate(image_response.generated_images, start=1):
                    # Handle the new image format from google.genai
                    if hasattr(img_obj, 'image') and hasattr(img_obj.image, '_pil_image'):
                        pil_img = img_obj.image._pil_image
                        st.image(pil_img, caption=f"Image {idx}", use_container_width=True)

                        # Add download button for the image
                        img_buffer = BytesIO()
                        pil_img.save(img_buffer, format='PNG')
                        img_buffer.seek(0)
                        st.download_button(
                            label=f"Download Image {idx}",
                            data=img_buffer.getvalue(),
                            file_name=f"generated_image_{idx}.png",
                            mime="image/png",
                            key=f"download_btn_{idx}_2"
                        )
                    elif hasattr(img_obj, 'image'):
                        st.image(img_obj.image, caption=f"Image {idx}", use_container_width=True)

                        # Add download button for the image (fallback)
                        try:
                            img_buffer = BytesIO()
                            img_obj.image.save(img_buffer, format='PNG')
                            img_buffer.seek(0)
                            st.download_button(
                                label=f"Download Image {idx}",
                                data=img_buffer.getvalue(),
                                file_name=f"generated_image_{idx}.png",
                                mime="image/png",
                                key=f"download_btn_fallback_{idx}_2"
                            )
                        except Exception as download_error:
                            st.warning(f"Could not create download button for Image {idx}: {download_error}")
                    else:
                        st.error(f"Could not display image {idx}")
                render_voice_output("Here are your generated images.", interaction_mode)
                # Save to CSV log
                save_query_response_to_csv(
                    user_query=prompt_gen,
                    bot_response=f"Generated {len(image_response.generated_images)} images.",
                    query_type="Image Generation",
                    model_used=generation_model_id,
                    additional_metadata={
                        'interaction_mode': interaction_mode,
                        'num_images': len(image_response.generated_images),
                        'aspect_ratio': aspect_ratio,
                        'model_option': model_option,
                        'safety_filter': safety_filter,
                        'person_generation': person_generation
                    }
                )
            except Exception as e:
                st.error(f"Error generating images: {e}")

    elif typepdf == "Images":
        upload_status = st.empty()
        progress_bar = st.empty()
        uploaded_images = st.file_uploader("Upload your image files.", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        if uploaded_images:
            upload_status.text("Uploading images...")
            progress = progress_bar.progress(0)
            for i in range(10):
                time.sleep(0.05)
                progress.progress((i + 1) * 10)
            upload_status.success(f"Successfully uploaded {len(uploaded_images)} images!")
            col_count = min(3, len(uploaded_images))
            cols = st.columns(col_count)
            for i, image_file in enumerate(uploaded_images):
                cols[i % col_count].image(image_file, caption=image_file.name, use_container_width=True)
            analysis_type = st.selectbox(
                "Select analysis type:",
                ["Custom Prompt", "Describe and Analyze", "Compare Images"],
                key="image_analysis_type"
            )
            if analysis_type == "Custom Prompt":
                prompt2 = get_voice_or_text_input(interaction_mode, "Enter your prompt for all uploaded images:", "image_custom_prompt")
                if prompt2 and st.button("Analyze with Custom Prompt", key="image_custom_btn"):
                    image_parts = []
                    for image_file in uploaded_images:
                        image_bytes = image_file.read()
                        image_part = {
                            "mime_type": image_file.type,
                            "data": image_bytes
                        }
                        image_parts.append(image_part)
                    generation_config = GenerationConfig(
                        temperature=temperature,
                        top_p=top_p,
                        max_output_tokens=max_tokens,
                    )
                    try:
                        model_instance = genai.GenerativeModel(
                            model_name=model,
                            generation_config=generation_config
                        )  # type: ignore[attr-defined]
                        with st.spinner("Processing your prompt... This may take a few moments."):
                            content_parts = image_parts + [prompt2]
                            response = model_instance.generate_content(content_parts)  # type: ignore[arg-type]
                            st.markdown(response.text)
                            render_voice_output(response.text, interaction_mode)
                            # Save to CSV log
                            save_query_response_to_csv(
                                user_query=prompt2,
                                bot_response=response.text,
                                query_type="Image Analysis",
                                model_used=model,
                                temperature=temperature,
                                additional_metadata={
                                    'interaction_mode': interaction_mode,
                                    'num_images': len(uploaded_images),
                                    'image_names': [img.name for img in uploaded_images]
                                }
                            )
                    except Exception as e:
                        st.error(f"Error generating content: {str(e)}")
            elif analysis_type == "Describe and Analyze":
                if st.button("Describe and Analyze Images", key="image_describe_btn"):
                    with st.spinner("Analyzing images... This may take a few moments."):
                        for i, image_file in enumerate(uploaded_images):
                            st.write(f"### Analysis for {image_file.name}")
                            image_bytes = image_file.read()
                            image_part = {
                                "mime_type": image_file.type,
                                "data": image_bytes
                            }
                            analysis_prompt = """
                            Describe and analyze this image in detail. Please include:
                            1. A general description of what's in the image
                            2. Key objects, people, or elements present
                            3. Notable visual characteristics (colors, composition, lighting)
                            4. Any text visible in the image
                            5. Context or setting of the image
                            6. Mood or atmosphere conveyed
                            Format your response with clear sections.
                            """
                            try:
                                generation_config = GenerationConfig(
                                    temperature=temperature,
                                    top_p=top_p,
                                    max_output_tokens=max_tokens,
                                )
                                model_instance = genai.GenerativeModel(
                                    model_name=model,
                                    generation_config=generation_config
                                )  # type: ignore[attr-defined]
                                response = model_instance.generate_content([image_part, analysis_prompt])
                                st.markdown(response.text)
                                render_voice_output(response.text, interaction_mode)
                                # Save to CSV log
                                save_query_response_to_csv(
                                    user_query=analysis_prompt,
                                    bot_response=response.text,
                                    query_type="Image Describe and Analyze",
                                    model_used=model,
                                    temperature=temperature,
                                    additional_metadata={
                                        'interaction_mode': interaction_mode,
                                        'image_name': image_file.name
                                    }
                                )
                                if i < len(uploaded_images) - 1:
                                    st.markdown("---")
                            except Exception as e:
                                st.error(f"Error analyzing {image_file.name}: {str(e)}")
            elif analysis_type == "Compare Images" and len(uploaded_images) > 1:
                if st.button("Compare Images", key="image_compare_btn"):
                    with st.spinner("Comparing images... This may take a few moments."):
                        image_parts = []
                        for image_file in uploaded_images:
                            image_bytes = image_file.read()
                            image_part = {
                                "mime_type": image_file.type,
                                "data": image_bytes
                            }
                            image_parts.append(image_part)
                        comparison_prompt = f"""
                        Compare these {len(uploaded_images)} images in detail. Please include:
                        1. Similarities between the images
                        2. Key differences between the images
                        3. Unique elements in each image
                        4. Which image might be more effective for different purposes
                        5. Overall comparative analysis
                        Format your response with clear sections for easy reading.
                        """
                        try:
                            generation_config = GenerationConfig(
                                temperature=temperature,
                                top_p=top_p,
                                max_output_tokens=max_tokens,
                            )
                            model_instance = genai.GenerativeModel(
                                model_name=model,
                                generation_config=generation_config
                            )  # type: ignore[attr-defined]
                            content_parts = image_parts + [comparison_prompt]
                            response = model_instance.generate_content(content_parts)
                            st.subheader("Image Comparison Results:")
                            st.markdown(response.text)
                            render_voice_output(response.text, interaction_mode)
                            # Save to CSV log
                            save_query_response_to_csv(
                                user_query=comparison_prompt,
                                bot_response=response.text,
                                query_type="Image Compare",
                                model_used=model,
                                temperature=temperature,
                                additional_metadata={
                                    'interaction_mode': interaction_mode,
                                    'num_images': len(uploaded_images),
                                    'image_names': [img.name for img in uploaded_images]
                                }
                            )
                        except Exception as e:
                            st.error(f"Error comparing images: {str(e)}")
            elif analysis_type == "Compare Images" and len(uploaded_images) <= 1:
                st.warning("Please upload at least 2 images for comparison.")

    elif typepdf == "Video, mp4 file":
        uploaded_videos = st.file_uploader("Upload your video files.", type="mp4", accept_multiple_files=True)
        if uploaded_videos:
            video_parts = []
            for video_file in uploaded_videos:
                # Read the video file
                video_bytes = video_file.read()

                # Create video part for Gemini
                video_part = {
                    "mime_type": video_file.type,
                    "data": video_bytes
                }
                video_parts.append(video_part)

            prompt3 = get_voice_or_text_input(interaction_mode, "Enter your prompt for all uploaded videos:", "video_prompt")
            if prompt3:
                try:
                    generation_config = GenerationConfig(
                        temperature=temperature,
                        top_p=top_p,
                        max_output_tokens=max_tokens,
                    )
                    model = genai.GenerativeModel(
                        model_name=model,
                        generation_config=generation_config
                    )  # type: ignore[attr-defined]

                    # Create content parts list with videos and prompt
                    content_parts = video_parts + [prompt3]

                    # Generate content
                    response = model.generate_content(content_parts)

                    # Display the response
                    st.markdown(response.text)
                    render_voice_output(response.text, interaction_mode)

                    # Get confidence scores if available
                    try:
                        # confidence_scores = get_confidence_scores(response)
                        # st.write(f"Confidence score: {confidence_scores.get('average_score')}")

                        # Save to CSV log
                        save_query_response_to_csv(
                            user_query=prompt3,
                            bot_response=response.text,
                            query_type="Video Analysis",
                            model_used=model,
                            temperature=temperature,
                            additional_metadata={
                                'interaction_mode': interaction_mode,
                                'num_files': len(uploaded_videos),
                                'file_names': [f.name for f in uploaded_videos]
                            }
                        )
                    except Exception as score_err:
                        logger.warning(f"Could not get confidence scores: {score_err}")

                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error generating content: {error_msg}")
                    st.error(f"Error: {error_msg}")

    elif typepdf == "Audio files":
        st.header("🎵 Audio Analysis", divider="blue")
        interaction_mode = render_voice_interaction_mode("audio_analysis")
        uploaded_audios = st.file_uploader("Upload your audio files.", type="wav", accept_multiple_files=True)
        if uploaded_audios:
            audio_parts = []
            for audio_file in uploaded_audios:
                # Read the audio file
                audio_bytes = audio_file.read()

                # Create audio part for Gemini
                audio_part = {
                    "mime_type": audio_file.type,
                    "data": audio_bytes
                }
                audio_parts.append(audio_part)

            prompt4 = get_voice_or_text_input(interaction_mode, "Enter your prompt for all uploaded audio files:", "audio_prompt")
            if prompt4:
                try:
                    generation_config = GenerationConfig(
                        temperature=temperature,
                        top_p=top_p,
                        max_output_tokens=max_tokens,
                    )
                    model = genai.GenerativeModel(
                        model_name=model,
                        generation_config=generation_config
                    )  # type: ignore[attr-defined]

                    # Create content parts list with audios and prompt
                    content_parts = audio_parts + [prompt4]

                    # Generate content
                    response = model.generate_content(content_parts)

                    # Display the response
                    st.markdown(response.text)
                    render_voice_output(response.text, interaction_mode)

                    # Get confidence scores if available
                    try:
                        # confidence_scores = get_confidence_scores(response)
                        # st.write(f"Confidence score: {confidence_scores.get('average_score')}")

                        # Save to CSV log
                        save_query_response_to_csv(
                            user_query=prompt4,
                            bot_response=response.text,
                            query_type="Audio Analysis",
                            model_used=model,
                            temperature=temperature,
                            additional_metadata={
                                'interaction_mode': interaction_mode,
                                'num_files': len(uploaded_audios),
                                'file_names': [f.name for f in uploaded_audios]
                            }
                        )
                    except Exception as score_err:
                        logger.warning(f"Could not get confidence scores: {score_err}")

                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error generating content: {error_msg}")
                    st.error(f"Error: {error_msg}")

    elif typepdf == "DAT files":
        uploaded_files = st.file_uploader("Choose maximum of 50 DAT files", type='dat', accept_multiple_files=True)
        if uploaded_files and len(uploaded_files) > 50:
            st.error("You can upload a maximum of 50 DAT files")
            return
        db_name_list = []
        for file in uploaded_files:
            original_name = file.name
            if len(original_name) > 20:
                name, ext = os.path.splitext(original_name)
                truncated_name = name[:20 - len(ext)] + ext
                db_name_list.append(os.path.splitext(truncated_name)[0])
            else:
                db_name_list.append(os.path.splitext(original_name)[0])
        combined_db_name = "_".join(db_name_list) if db_name_list else "dat_db"
        if len(combined_db_name) > 50:
            combined_db_name = combined_db_name[:50]
        if uploaded_files:
            st.write(f"Processing {len(uploaded_files)} files...")
            embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=SecretStr(GOOGLE_API_KEY), model="models/embedding-001")
            db = embed_and_store_files(uploaded_files, embedding_model, combined_db_name)
            question = get_voice_or_text_input(interaction_mode, "Enter your question about the DAT files:", "dat_question")
            if question and db:
                try:
                    # Show loading spinner while processing the question
                    with st.spinner("Processing your question... This may take a few moments."):
                        # Query Vector DB (modify as needed)
                        response = rag_pipeline(question, model, combined_db_name, temperature, top_p, max_tokens)
                        if response:
                            json_content = response.content
                            if isinstance(json_content, str):
                                json_content = json_content.strip("```json\n").strip("```")
                                try:
                                    parsed_data = json.loads(json_content)
                                except Exception:
                                    st.warning("Could not parse model response as JSON.")
                                    parsed_data = {}
                            else:
                                st.warning("Model response was not a string and could not be parsed as JSON.")
                                parsed_data = {}
                            st.subheader("Response:")
                            answer_text = parsed_data.get("Answer", "No answer found.")
                            st.write(answer_text)
                            render_voice_output(answer_text, interaction_mode)
                            # st.write(f"Similarity Score: {parsed_data.get('Overall Similarity', 'N/A')}")
                            st.write(f"Used Chunks: {parsed_data.get('Used Chunks', 'N/A')}")
                            st.write(f"Reasoning: {parsed_data.get('Reasoning', 'N/A')}")

                        # Save to CSV log
                        save_query_response_to_csv(
                            user_query=question,
                            bot_response=answer_text,
                            query_type="DAT RAG Query",
                            model_used=model,
                            temperature=temperature,
                            additional_metadata={
                                'interaction_mode': interaction_mode,
                                'num_files': len(uploaded_files),
                                'file_names': [f.name for f in uploaded_files],
                                'similarity_score': parsed_data.get('Overall Similarity', 'N/A'),
                                'used_chunks': parsed_data.get('Used Chunks', 'N/A'),
                                'reasoning': parsed_data.get('Reasoning', 'N/A')
                            }
                        )
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    elif typepdf == "Markdown files":
        st.header("📄 Markdown Files", divider="blue")
        interaction_mode = render_voice_interaction_mode("md_files")
        uploaded_files = st.file_uploader("Choose maximum of 50 Markdown files", type='md', accept_multiple_files=True)
        if uploaded_files and len(uploaded_files) > 50:
            st.error("You can upload a maximum of 50 Markdown files")
            return
        db_name_list = []
        for file in uploaded_files:
            original_name = file.name
            if len(original_name) > 20:
                name, ext = os.path.splitext(original_name)
                truncated_name = name[:20 - len(ext)] + ext
                db_name_list.append(os.path.splitext(truncated_name)[0])
            else:
                db_name_list.append(os.path.splitext(original_name)[0])
        combined_db_name = "_".join(db_name_list) if db_name_list else "md_db"
        if len(combined_db_name) > 50:
            combined_db_name = combined_db_name[:50]
        if uploaded_files:
            st.write(f"Processing {len(uploaded_files)} files...")
            embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=SecretStr(GOOGLE_API_KEY), model="models/embedding-001")
            db = embed_and_store_files(uploaded_files, embedding_model, combined_db_name)
            question = get_voice_or_text_input(interaction_mode, "Enter your question about the Markdown files:", "md_question")
            if question and db:
                try:
                    # Show loading spinner while processing the question
                    with st.spinner("Processing your question... This may take a few moments."):
                        # Query Vector DB (modify as needed)
                        response = rag_pipeline(question, model, combined_db_name, temperature, top_p, max_tokens)
                        if response:
                            json_content = response.content
                            if isinstance(json_content, str):
                                json_content = json_content.strip("```json\n").strip("```")
                                try:
                                    parsed_data = json.loads(json_content)
                                except Exception:
                                    st.warning("Could not parse model response as JSON.")
                                    parsed_data = {}
                            else:
                                st.warning("Model response was not a string and could not be parsed as JSON.")
                                parsed_data = {}
                            st.subheader("Response:")
                            answer_text = parsed_data.get("Answer", "No answer found.")
                            st.write(answer_text)
                            render_voice_output(answer_text, interaction_mode)
                            # st.write(f"Similarity Score: {parsed_data.get('Overall Similarity', 'N/A')}")
                            st.write(f"Used Chunks: {parsed_data.get('Used Chunks', 'N/A')}")
                            st.write(f"Reasoning: {parsed_data.get('Reasoning', 'N/A')}")

                        # Save to CSV log
                        save_query_response_to_csv(
                            user_query=question,
                            bot_response=answer_text,
                            query_type="Markdown RAG Query",
                            model_used=model,
                            temperature=temperature,
                            additional_metadata={
                                'interaction_mode': interaction_mode,
                                'num_files': len(uploaded_files),
                                'file_names': [f.name for f in uploaded_files],
                                'similarity_score': parsed_data.get('Overall Similarity', 'N/A'),
                                'used_chunks': parsed_data.get('Used Chunks', 'N/A'),
                                'reasoning': parsed_data.get('Reasoning', 'N/A')
                            }
                        )
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    elif typepdf == "Multiple Media":
        st.write("Upload multiple types of media files (PDF, Images, Videos, Audio, etc.)")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "png", "jpg", "jpeg", "mp4", "wav", "txt", "docx", "pptx", "csv", "xml", "json", "yaml", "yml", "h", "xlsx", "xls", "dat", "md"],
            accept_multiple_files=True
        )
        if uploaded_files:
            progress_bar = st.progress(0)
            status_text = st.empty()
            all_content_parts, file_info = process_multiple_media_files(uploaded_files, model, temperature, top_p, max_tokens, progress_bar, status_text)

            if uploaded_files:
                st.success("Files processed successfully! You can now ask questions about the content.")
                prompt = get_voice_or_text_input(interaction_mode, "Enter your question about the uploaded files:", "multi_media_question")

                if prompt:
                    # Show loading spinner while processing
                    with st.spinner("Processing your question... This may take a few moments."):
                        try:
                            # Create the model with generation config
                            generation_config = GenerationConfig(
                                temperature=temperature,
                                top_p=top_p,
                                max_output_tokens=max_tokens,
                            )

                            model = genai.GenerativeModel(
                                model_name=model,
                                generation_config=generation_config
                            )  # type: ignore[attr-defined]

                            # Create the final prompt with context about the files
                            context_prompt = f"""I have uploaded the following files:
{', '.join(f'{name} ({type})' for name, type in file_info)}

Please analyze all these files together and answer the following question: {prompt}

Provide a comprehensive answer that takes into account information from all the uploaded files."""

                            # Process content in smaller chunks to prevent timeout
                            try:
                                # If there are too many content parts, process them in batches
                                if len(all_content_parts) > 10:
                                    st.write("Processing a large number of files. This may take some time...")

                                    # Split content parts into batches
                                    batch_size = 10
                                    all_responses = []

                                    for i in range(0, len(all_content_parts), batch_size):
                                        batch = all_content_parts[i:i+batch_size]
                                        st.write(f"Processing batch {i//batch_size + 1} of {(len(all_content_parts) + batch_size - 1)//batch_size}...")

                                        # Add a small delay between batches
                                        if i > 0:
                                            time.sleep(1)

                                        # Generate content for this batch
                                        batch_response = model.generate_content(batch)
                                        all_responses.append(batch_response.text)

                                    # Combine all responses
                                    combined_response = "\n\n".join(all_responses)

                                    # Display the combined response
                                    st.subheader("Response:")
                                    st.markdown(combined_response)
                                    render_voice_output(combined_response, interaction_mode)

                                    # Use the last response for confidence scores
                                    response = batch_response
                                else:
                                    # Generate content for all parts at once
                                    response = model.generate_content(all_content_parts + [context_prompt])  # type: ignore[arg-type]

                                    # Display the response
                                    st.subheader("Response:")
                                    st.markdown(response.text)
                                    render_voice_output(response.text, interaction_mode)
                            except Exception as e:
                                error_msg = str(e)
                                logger.error(f"Error generating content: {error_msg}")
                                st.error(f"Error: {error_msg}")

                                # Try a fallback approach with just the context prompt
                                try:
                                    st.write("Trying a simplified approach...")
                                    fallback_response = model.generate_content(context_prompt)
                                    st.subheader("Response (Simplified):")
                                    st.markdown(fallback_response.text)
                                    response = fallback_response
                                except Exception as fallback_error:
                                    st.error(f"Fallback approach also failed: {str(fallback_error)}")
                                    return
                        except Exception as e:
                            error_msg = str(e)
                            logger.error(f"Error in multiple media processing: {error_msg}")
                            st.error(f"Error: {error_msg}")
            else:
                st.error("No files were successfully processed. Please check your file formats and try again.")

    elif typepdf == "Voice to Voice":
        st.header("🎙️ Voice to Voice", divider="blue")
        st.write("Speak your question and get an audio response back!")

        # Initialize chat state for Voice to Voice (duplicate branch)
        if 'v2v_history' not in st.session_state:
            st.session_state['v2v_history'] = []
        if 'v2v_input' not in st.session_state:
            st.session_state['v2v_input'] = ""

        # Voice input section
        with st.container(border=True):
            st.subheader("🎤 Voice Input", anchor=False)
            st.caption("Tap the microphone, speak your question, then tap again to stop. The audio will be transcribed and processed.")

            audio_bytes = audio_recorder(
                text="🎤 Tap to Record / Stop",
                icon_name="microphone",
                icon_size="2x",
                neutral_color="#6C757D",
                recording_color="#FF4B4B",
                key="voice_to_voice_recorder",
            )

        voice_text = ""
        transcription_error = False

        if audio_bytes:
            # Playback preview
            st.audio(audio_bytes, format='audio/wav', start_time=0)

            # Transcribe audio to text
            if sr is not None and AudioSegment is not None:
                with st.spinner("🎧 Transcribing your voice..."):
                    try:
                        audio_file = io.BytesIO(audio_bytes)
                        audio_file.seek(0)
                        audio_segment = AudioSegment.from_file(audio_file, format="wav")
                        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                        audio_segment.export(temp_wav.name, format="wav")
                        temp_wav.close()

                        recognizer = sr.Recognizer()
                        with sr.AudioFile(temp_wav.name) as source:
                            audio_data = recognizer.record(source)
                            voice_text = recognizer.recognize_google(audio_data)

                        os.remove(temp_wav.name)
                        st.success(f"✅ Transcription: {voice_text}")

                    except Exception as e:
                        transcription_error = True
                        st.warning(f"⚠️ Automatic transcription failed: {e}. Please type what you said.")
            else:
                transcription_error = True
                st.info("ℹ️ SpeechRecognition or pydub not installed. Please type what you said.")

            # Manual transcription fallback
            if transcription_error or (audio_bytes and not voice_text):
                manual_text = st.text_area("Manual transcription (type what you said):", "", height=100)
                if manual_text:
                    voice_text = manual_text

            # Prefill the text input from transcription without auto-sending
            if voice_text:
                st.session_state['v2v_input'] = voice_text
                st.info("Transcription added to the text box below. You can edit it and click Generate Response.")

        # Display chat history
        if st.session_state['v2v_history']:
            st.subheader("💬 Chat History")
            for i, turn in enumerate(st.session_state['v2v_history']):
                with st.container(border=True):
                    col1, col2 = st.columns([1, 20])
                    with col1:
                        st.markdown("👤")
                    with col2:
                        st.markdown(f"**You:** {turn.get('user', '')}")

                    col1, col2 = st.columns([1, 20])
                    with col1:
                        st.markdown("🤖")
                    with col2:
                        st.markdown(f"**Assistant:** {turn.get('assistant', '')}")
            st.markdown("---")

        # Always-visible text input and action buttons
        user_input = st.text_area(
            "Type your message (or use the mic above):",
            value=st.session_state.get('v2v_input', ''),
            key="v2v_text_area_dup",
            placeholder="Type here, or use the mic, then click Generate Response",
            height=100
        )
        st.session_state['v2v_input'] = user_input

        col_gen2, col_clear2 = st.columns([2, 1])
        with col_gen2:
            gen_clicked2 = st.button("Generate Response", key="v2v_generate_btn_dup", use_container_width=True)
        with col_clear2:
            clear_clicked2 = st.button("Clear Chat History", key="v2v_clear_btn_dup", use_container_width=True)

        if clear_clicked2:
            st.session_state['v2v_history'] = []
            st.session_state['v2v_input'] = ""
            st.rerun()

        if gen_clicked2:
            if not st.session_state['v2v_input'].strip():
                st.warning("Please enter a message or use the microphone.")
            else:
                question_text = st.session_state['v2v_input'].strip()
                with st.spinner("🤖 Generating response..."):
                    try:
                        generation_config = GenerationConfig(
                            temperature=temperature,
                            top_p=top_p,
                            max_output_tokens=max_tokens,
                        )
                        model_instance = genai.GenerativeModel(
                            model_name=model,
                            generation_config=generation_config
                        )
                        prompt = f"You are having a friendly conversation with someone. They just said: '{question_text}'. Respond naturally and conversationally as if you're talking to a friend. Keep your response brief and natural."
                        response = model_instance.generate_content(prompt)
                        response_text = response.text

                        # Show text response
                        st.subheader("📝 Text Response:")
                        st.success(response_text)

                        # TTS output
                        st.subheader("🔊 Audio Response:")
                        if gTTS is not None:
                            try:
                                with st.spinner("🎵 Converting response to speech..."):
                                    clean_response = re.sub(r'^(agent|expert):\s*', '', response_text, flags=re.I)
                                    tts = gTTS(text=clean_response, lang='en')
                                    tts_fp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                                    tts.save(tts_fp.name)
                                    tts_fp.close()
                                    with open(tts_fp.name, 'rb') as audio_file:
                                        audio_bytes_out = audio_file.read()
                                    st.audio(audio_bytes_out, format='audio/mp3')
                                    os.remove(tts_fp.name)
                            except Exception as tts_error:
                                st.error(f"❌ Text-to-speech generation failed: {tts_error}")
                        else:
                            st.warning("⚠️ gTTS not available. Install gTTS for voice output: `pip install gtts`")

                        # Append to chat history
                        st.session_state['v2v_history'].append({
                            'user': question_text,
                            'assistant': response_text
                        })

                        # Save to CSV log
                        save_query_response_to_csv(
                            user_query=question_text,
                            bot_response=response_text,
                            query_type="Voice to Voice",
                            model_used=model,
                            temperature=temperature,
                            additional_metadata={
                                'interaction_mode': 'Voice',
                                'transcription_method': 'automatic' if not transcription_error else 'manual',
                                'tts_enabled': gTTS is not None
                            }
                        )
                    except Exception as e:
                        st.error(f"❌ Error processing your question: {str(e)}")

        # Instructions for users
        with st.expander("ℹ️ How to use Voice to Voice"):
            st.markdown("""
            **Steps:**
            1. Click the microphone button to start recording
            2. Speak your question clearly
            3. Click the microphone button again to stop recording
            4. Wait for automatic transcription (or type manually if needed)
            5. Get both text and audio responses

            **Tips:**
            - Speak clearly and at a moderate pace
            - Ensure you have a good microphone
            - The system works best in quiet environments
            - You can type manually if transcription fails
            """)

    elif typepdf == "Multiple Media":
        st.write("Upload multiple types of media files (PDF, Images, Videos, Audio, etc.)")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "png", "jpg", "jpeg", "mp4", "wav", "txt", "docx", "pptx", "csv", "xml", "json", "yaml", "yml", "h", "xlsx", "xls", "dat", "md"],
            accept_multiple_files=True
        )
        if uploaded_files:
            progress_bar = st.progress(0)
            status_text = st.empty()
            all_content_parts, file_info = process_multiple_media_files(uploaded_files, model, temperature, top_p, max_tokens, progress_bar, status_text)

            if uploaded_files:
                st.success("Files processed successfully! You can now ask questions about the content.")
                prompt = get_voice_or_text_input(interaction_mode, "Enter your question about the uploaded files:", "multi_media_question")

                if prompt:
                    # Show loading spinner while processing
                    with st.spinner("Processing your question... This may take a few moments."):
                        try:
                            # Create the model with generation config
                            generation_config = GenerationConfig(
                                temperature=temperature,
                                top_p=top_p,
                                max_output_tokens=max_tokens,
                            )

                            model = genai.GenerativeModel(
                                model_name=model,
                                generation_config=generation_config
                            )  # type: ignore[attr-defined]

                            # Create the final prompt with context about the files
                            context_prompt = f"""I have uploaded the following files:
{', '.join(f'{name} ({type})' for name, type in file_info)}

Please analyze all these files together and answer the following question: {prompt}

Provide a comprehensive answer that takes into account information from all the uploaded files."""

                            # Process content in smaller chunks to prevent timeout
                            try:
                                # If there are too many content parts, process them in batches
                                if len(all_content_parts) > 10:
                                    st.write("Processing a large number of files. This may take some time...")

                                    # Split content parts into batches
                                    batch_size = 10
                                    all_responses = []

                                    for i in range(0, len(all_content_parts), batch_size):
                                        batch = all_content_parts[i:i+batch_size]
                                        st.write(f"Processing batch {i//batch_size + 1} of {(len(all_content_parts) + batch_size - 1)//batch_size}...")

                                        # Add a small delay between batches
                                        if i > 0:
                                            time.sleep(1)

                                        # Generate content for this batch
                                        batch_response = model.generate_content(batch)
                                        all_responses.append(batch_response.text)

                                    # Combine all responses
                                    combined_response = "\n\n".join(all_responses)

                                    # Display the combined response
                                    st.subheader("Response:")
                                    st.markdown(combined_response)
                                    render_voice_output(combined_response, interaction_mode)

                                    # Use the last response for confidence scores
                                    response = batch_response
                                else:
                                    # Generate content for all parts at once
                                    response = model.generate_content(all_content_parts + [context_prompt])  # type: ignore[arg-type]

                                    # Display the response
                                    st.subheader("Response:")
                                    st.markdown(response.text)
                                    render_voice_output(response.text, interaction_mode)
                            except Exception as e:
                                error_msg = str(e)
                                logger.error(f"Error generating content: {error_msg}")
                                st.error(f"Error: {error_msg}")

                                # Try a fallback approach with just the context prompt
                                try:
                                    st.write("Trying a simplified approach...")
                                    fallback_response = model.generate_content(context_prompt)
                                    st.subheader("Response (Simplified):")
                                    st.markdown(fallback_response.text)
                                    response = fallback_response
                                except Exception as fallback_error:
                                    st.error(f"Fallback approach also failed: {str(fallback_error)}")
                                    return
                        except Exception as e:
                            error_msg = str(e)
                            logger.error(f"Error in multiple media processing: {error_msg}")
                            st.error(f"Error: {error_msg}")
            else:
                st.error("No files were successfully processed. Please check your file formats and try again.")

if __name__ == '__main__':

    main()
