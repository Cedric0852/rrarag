import os
import asyncio
import asyncio
import qrcode
from PIL import Image
import io
from dotenv import load_dotenv
from io import BytesIO
import streamlit as st
import logging
import os
import requests
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from detectandtranslate import detect_language,translate_text
from io import BytesIO
import tempfile
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
import uuid
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
logging.basicConfig(filename='app.log', level=logging.ERROR, format='%(asctime)s %(levelname)s %(message)s')
load_dotenv()
# Load environment variables from Streamlit secrets
api_key = st.secrets["API_KEY1"]
google_api_key = st.secrets["GOOGLE_API_KEY"]
groq_api_key = st.secrets["GROQ_API_KEY"]
eleven_api_key = st.secrets["ELEVEN_API_KEY"]
google_application_credentials = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]

# Set the path for the Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_application_credentials
# Initialize the database client
# Caching the database client initialization
@st.cache_resource(ttl=300)  # Cache for 5 minutes
def initialize_database_client(api_key):
    client = QdrantClient(
        url="https://dd35784a-143b-4fa3-8c18-b34b4bb9ef8e.us-east4-0.gcp.cloud.qdrant.io:6333",
        port=6333,
        verify=False,
        api_key=api_key,
    )
    return client

# Caching the LLM initialization
@st.cache_resource(ttl=300)  # Cache for 5 minutes
def initialize_llm(groq_api_key):
    try:
        llm = ChatGroq(temperature=0, model_name="llama3-8b-8192", api_key=groq_api_key)
        llm_model_name = "llama3-8b-8192"
    except Exception as e:
        logging.warning(f"Failed to initialize ChatGroq: {e}. Falling back to mixtral-8x7b-32768.")
        llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", api_key=groq_api_key)
        llm_model_name = "mixtral-8x7b-32768"
    return llm, llm_model_name

# Load environment variables
api_key = st.secrets["API_KEY1"]
groq_api_key = st.secrets["GROQ_API_KEY"]

# Initialize the database client and LLM with caching
client = initialize_database_client(api_key)
llm, llm_model_name = initialize_llm(groq_api_key)

# Update session state with LLM model name
st.session_state['llm_initialized'] = llm_model_name

print("Database loaded")
print("LLM Initialized...")

# Define the prompt template with memory and chat history
prompt_template = PromptTemplate(template="""Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer, Your name is RRA Assistant created by Cedric as a Kinyarwanda RAG hackathon prepared by IndabaX and DigitalUmuganda 2024.
Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
""", input_variables=['context', 'question'])

hf = FastEmbedEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5-Q")

db = Qdrant(
    client=client,
    embeddings=hf,
    collection_name="RR3"
)

#prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
load_vector_store = db
retriever = load_vector_store.as_retriever(search_kwargs={"k":3})

#chain_type_kwargs = {"prompt": prompt}

async def extract_result_text(response):
    result_text = response.get('result', '')
    newline_position = result_text.find('\n\n\n\n')
    if (newline_position != -1):
        result_text = result_text[:newline_position]
    return result_text

async def get_response(message,history):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True,
        verbose=True
    )
    project_id = "banguka-406711"
    # Detect the language of the input message
    detected_language = detect_language(message, project_id)
    # Translate the message to English
    if detected_language != "en":
        # Translate the message to English
        message_in_english = await translate_text(message, detected_language, "en", project_id)
    else:
        message_in_english = message
    response = await asyncio.to_thread(qa, message_in_english)
    result_text = await extract_result_text(response)
    
    if detected_language != "en":
        # Translate the result text from English back to the detected language
        translated = await translate_text(result_text, "en", detected_language, project_id)
    else:
        translated = result_text
    return translated
async def fetch_audio(translated_text):
    url = "https://api.elevenlabs.io/v1/text-to-speech/XB0fDUnXU5powFXDhCwa/stream"
    payload = {
        "text": translated_text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "use_speaker_boost": True
        }
    }
    headers = {
        "xi-api-key": os.getenv("ELEVEN_API_KEY"),
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error("Can't generate audio now, try again later (amajwi ntabontse nonaha)")
        return None
    # Use BytesIO to store audio in memory
    audio_buffer = BytesIO()
    
    for chunk in response.iter_content(chunk_size=8192):
        audio_buffer.write(chunk)

    # Ensure the buffer is ready for reading
    audio_buffer.seek(0)

    # Create a temporary file to store audio
    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_audio_file.write(audio_buffer.read())
    temp_audio_file.flush()
    # Return the path of the temporary audio file
    return temp_audio_file.name
# Streamlit app
# Streamlit app
# Streamlit app
# Title of the app
st.title("RRA RAG Chatbot (Ask in any language)")

# Add a sidebar with navigation options
with st.sidebar:
    selected_page = st.radio("Navigation", ["Chatbot", "About"])

if selected_page == "Chatbot":
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask Tax Related Questions?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response_text = asyncio.run(get_response(prompt, history=st.session_state.messages))
        with st.chat_message("assistant"):
            st.markdown(response_text)
            audio_file_path = asyncio.run(fetch_audio(response_text))
            if audio_file_path:
                audio_file = open(audio_file_path, "rb")
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format="audio/mp3")
                audio_file.close()
        st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.markdown("""Telegram :[Telegram Bot](https://t.me/rra_chat_bot)""")
elif selected_page == "About":
    st.header("About this App")
    st.markdown("""
    **RRA RAG Chatbot** is a project created to provide quick and accurate tax-related information.
                
    **Languages**: you can ask in any language.
                
    **Credits:**
    - Created by: [Cedric](mailto:mugishac777@gmail.com)
    - Special thanks to IndabaX and DigitalUmuganda 2024
    
    To access RRA RAG Chatbot on Telegram scan the QR code below.
    """)
    
    # Generate a QR code
    telegram_bot_url = "https://t.me/rra_chat_bot"
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(telegram_bot_url)
    qr.make(fit=True)

    img = qr.make_image(fill="black", back_color="white")

    # Display the QR code
    buf = io.BytesIO()
    img.save(buf)
    st.image(buf.getvalue(), caption="Scan to access the Telegram bot")