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
import re
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
logging.basicConfig(filename='app.log', level=logging.ERROR, format='%(asctime)s %(levelname)s %(message)s')
load_dotenv()
import cohere
# Load environment variables from Streamlit secrets
api_key = st.secrets["API_KEY1"]
groq_api_key = st.secrets["GROQ_API_KEY"]

qdurl=st.secrets["QDURL"]
cohere_api_key = st.secrets["COHERE_API_KEY"]

@st.cache_resource(ttl=300)  
def initialize_database_client(api_key):
    client = QdrantClient(
        url=qdurl,
        port=6333,
        verify=False,
        api_key=api_key,
    )
    return client
@st.cache_resource(ttl=300)
def initialize_reranker():
    co = cohere.Client(api_key=cohere_api_key)
    return CohereRerank(client=co, top_n=3)
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

# Initialize the database client and LLM with caching
client = initialize_database_client(api_key)
llm, llm_model_name = initialize_llm(groq_api_key)

# Update session state with LLM model name
st.session_state['llm_initialized'] = llm_model_name

print("Database loaded")
print("LLM Initialized...")

# Define the prompt template with memory and chat history
prompt_template = PromptTemplate(template="""You are RRA Assistant, created by Cedric as a Kinyarwanda RAG hackathon prepared by IndabaX and DigitalUmuganda 2024. Your task is to answer tax-related questions using the provided context.

Context: {context}

User's Question: {question}

Please follow these steps to answer the question:

Step 1: Analyze the question
Briefly explain your understanding of the question and any key points to address.

Step 2: Provide relevant information
Using the context provided, give detailed information related to the question. Include specific facts, figures, or explanations from the context.

Step 3: Final answer
Provide a clear, concise answer to the original question. Start directly with the relevant information, avoiding phrases like "In summary" or "To conclude".

Remember:
- If you don't know the answer or can't find relevant information in the context, say so honestly.
- Do not make up information.
- Use the provided context to support your answer.
-Include "For more information, call 3004" at the end of every answer

Your response:
""", input_variables=['context', 'question'])

hf = FastEmbedEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5-Q")

db = Qdrant(
    client=client,
    embeddings=hf,
    collection_name="RR3"
)

#prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
reranker = initialize_reranker()
load_vector_store = db
base_retriever = load_vector_store.as_retriever(search_kwargs={"k": 14})
retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base_retriever)


#chain_type_kwargs = {"prompt": prompt}
async def extract_result_text(response):
    result_text = response.get('result', '')
    source_documents = response.get('source_documents', [])
    
    # Extract the answer from the result text
    final_answer_start = result_text.find("Step 3: Final answer")
    if final_answer_start != -1:
        final_answer = result_text[final_answer_start + len("Step 3: Final answer"):].strip()
    else:
        final_answer = result_text
    
    # Remove unnecessary asterisks and step labels
    final_answer = re.sub(r'\*\*', '', final_answer).strip()  
    final_answer = re.sub(r'Step \d+:', '', final_answer).strip()
    
    # Extract unique sources and pages
    sources = {}
    for doc in source_documents:
        metadata = doc.metadata
        source = metadata.get('source', 'Unknown')
        page = metadata.get('page', 'N/A')
        if source in sources:
            if page not in sources[source]:
                sources[source].append(page)
        else:
            sources[source] = [page]
    
    # Format the reference string
    references = []
    for source, pages in sources.items():
        if len(pages) == 1:
            references.append(f"{source} (Page {pages[0]})")
        else:
            pages_str = ", ".join(map(str, pages))
            references.append(f"{source} (Pages {pages_str})")
    
    reference_text = "\n\nReferences:\n" + "\n".join(references) if references else ""
    
    return final_answer + reference_text

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
        translated = await translate_text(result_text, "en", detected_language, project_id)
    else:
        translated = result_text
    return translated

st.title("RRA FAQ Chatbot (Ask in any language)")
st.markdown("""Telegram :[Telegram Bot](https://t.me/rra_chat_bot) comming soon!!!""")
hide_streamlit_style = """
<style>
.main .block-container {
       padding-bottom: 100px;
   }
.viewerBadge_link__qRIco {display: none !important;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
# Add a sidebar with navigation options
with st.sidebar:
    selected_page = st.radio("Navigation", ["RAG", "About"])

if selected_page == "RAG":
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
        st.session_state.messages.append({"role": "assistant", "content": response_text})
    
elif selected_page == "About":
    st.header("About this App")
    st.markdown("""
    **Version 1.0**  
    **RRA FAQ Chatbot** is a project created to provide quick and accurate tax-related information in Rwanda.
                
    **Languages**: you can ask in any language.
                
    **Credits:**
    - Created by: [Cedric](mailto:mugishac777@gmail.com)
    - All data used in this project are publicly available on [RRA Website faq](https://www.rra.gov.rw/en/domestic-tax-services/faqs) last accessed on 20/10/2024
     
    
    **Disclaimer:**
    The RRA FAQ Chatbot is designed to provide users with fast and convenient access to general tax-related information. However, please note that the information provided by the chatbot may not always be complete, accurate, or up to date. Tax laws and regulations can be complex and subject to change. We strongly recommend that you consult with a qualified tax advisor and the Rwanda Revenue Authority [RRA](https://www.rra.gov.rw/en/home) or call 3004 for personalized advice and official guidance before making any tax-related decisions. The chatbot should not be relied upon as a sole source for tax information or advice.
    """)

    
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
    buf = io.BytesIO()
    img.save(buf)
    st.image(buf.getvalue(), caption="Telegram bot Comming soon")