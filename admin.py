import os
import asyncio
import logging
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from langchain_community.document_loaders import PyPDFDirectoryLoader
import gradio as gr

# Set up logging
logging.basicConfig(filename='app.log', level=logging.ERROR, format='%(asctime)s %(levelname)s %(message)s')
load_dotenv()

# Environment variables
api_key = os.getenv('API_KEY1')
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
qdurl = os.getenv("QDURL")

# Initialize Qdrant Client
try:
    client = QdrantClient(
        url=qdurl,
        port=6333,
        verify=False,
        api_key=api_key,
    )
    collections = client.get_collections()
except Exception as e:
    print("An error occurred: %s", e)
    if "server engine not running" in str(e).lower():
        print("The database engine is not running. Please check the server status.")
    exit()

print("Database loaded")

# Initialize embeddings and database
hf = FastEmbedEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5-Q")

db = Qdrant(
    client=client,
    embeddings=hf,
    collection_name="RR2"
)

load_vector_store = db
retriever = load_vector_store.as_retriever(search_kwargs={"k":3})
llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")

# Collection Management Functions
async def create_collection(url, port, collection_name, vector_size):
    try:
        client = QdrantClient(
            url=url,
            port=int(port),
            api_key=api_key,
            verify=False,
        )
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(
                size=int(vector_size),
                distance=rest.Distance.COSINE,
            )
        )
        return "Collection created successfully."
    except Exception as e:
        return f"Failed to create collection: {str(e)}"

# Data Processing Function
async def data_ingest_function(data_path, url, collection_name):
    loop = asyncio.get_event_loop()
    try:
        def load_documents():
            loader = PyPDFDirectoryLoader(data_path)
            hf = FastEmbedEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5-Q")
            text_splitter = SemanticChunker(hf, breakpoint_threshold_type="interquartile")
            documents = loader.load_and_split(text_splitter=text_splitter)
            return documents

        texts = await loop.run_in_executor(None, load_documents)
        print(f"Processed {len(texts)} text chunks")

        def index_documents():
            qdrant = Qdrant.from_documents(
                texts,
                hf,
                url=url,
                api_key=api_key,
                collection_name=collection_name
            )
            return qdrant

        await loop.run_in_executor(None, index_documents)
        return "Data processing and indexing completed successfully."
    except Exception as e:
        return f"Failed to process data: {str(e)}"

# Gradio Admin Interface
with gr.Blocks(theme="soft", title="Admin LLM System", head="Admin for LARGE LANGUAGE MODEL SYSTEM") as admin:
    with gr.Tab("Collection Management"):
        with gr.Row():
            url_input = gr.Textbox(label="Qdrant URL", value="")
            port_input = gr.Number(label="Port", value=6333)
            collection_name_input = gr.Textbox(label="Collection Name", value="RR2")
            vector_size_input = gr.Number(label="Vector Size", value=768)
            create_collection_btn = gr.Button("Create Collection")
        create_collection_btn.click(
            create_collection, 
            inputs=[url_input, port_input, collection_name_input, vector_size_input], 
            outputs=gr.Textbox(label="Result")
        )
        
        with gr.Row():
            data_path_input = gr.Textbox(label="Data Folder Path")
            url_processing_input = gr.Textbox(label="Qdrant URL for Processing", value="")
            collection_name_processing_input = gr.Textbox(label="Collection Name for Processing", value="RR2")
            start_processing_btn = gr.Button("Start Processing")
        start_processing_btn.click(
            data_ingest_function, 
            inputs=[data_path_input, url_processing_input, collection_name_processing_input], 
            outputs=gr.Textbox(label="Result")
        )

# Launch Interface
if __name__ == "__main__":
    admin.launch(server_name="0.0.0.0", server_port=7860, share=False)