import os
import gradio as gr
import logging
import asyncio
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_qdrant import QdrantVectorStore
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from qdrant_client.models import PointStruct, VectorParams, Distance
import uuid
from qdrant_client.http import models
from datetime import datetime
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from qdrant_client import QdrantClient
import cohere
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
import re
from translation_service import TranslationService

# Load environment variables
load_dotenv()

# Initialize logging with INFO level and detailed format
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize services
translator = TranslationService()

def initialize_database_client():
    """Initialize Qdrant client"""
    try:
        client = QdrantClient(
            url=os.getenv("QDURL"),
            api_key=os.getenv("API_KEY1"),
            verify=True # Set to True if using SSL
        )
        logging.info("Qdrant client initialized successfully.")
        return client
    except Exception as e:
        logging.error(f"Failed to initialize Qdrant client: {e}")
        raise

def initialize_llm():
    """Initialize LLM with fallback"""
    try:
        llm = ChatGroq(
            temperature=0, 
            model_name="llama3-8b-8192", 
            api_key=os.getenv("GROQ_API_KEY")
        )
        logging.info("ChatGroq initialized with model llama3-8b-8192.")
        return llm
    except Exception as e:
        logging.warning(f"Failed to initialize ChatGroq with llama3: {e}. Falling back to mixtral.")
        try:
            llm = ChatGroq(
                temperature=0, 
                model_name="mixtral-8x7b-32768", 
                api_key=os.getenv("GROQ_API_KEY")
            )
            logging.info("ChatGroq initialized with fallback model mixtral-8x7b-32768.")
            return llm
        except Exception as fallback_e:
            logging.error(f"Failed to initialize fallback LLM: {fallback_e}")
            raise

def initialize_services():
    """Initialize all services"""
    try:
        # Initialize Qdrant client
        client = initialize_database_client()
        
        # Initialize embeddings
        embeddings = FastEmbedEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5-Q")
        logging.info("FastEmbedEmbeddings initialized successfully.")
        
        # Initialize Qdrant DB
        db = QdrantVectorStore(
            client=client,
            embedding=embeddings,
            collection_name="RR3"
        )
        logging.info("QdrantVectorStore initialized with collection 'RR3'.")
        
        # Initialize retriever with reranker
        cohere_client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
        reranker = CohereRerank(
            client=cohere_client,
            top_n=3,
            model="rerank-multilingual-v3.0"
        )
        base_retriever = db.as_retriever(search_kwargs={"k": 14})
        retriever = ContextualCompressionRetriever(
            base_compressor=reranker, 
            base_retriever=base_retriever
        )
        logging.info("Retriever with reranker initialized successfully.")
        
        # Initialize LLM
        llm = initialize_llm()
        
        return retriever, llm
    except Exception as e:
        logging.error(f"Service initialization error: {str(e)}")
        raise

def initialize_feedback_collection():
    """Initialize and verify feedback collection"""
    try:
        client = initialize_database_client()
        
        # Check if collection exists
        collections = client.get_collections().collections
        collection_exists = any(c.name == "chat_feedback" for c in collections)
        
        if not collection_exists:
            # Create collection with proper configuration
            client.create_collection(
                collection_name="chat_feedback",
                vectors_config=VectorParams(
                    size=768,  # Ensure this matches the embedding size
                    distance=Distance.COSINE
                )
            )
            logging.info("Created 'chat_feedback' collection with vector size 768 and Cosine distance.")
        else:
            logging.info("'chat_feedback' collection already exists.")
        
        # Verify collection exists and has correct configuration
        collection_info = client.get_collection("chat_feedback")
        if collection_info.config.params.vectors.size != 768:
            raise ValueError("Incorrect vector size in 'chat_feedback' collection.")
        logging.info("'chat_feedback' collection verified successfully with correct vector size.")
        
        return True
    except Exception as e:
        logging.error(f"Failed to initialize feedback collection: {e}")
        raise

async def submit_feedback(feedback_type, chat_history, language_choice):
    """Submit feedback with improved error handling and logging."""
    try:
        if not chat_history or len(chat_history) < 2:
            logging.warning("Attempted to submit feedback with insufficient chat history.")
            return "No recent interaction to provide feedback for."

        # Get last question and answer
        last_interaction = chat_history[-2:]
        question = last_interaction[0].get("content", "").strip()
        answer = last_interaction[1].get("content", "").strip()

        if not question or not answer:
            logging.warning("Question or answer content is missing.")
            return "Incomplete interaction data. Cannot submit feedback."

        logging.info(f"Processing feedback for question: {question[:50]}...")

        # Initialize client
        client = initialize_database_client()

        # Create point ID
        point_id = str(uuid.uuid4())

        # Create payload
        payload = {
            "question": question,
            "answer": answer,
            "language": language_choice,
            "timestamp": datetime.utcnow().isoformat(),
            "feedback": feedback_type
        }

        # Initialize embeddings
        embeddings = FastEmbedEmbeddings(model_name="nomic-ai/nomic-embed-text-v1.5-Q")

        # Create embeddings for the Q&A pair
        try:
            embedding_text = f"{question} {answer}"
            vector = await asyncio.to_thread(embeddings.embed_query, embedding_text)
            logging.info(f"Generated embedding vector of length {len(vector)}.")
        except Exception as embed_error:
            logging.error(f"Embedding generation failed: {embed_error}")
            return "Failed to generate embeddings for your feedback."

        if not isinstance(vector, list) or not vector:
            logging.error("Invalid vector generated from embeddings.")
            return "Failed to generate valid embeddings for your feedback."

        # Create point
        point = PointStruct(
            id=point_id,
            payload=payload,
            vector=vector
        )

        # Store in Qdrant
        try:
            operation_info = await asyncio.to_thread(
                client.upsert,
                collection_name="chat_feedback",
                points=[point]
            )
            logging.info(f"Feedback submitted successfully: {point_id}")
            return "Thanks for your feedback! Your response has been recorded."
        except Exception as db_error:
            logging.error(f"Failed to upsert point to Qdrant: {db_error}")
            return "Sorry, there was an error submitting your feedback."

    except Exception as e:
        logging.error(f"Unexpected error in submit_feedback: {e}")
        return "Sorry, there was an unexpected error submitting your feedback."

# Initialize services and feedback collection
try:
    retriever, llm = initialize_services()
    initialize_feedback_collection()
except Exception as initialization_error:
    logging.critical(f"Initialization failed: {initialization_error}")
    raise

# Prompt template
prompt_template = PromptTemplate(
    template="""You are RRA Assistant, created by Cedric to help users get tax related information in Rwanda. Your task is to answer tax-related questions using the provided context.

Context: {context}

User's Question: {question}

Please follow these steps to answer the question:

Step 1: Analyze the question
Briefly explain your understanding of the question and any key points to address. If it is hi or hello, skip to step 3 and respond with a greeting.

Step 2: Provide relevant information
Using the context provided, give detailed information related to the question. Include specific facts, figures, or explanations from the context.

Step 3: Final answer
Provide a clear, concise answer to the original question. Start directly with the relevant information, avoiding phrases like "In summary" or "To conclude".

Remember:
- If you don't know the answer or can't find relevant information in the context, say so honestly.
- Do not make up information.
- Use the provided context to support your answer.
- Include "For more information, call 3004" at the end of every answer.

Your response:
""",
    input_variables=['context', 'question']
)

async def process_query(message: str, language: str, chat_history: list) -> str:
    try:
        # Handle translation based on selected language
        if language == "Kinyarwanda":
            query = translator.translate(message, "rw", "en")
            logging.info(f"Translated query to English: {query}")
        else:
            query = message
        
        # Create QA chain
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt_template},
            return_source_documents=True
        )
        
        # Get response
        response = await asyncio.to_thread(
            lambda: qa.invoke({"query": query})
        )
        logging.info("QA chain invoked successfully.")
        
        # Extract final answer
        result_text = response.get('result', '')
        final_answer_start = result_text.find("Step 3: Final answer")
        if final_answer_start != -1:
            answer = result_text[final_answer_start + len("Step 3: Final answer"):].strip()
        else:
            answer = result_text
        
        # Clean up the answer
        answer = re.sub(r'\*\*', '', answer).strip()
        answer = re.sub(r'Step \d+:', '', answer).strip()
        
        # Translate response if needed
        if language == "Kinyarwanda":
            answer = translator.translate(answer, "en", "rw")
            logging.info(f"Translated answer to Kinyarwanda: {answer}")
        
        return answer
    except Exception as e:
        logging.error(f"Query processing error: {str(e)}")
        return f"An error occurred: {str(e)}"

# Define separate feedback submission functions to pass feedback type correctly
async def submit_positive_feedback(chat_history, language_choice):
    return await submit_feedback("positive", chat_history, language_choice)

async def submit_negative_feedback(chat_history, language_choice):
    return await submit_feedback("negative", chat_history, language_choice)

# Create Gradio interface
with gr.Blocks(title="RRA FAQ Chatbot") as demo:
    gr.Markdown(
        """
        # RRA FAQ Chatbot
        Ask tax-related questions in English or Kinyarwanda
        > 🔒 Your questions and interactions remain private unless you choose to submit feedback, which helps improve our service.
        """
    )
    
    # Add language selector
    language = gr.Radio(
        choices=["English", "Kinyarwanda"],
        value="English",
        label="Select Language / Hitamo Ururimi"
    )
    
    chatbot = gr.Chatbot(
        value=[],
        show_label=False,
        height=400,
        type='messages'
    )
    
    with gr.Row():
        msg = gr.Textbox(
            label="Ask your question",
            placeholder="Type your tax-related question here...",
            show_label=False
        )
        submit = gr.Button("Send")
    
    # Add feedback section
    with gr.Row():
        with gr.Column(scale=2):
            feedback_label = gr.Markdown("Was this response helpful?")
        with gr.Column(scale=1):
            feedback_positive = gr.Button("👍 Helpful")
        with gr.Column(scale=1):
            feedback_negative = gr.Button("👎 Not Helpful")
    
    # Add feedback status message
    feedback_status = gr.Markdown("")
    
    # Connect feedback buttons to their respective functions
    feedback_positive.click(
        fn=submit_positive_feedback,
        inputs=[chatbot, language],
        outputs=feedback_status
    )
    
    feedback_negative.click(
        fn=submit_negative_feedback,
        inputs=[chatbot, language],
        outputs=feedback_status
    )
    
    # Create two sets of examples
    with gr.Row() as english_examples_row:
        gr.Examples(
            examples=[
                "What is VAT in Rwanda?",
                "How do I register for taxes?",
                "What are the tax payment deadlines?",
                "How can I get a TIN number?",
                "How do I get purchase code?"
            ],
            inputs=msg,
            label="English Examples"
        )
    
    with gr.Row(visible=False) as kinyarwanda_examples_row:
        gr.Examples(
            examples=[
                "Ese VAT ni iki mu Rwanda?",
                "Nabona TIN number nte?",
                "Ni ryari tugomba kwishyura imisoro?",
                "Ese nandikwa nte ku musoro?",
                "Ni gute nabone kode yo kugura?"
            ],
            inputs=msg,
            label="Kinyarwanda Examples"
        )
    
    async def respond(message, lang, chat_history):
        bot_message = await process_query(message, lang, chat_history)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        return "", chat_history
    
    def toggle_language_interface(language_choice):
        if language_choice == "English":
            placeholder_text = "Type your tax-related question here..."
            return {
                msg: gr.update(placeholder=placeholder_text),
                english_examples_row: gr.update(visible=True),
                kinyarwanda_examples_row: gr.update(visible=False)
            }
        else:
            placeholder_text = "Andika ibibazo bijyanye n'umusoro hano"
            return {
                msg: gr.update(placeholder=placeholder_text),
                english_examples_row: gr.update(visible=False),
                kinyarwanda_examples_row: gr.update(visible=True)
            }
    
    msg.submit(respond, [msg, language, chatbot], [msg, chatbot])
    submit.click(respond, [msg, language, chatbot], [msg, chatbot])
    
    # Update both examples visibility and placeholder when language changes
    language.change(
        fn=toggle_language_interface,
        inputs=language,
        outputs=[msg, english_examples_row, kinyarwanda_examples_row]
    )
    
    gr.Markdown(
        """
        ### About
        - Created by: [Cedric](mailto:mugishac777@gmail.com)
        - Data source: [RRA Website FAQ](https://www.rra.gov.rw/en/domestic-tax-services/faqs)
        
        **Disclaimer:** This chatbot provides general tax information. For official guidance, 
        consult RRA or call 3004.
        🔒 **Privacy:** Your interactions remain private unless you choose to submit feedback.
        """
    )

# Launch the app
if __name__ == "__main__":
    try:
        demo.launch(share=False)
        logging.info("Gradio app launched successfully.")
    except Exception as launch_error:
        logging.critical(f"Failed to launch Gradio app: {launch_error}")
        raise
