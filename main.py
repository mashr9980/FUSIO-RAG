
import asyncio
import os
from typing import Dict, Optional, List
from datetime import datetime
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import json
import uuid
from fastapi.responses import FileResponse, RedirectResponse
import httpx
import numpy as np
import faiss
import pickle
from pathlib import Path
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from models.models import DocumentUpload, DocumentResponse, DocumentStatus
from models.websearch import DirectSearchService, WebSearchService
from config import config
from models.llm import LLMModel
from utils.helpers import Timer
from langchain.callbacks.base import BaseCallbackHandler
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentStatus:
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DocumentStore:
    def __init__(self, base_path: str):
        logger.info(f"Initializing DocumentStore with base path: {base_path}")
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_path / "faiss_index"
        self.metadata_path = self.base_path / "metadata.pickle"
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDINGS_MODEL,
            model_kwargs={'device': "cuda" if torch.cuda.is_available() else "cpu"}
        )
        
        # Initialize or load FAISS index and metadata
        self._initialize_storage()

    def _initialize_storage(self):
        logger.info("Initializing storage")
        try:
            if self.index_path.exists() and self.metadata_path.exists():
                logger.info("Loading existing index and metadata")
                self.index = faiss.read_index(str(self.index_path))
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
            else:
                logger.info("Creating new index and metadata")
                embedding_dim = len(self.embeddings.embed_query("test"))
                self.index = faiss.IndexFlatL2(embedding_dim)
                self.metadata = {
                    'documents': {}, 
                    'id_mapping': {}  
                }
                self._save_storage()
        except Exception as e:
            logger.error(f"Error initializing storage: {str(e)}")
            raise

    def _save_storage(self):
        logger.info("Saving storage")
        try:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            logger.error(f"Error saving storage: {str(e)}")
            raise

    async def add_document(self, document_id: str, url: str = None, filename: str = None) -> None:
        logger.info(f"Adding document {document_id} with filename {filename}")
        self.metadata['documents'][document_id] = {
            'status': DocumentStatus.PROCESSING,
            'chunks': [],
            'filename': filename,  # Store filename instead of URL
            'url': url,  # Keep URL field for backward compatibility
            'created_at': datetime.utcnow().isoformat()
        }
        self._save_storage()

    async def process_document(self, document_id: str, pdf_path: str) -> None:
        logger.info(f"Processing document {document_id}")
        try:
            # Load and split document
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.SPLIT_CHUNK_SIZE,
                chunk_overlap=config.SPLIT_OVERLAP
            )
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Split document into {len(chunks)} chunks")
            
            # Create embeddings for chunks
            chunk_texts = [chunk.page_content for chunk in chunks]
            logger.info("Creating embeddings")
            embeddings = self.embeddings.embed_documents(chunk_texts)
            
            # Add to FAISS index
            logger.info("Adding to FAISS index")
            start_idx = self.index.ntotal
            self.index.add(np.array(embeddings))
            
            # Update metadata
            chunk_metadata = []
            for i, chunk in enumerate(chunks):
                faiss_id = start_idx + i
                self.metadata['id_mapping'][faiss_id] = (document_id, i)
                chunk_metadata.append({
                    'text': chunk.page_content,
                    'page': chunk.metadata.get('page', 0)
                })
            
            logger.info("Updating document status")
            self.metadata['documents'][document_id].update({
                'status': DocumentStatus.COMPLETED,
                'chunks': chunk_metadata
            })
            
            self._save_storage()
            logger.info(f"Successfully processed document {document_id}")
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            self.metadata['documents'][document_id]['status'] = DocumentStatus.FAILED
            self.metadata['documents'][document_id]['error'] = str(e)
            self._save_storage()
            raise

    async def search(self, document_id: str, query: str, k: int = 4) -> List[str]:
        """Search for relevant chunks in a specific document"""
        if document_id not in self.metadata['documents']:
            raise ValueError(f"Document {document_id} not found")
            
        if self.metadata['documents'][document_id]['status'] != DocumentStatus.COMPLETED:
            raise ValueError(f"Document {document_id} is not ready")
            
        # Create query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Search in FAISS
        D, I = self.index.search(np.array([query_embedding]), k * 2)  # Get more results than needed
        
        # Filter results for specific document
        relevant_chunks = []
        for idx in I[0]:
            if idx != -1:  # Valid FAISS id
                doc_id, chunk_id = self.metadata['id_mapping'][int(idx)]
                if doc_id == document_id:
                    chunk = self.metadata['documents'][doc_id]['chunks'][chunk_id]
                    relevant_chunks.append(chunk['text'])
                    if len(relevant_chunks) == k:
                        break
                        
        return relevant_chunks

    def get_document_status(self, document_id: str) -> Optional[Dict]:
        """Get document processing status"""
        return self.metadata['documents'].get(document_id)

class RAGApplication:
    def __init__(self):
        self.app = FastAPI()
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        @self.app.get("/")
        async def get_main_page():
            return FileResponse("static/disclaimer.html")
        
        @self.app.get("/chatbot")
        async def get_chatbot_page(request: Request):
            # Check for disclaimer acceptance
            disclaimer_accepted = request.cookies.get("disclaimer_accepted")
            if not disclaimer_accepted:
                # Redirect to disclaimer page if not accepted
                return RedirectResponse(url="/")
            return FileResponse("static/chatbot.html")
        
        # Initialize core components
        self.llm_model = LLMModel()
        self.active_connections = {}
        self.document_store = DocumentStore(config.OUTPUT_FOLDER)

        @self.app.on_event("startup")
        async def start_heartbeat_task():
            asyncio.create_task(self.websocket_heartbeat())
        
        # Set up routes
        self.setup_routes()
    
    async def websocket_heartbeat(self):
        """Send periodic heartbeats to keep WebSocket connections alive"""
        while True:
            await asyncio.sleep(30)  # Send heartbeat every 30 seconds
            
            # Copy the dictionary keys to avoid modification during iteration
            document_ids = list(self.active_connections.keys())
            
            for doc_id in document_ids:
                if doc_id in self.active_connections:
                    client_ids = list(self.active_connections[doc_id].keys())
                    
                    for client_id in client_ids:
                        try:
                            if client_id in self.active_connections.get(doc_id, {}):
                                websocket = self.active_connections[doc_id][client_id]
                                await websocket.send_text(json.dumps({
                                    "status": "heartbeat"
                                }))
                        except Exception as e:
                            logger.error(f"Error sending heartbeat: {str(e)}")
                            # Remove failed connection
                            if doc_id in self.active_connections and client_id in self.active_connections[doc_id]:
                                del self.active_connections[doc_id][client_id]
                                if not self.active_connections[doc_id]:
                                    del self.active_connections[doc_id]

    async def process_document_task(self, document_id: str, temp_pdf_path: str):
        """Background task for document processing"""
        logger.info(f"Starting processing for document {document_id}")
        try:
            # Process the uploaded document
            logger.info(f"Processing document {document_id}")
            await self.document_store.process_document(document_id, temp_pdf_path)
            logger.info(f"Successfully processed document {document_id}")
                
        except Exception as e:
            logger.error(f"Error during document processing: {str(e)}")
            # Update document status to failed
            if hasattr(self, 'document_store'):
                self.document_store.metadata['documents'][document_id]['status'] = DocumentStatus.FAILED
                self.document_store.metadata['documents'][document_id]['error'] = str(e)
                self.document_store._save_storage()
        finally:
            # Cleanup
            if os.path.exists(temp_pdf_path):
                logger.info(f"Removing temporary file {temp_pdf_path}")
                os.remove(temp_pdf_path)

    def setup_routes(self):
        @self.app.post("/api/documents", response_model=DocumentResponse)
        async def upload_document(
            file: UploadFile = File(...),
            background_tasks: BackgroundTasks = BackgroundTasks()
        ):
            try:
                # Generate unique ID
                document_id = str(uuid.uuid4())
                logger.info(f"Created document ID: {document_id}")
                
                # Save uploaded file temporarily
                temp_pdf_path = f"temp_{document_id}.pdf"
                logger.info(f"Saving uploaded file to {temp_pdf_path}")
                
                # Read and save the file content
                content = await file.read()
                with open(temp_pdf_path, "wb") as f:
                    f.write(content)
                
                # Register document with filename
                await self.document_store.add_document(document_id, filename=file.filename)
                logger.info(f"Registered document {document_id}")
                
                # Start processing in background
                background_tasks.add_task(
                    self.process_document_task,
                    document_id,
                    temp_pdf_path
                )
                logger.info(f"Added background task for document {document_id}")
                
                return DocumentResponse(
                    document_id=document_id,
                    message="Document upload initiated"
                )
                
            except Exception as e:
                logger.error(f"Error in upload_document: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/documents/{document_id}/status")
        async def get_document_status(document_id: str):
            logger.info(f"Checking status for document {document_id}")
            try:
                status = self.document_store.get_document_status(document_id)
                if not status:
                    logger.info(f"Document {document_id} not found")
                    raise HTTPException(status_code=404, detail="Document not found")
                logger.info(f"Document {document_id} status: {status['status']}")
                return {
                    "document_id": document_id,
                    "status": status['status'],
                    "error": status.get('error'),
                    "created_at": status.get('created_at'),
                    "chunks_count": len(status.get('chunks', []))
                }
            except Exception as e:
                logger.error(f"Error getting document status: {str(e)}")
                raise
    
        @self.app.websocket("/ws/chat")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
        
            document_id = None
            is_initialized = False
            # Create a list to store chat history
            chat_history = []
            
            # Generate a unique client ID for this connection
            client_id = str(uuid.uuid4())
        
            try:
                # Step 1: Initialization - receive document_id
                init_data = await websocket.receive_text()
                init_message = json.loads(init_data)
        
                document_id = init_message.get("document_id")
                if not document_id:
                    await websocket.send_text(json.dumps({
                        "status": "error",
                        "error": "Missing document_id in initialization message."
                    }))
                    return
        
                # Store this connection in the active connections dictionary
                if document_id not in self.active_connections:
                    self.active_connections[document_id] = {}
                self.active_connections[document_id][client_id] = websocket
        
                # Validate document status
                status = self.document_store.get_document_status(document_id)
                if not status:
                    await websocket.send_text(json.dumps({
                        "status": "error",
                        "error": "Document not found."
                    }))
                    return
        
                if status['status'] != "completed":
                    await websocket.send_text(json.dumps({
                        "status": "error",
                        "error": f"Document is not ready yet. (status: {status['status']})"
                    }))
                    return
        
                # Prepare the combined prompt with chat history
                base_prompt = (
                    "You are Scott S. McLean, a seasoned financial advisor with years of experience.\n"
                    "You must respond as a financial expert who is:\n"
                    "- Friendly\n"
                    "- Professional\n"
                    "- Polite\n"
                    "- Patient\n"
                    "- Knowledgeable\n"
                    "- Empathetic\n"
                    "- Clear Communicator\n\n"
                    "CRITICAL INSTRUCTION: If the LATEST FINANCIAL DATA section contains information relevant to the user's question, "
                    "you MUST prioritize this information over anything in the document context or your general knowledge. "
                    "The LATEST FINANCIAL DATA contains the most current and accurate information.\n\n"
                    
                    "CONVERSATIONAL FLOW GUIDELINES:\n"
                    "1. Detect User Communication Style from their messages:\n"
                    "   - Visual: Look for words like 'see', 'look', 'picture this'\n"
                    "   - Auditory: Look for words like 'hear', 'talk', 'sounds like'\n"
                    "   - Kinesthetic: Look for words like 'feel', 'hold', 'walk through'\n"
                    "   - Logical: Look for words like 'calculate', 'break down', 'analyze'\n\n"
                    
                    "2. Adapt Your Response Format to Match Their Style:\n"
                    "   - For Visual users: Use imagery, diagrams, and visual metaphors\n"
                    "   - For Auditory users: Use dialogue, rhythm, and discuss how things sound\n"
                    "   - For Kinesthetic users: Use action-based examples and tangible metaphors\n"
                    "   - For Logical users: Use numbered lists, clear steps, and logical frameworks\n\n"
                    
                    "3. Incorporate These NLP Techniques:\n"
                    "   - Presuppositions: 'When we review your full financial picture...'\n"
                    "   - Embedded Suggestions: 'You might start to realize there's more opportunity here...'\n"
                    "   - Tag Questions: 'That's something worth considering, isn't it?'\n\n"
                    
                    "Example Responses Based on Style:\n"
                    "- Visual User: 'Picture a trust like a protective folder where you place your valuable assets. You decide who opens that folder and when.'\n"
                    "- Logical User: 'Let's break it into 3 key steps: 1) Move pre-tax IRA funds into a Roth account. 2) Pay taxes now to enjoy tax-free growth. 3) Withdraw tax-free in retirement.'\n\n"
                    
                    "When using information from the LATEST FINANCIAL DATA section:\n"
                    "- Include specific figures, limits, rates, and dates from this section\n"
                    "- Explicitly mention the year (e.g., 'For 2025, the contribution limit is...')\n"
                    "- Make sure to incorporate the exact numbers and thresholds provided\n\n"
                    
                    "Only use the document CONTEXT if the LATEST FINANCIAL DATA doesn't address the question.\n\n"
                    "If neither source has the answer but the topic is related to financial planning, use your professional knowledge carefully.\n\n"
                    "If the question is unrelated to financial topics (e.g., movies, cooking, gaming), politely respond: 'I'm specialized in financial planning topics. Please ask me something related to finance, investment, or retirement.'\n\n"
                    "Use the CHAT HISTORY to maintain continuity in the conversation.\n\n"
                    "Important instructions when responding:\n"
                    "- Start with a short, polite greeting that sounds warm, human, and professional.\n"
                    "- Make answers concise and focused - no more than 4-6 sentences unless necessary.\n"
                    "- ALWAYS prioritize accuracy and current information over document context.\n"
                    "- Avoid over-explaining. Prioritize clarity and brevity.\n\n"
                    "LATEST FINANCIAL DATA:\n{latest_financial_data}\n\n"
                    "CONTEXT FROM DOCUMENT:\n{context}\n\n"
                    "CHAT HISTORY:\n{chat_history}\n\n"
                    "User's Question: {input}\n\n"
                    "Respond naturally without using section headers."
                )
        
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", base_prompt),
                    ("human", "{input}")
                ])
        
                # Confirm initialization
                await websocket.send_text(json.dumps({
                    "status": "initialized",
                    "document_id": document_id,
                    "message": "Connection initialized successfully. You can now send questions."
                }))
                is_initialized = True
        
                # Step 2: Main chat loop
                while True:
                    try:
                        data = await websocket.receive_text()
        
                        try:
                            message_data = json.loads(data)
                            question = message_data.get("question", data)
                        except json.JSONDecodeError:
                            question = data
        
                        if not question or not isinstance(question, str):
                            await websocket.send_text(json.dumps({
                                "status": "error",
                                "error": "Invalid question format."
                            }))
                            continue
        
                        with Timer() as timer:
                            # Detect if question likely needs current financial information
                            time_sensitive_keywords = [
                                "current", "latest", "2024", "2025", "2026", "today", "this year", 
                                "recent", "now", "limit", "contribution", "tax rate", "interest rate",
                                "regulation", "law", "deadline", "cap", "maximum", "minimum",
                                "cutoff", "threshold", "bracket", "percent", "market", "trend",
                                "update", "change", "new rule", "policy", "legislation"
                            ]
                            financial_keywords = [
                                "ira", "401k", "401(k)", "roth", "traditional", "tax", "deduction", 
                                "income", "investment", "retirement", "social security", "medicare",
                                "capital gain", "dividend", "interest", "mortgage", "loan", "debt",
                                "credit", "inflation", "rate", "contribution", "withdrawal", "distribution"
                            ]
                            
                            is_time_sensitive = any(keyword in question.lower() for keyword in time_sensitive_keywords)
                            is_financial = any(keyword in question.lower() for keyword in financial_keywords)
                            
                            # ALWAYS get latest financial data for financial questions
                            latest_financial_data = ""
                            if is_financial:
                                await websocket.send_text(json.dumps({
                                    "status": "searching",
                                    "message": "Checking for the latest financial information..."
                                }))
                                
                                # Use the DirectSearchService to get updated financial information
                                search_service = DirectSearchService(config.OPENAI_API_KEY)
                                success, search_result = await search_service.search(question)
                                
                                if success:
                                    latest_financial_data = search_result
                                else:
                                    logger.warning(f"Direct search failed: {search_result}")
                                    latest_financial_data = "(No current financial data available. Using document context.)"
                            else:
                                latest_financial_data = "(Not a financial question - no current data needed)"
                            
                            # Retrieve context chunks from document as backup
                            context_chunks = await self.document_store.search(
                                document_id,
                                question,
                                k=config.SIMILAR_DOCS_COUNT
                            )
        
                            documents = [Document(page_content=chunk) for chunk in context_chunks]
                            retriever = get_static_retriever(documents)
        
                            # Create streaming callback handler
                            class WebSocketCallbackHandler(BaseCallbackHandler):
                                def __init__(self, websocket):
                                    self.websocket = websocket
                                    self.collected_tokens = ""
                                    
                                async def on_llm_new_token(self, token: str, **kwargs):
                                    # Send each token as it comes
                                    self.collected_tokens += token
                                    await self.websocket.send_text(json.dumps({
                                        "status": "streaming",
                                        "token": token
                                    }))
                                        
                            # Create callback manager with our handler
                            callback_handler = WebSocketCallbackHandler(websocket)
                            
                            # Format chat history for inclusion in the prompt
                            formatted_chat_history = ""
                            for entry in chat_history:
                                formatted_chat_history += f"User: {entry['question']}\nScott: {entry['answer']}\n\n"
                            
                            # Create retrieval-based QA chain with streaming
                            qa_chain = create_retrieval_chain(
                                retriever,
                                create_stuff_documents_chain(
                                    self.llm_model.get_llm().with_config(
                                        {"callbacks": [callback_handler]}
                                    ),
                                    prompt_template
                                )
                            )
        
                            # Generate combined response with latest financial data prioritized over document context
                            result = await qa_chain.ainvoke({
                                "input": question,
                                "chat_history": formatted_chat_history,
                                "latest_financial_data": latest_financial_data,
                                "context": "\n\n".join(context_chunks) if context_chunks else "(No relevant document context found)"
                            })
                            final_response = result.get("answer", "").strip()
        
                            # Add the current Q&A to chat history
                            chat_history.append({
                                "question": question,
                                "answer": final_response
                            })
                            
                            # Keep chat history to a reasonable size (last 10 exchanges)
                            if len(chat_history) > 10:
                                chat_history = chat_history[-10:]
        
                        # Send final complete response
                        await websocket.send_text(json.dumps({
                            "status": "complete",
                            "answer": final_response,
                            "time": timer.interval,
                            "used_latest_data": is_financial and latest_financial_data != "(No current financial data available. Using document context.)"
                        }))
        
                    except WebSocketDisconnect:
                        logger.info(f"WebSocket disconnected (initialized: {is_initialized})")
                        break
                    except Exception as e:
                        logger.error(f"Error in WebSocket chat: {str(e)}")
                        await websocket.send_text(json.dumps({
                            "status": "error",
                            "error": str(e)
                        }))
        
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected before full initialization.")
            except Exception as e:
                logger.error(f"WebSocket startup error: {str(e)}")
                await websocket.send_text(json.dumps({
                    "status": "error",
                    "error": str(e)
                }))
            finally:
                # Clean up the connection when done
                if document_id and document_id in self.active_connections and client_id in self.active_connections[document_id]:
                    del self.active_connections[document_id][client_id]
                    # Clean up empty dictionaries
                    if not self.active_connections[document_id]:
                        del self.active_connections[document_id]

        # Helper for static retriever
        def get_static_retriever(documents):
            class StaticRetriever(BaseRetriever):
                def _get_relevant_documents(self, query: str):
                    return documents

                async def _aget_relevant_documents(self, query: str):
                    return documents

            return StaticRetriever()