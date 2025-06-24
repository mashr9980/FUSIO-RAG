import asyncio
import os
from typing import Dict, Optional, List
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import json
import uuid
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
from models.models import DocumentResponse, DocumentStatus
from config import config
from models.llm import LLMModel
from utils.helpers import Timer
from langchain.callbacks.base import BaseCallbackHandler
import logging

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
            'filename': filename,
            'url': url,
            'created_at': datetime.utcnow().isoformat()
        }
        self._save_storage()

    async def process_document(self, document_id: str, pdf_path: str) -> None:
        logger.info(f"Processing document {document_id}")
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.SPLIT_CHUNK_SIZE,
                chunk_overlap=config.SPLIT_OVERLAP
            )
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Split document into {len(chunks)} chunks")
            
            chunk_texts = [chunk.page_content for chunk in chunks]
            logger.info("Creating embeddings")
            embeddings = self.embeddings.embed_documents(chunk_texts)
            
            logger.info("Adding to FAISS index")
            start_idx = self.index.ntotal
            self.index.add(np.array(embeddings))
            
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
        if document_id not in self.metadata['documents']:
            raise ValueError(f"Document {document_id} not found")
            
        if self.metadata['documents'][document_id]['status'] != DocumentStatus.COMPLETED:
            raise ValueError(f"Document {document_id} is not ready")
            
        query_embedding = self.embeddings.embed_query(query)
        
        D, I = self.index.search(np.array([query_embedding]), k * 2)
        
        relevant_chunks = []
        for idx in I[0]:
            if idx != -1:
                doc_id, chunk_id = self.metadata['id_mapping'][int(idx)]
                if doc_id == document_id:
                    chunk = self.metadata['documents'][doc_id]['chunks'][chunk_id]
                    relevant_chunks.append(chunk['text'])
                    if len(relevant_chunks) == k:
                        break
                        
        return relevant_chunks

    def get_document_status(self, document_id: str) -> Optional[Dict]:
        return self.metadata['documents'].get(document_id)

class RAGApplication:
    def __init__(self):
        self.app = FastAPI()
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self.llm_model = LLMModel()
        self.active_connections = {}
        self.document_store = DocumentStore(config.OUTPUT_FOLDER)

        @self.app.on_event("startup")
        async def start_heartbeat_task():
            asyncio.create_task(self.websocket_heartbeat())
        
        self.setup_routes()
    
    async def websocket_heartbeat(self):
        while True:
            await asyncio.sleep(30)
            
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
                            if doc_id in self.active_connections and client_id in self.active_connections[doc_id]:
                                del self.active_connections[doc_id][client_id]
                                if not self.active_connections[doc_id]:
                                    del self.active_connections[doc_id]

    async def process_document_task(self, document_id: str, temp_pdf_path: str):
        logger.info(f"Starting processing for document {document_id}")
        try:
            logger.info(f"Processing document {document_id}")
            await self.document_store.process_document(document_id, temp_pdf_path)
            logger.info(f"Successfully processed document {document_id}")
                
        except Exception as e:
            logger.error(f"Error during document processing: {str(e)}")
            if hasattr(self, 'document_store'):
                self.document_store.metadata['documents'][document_id]['status'] = DocumentStatus.FAILED
                self.document_store.metadata['documents'][document_id]['error'] = str(e)
                self.document_store._save_storage()
        finally:
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
                document_id = str(uuid.uuid4())
                logger.info(f"Created document ID: {document_id}")
                
                temp_pdf_path = f"temp_{document_id}.pdf"
                logger.info(f"Saving uploaded file to {temp_pdf_path}")
                
                content = await file.read()
                with open(temp_pdf_path, "wb") as f:
                    f.write(content)
                
                await self.document_store.add_document(document_id, filename=file.filename)
                logger.info(f"Registered document {document_id}")
                
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
            chat_history = []
            
            client_id = str(uuid.uuid4())
        
            try:
                init_data = await websocket.receive_text()
                init_message = json.loads(init_data)
        
                document_id = init_message.get("document_id")
                if not document_id:
                    await websocket.send_text(json.dumps({
                        "status": "error",
                        "error": "Missing document_id in initialization message."
                    }))
                    return
        
                if document_id not in self.active_connections:
                    self.active_connections[document_id] = {}
                self.active_connections[document_id][client_id] = websocket
        
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
        
                base_prompt = (
                    "You are a Fusio Assistant a Financial Planning Assistant, a knowledgeable and helpful AI specialized in the Four Bucket Financial System.\n"
                    "You must respond as a financial planning expert who is:\n"
                    "- Friendly and approachable\n"
                    "- Professional and reliable\n"
                    "- Clear and concise\n"
                    "- Patient and understanding\n"
                    "- Knowledgeable about the bucket system\n"
                    "- Helpful and supportive\n\n"
                    
                    "CRITICAL INSTRUCTION: You MUST ONLY provide information that is available in the provided FINANCIAL PLANNING DOCUMENTATION. "
                    "Do not use external knowledge or make assumptions about strategies not mentioned in the documentation.\n\n"
                    
                    "THE FOUR BUCKET SYSTEM:\n"
                    "Based on the documentation, always reference these four buckets when relevant:\n"
                    "- Bucket 1: Cash/Reserves (Emergency funds, working capital, short-term planned expenses)\n"
                    "- Bucket 2: Cash Flow (Income vs expenses, passive income investments)\n"
                    "- Bucket 3: Legacy/Growth (Long-term wealth building, retirement accounts, investments)\n"
                    "- Bucket 4: Protection (Insurance, estate planning, risk management)\n\n"
                    
                    "RESPONSE GUIDELINES:\n"
                    "1. Always prioritize information from the FINANCIAL PLANNING DOCUMENTATION section\n"
                    "2. Keep ALL responses short and concise - maximum 2-3 sentences unless absolutely necessary\n"
                    "3. Get straight to the point - no lengthy explanations or background context\n"
                    "4. Use specific numbers, percentages, and rules from the docs when relevant\n"
                    "5. Reference bucket numbers when applicable (Bucket 1, 2, 3, or 4)\n"
                    "6. If the documentation doesn't contain the answer, give a brief response about related bucket concepts\n\n"
                    
                    "CONVERSATIONAL APPROACH:\n"
                    "- Give brief, direct answers without unnecessary greetings\n"
                    "- Focus on actionable information only\n"
                    "- Reference specific bucket numbers when relevant\n"
                    "- Skip background explanations unless specifically asked\n\n"
                    
                    "BOUNDARY HANDLING:\n"
                    "If someone asks about topics unrelated to financial planning or the bucket system (e.g., cooking, sports, entertainment, general life advice), "
                    "politely respond: 'I'm a Financial Planning Assistant specialized in helping with the Four Bucket Financial System. "
                    "I'm here to help you with questions about cash reserves, cash flow, legacy/growth investments, and financial protection strategies. "
                    "Please ask me something related to financial planning, budgeting, investing, or the bucket system, and I'll be happy to assist you!'\n\n"
                    
                    "FINANCIAL PLANNING DOCUMENTATION:\n{context}\n\n"
                    
                    "CHAT HISTORY:\n{chat_history}\n\n"
                    
                    "User's Question: {input}\n\n"
                    
                    "Instructions for your response:\n"
                    "- Answer based ONLY on the financial planning documentation provided\n"
                    "- Keep responses SHORT - maximum 2-3 sentences\n"
                    "- Get straight to the point with actionable information\n"
                    "- Reference bucket numbers when relevant\n"
                    "- Include specific numbers from docs when applicable\n"
                    "- Skip greetings and lengthy explanations\n\n"
                    
                    "Respond naturally without using section headers or referencing this prompt structure."
                )
        
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", base_prompt),
                    ("human", "{input}")
                ])
        
                await websocket.send_text(json.dumps({
                    "status": "initialized",
                    "document_id": document_id,
                    "message": "Connection initialized successfully. You can now send questions."
                }))
                is_initialized = True
        
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
                            context_chunks = await self.document_store.search(
                                document_id,
                                question,
                                k=config.SIMILAR_DOCS_COUNT
                            )
        
                            documents = [Document(page_content=chunk) for chunk in context_chunks]
                            retriever = get_static_retriever(documents)
        
                            class WebSocketCallbackHandler(BaseCallbackHandler):
                                def __init__(self, websocket):
                                    self.websocket = websocket
                                    self.collected_tokens = ""
                                    
                                async def on_llm_new_token(self, token: str, **kwargs):
                                    self.collected_tokens += token
                                    await self.websocket.send_text(json.dumps({
                                        "status": "streaming",
                                        "token": token
                                    }))
                                        
                            callback_handler = WebSocketCallbackHandler(websocket)
                            
                            formatted_chat_history = ""
                            for entry in chat_history:
                                formatted_chat_history += f"User: {entry['question']}\nScott: {entry['answer']}\n\n"
                            
                            qa_chain = create_retrieval_chain(
                                retriever,
                                create_stuff_documents_chain(
                                    self.llm_model.get_llm().with_config(
                                        {"callbacks": [callback_handler]}
                                    ),
                                    prompt_template
                                )
                            )
        
                            result = await qa_chain.ainvoke({
                                "input": question,
                                "chat_history": formatted_chat_history,
                                "context": "\n\n".join(context_chunks) if context_chunks else "(No relevant document context found)"
                            })
                            final_response = result.get("answer", "").strip()
        
                            chat_history.append({
                                "question": question,
                                "answer": final_response
                            })
                            
                            if len(chat_history) > 10:
                                chat_history = chat_history[-10:]
        
                        await websocket.send_text(json.dumps({
                            "status": "complete",
                            "answer": final_response,
                            "time": timer.interval,
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
                if document_id and document_id in self.active_connections and client_id in self.active_connections[document_id]:
                    del self.active_connections[document_id][client_id]
                    if not self.active_connections[document_id]:
                        del self.active_connections[document_id]

        def get_static_retriever(documents):
            class StaticRetriever(BaseRetriever):
                def _get_relevant_documents(self, query: str):
                    return documents

                async def _aget_relevant_documents(self, query: str):
                    return documents

            return StaticRetriever()