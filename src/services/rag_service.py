"""
RAG (Retrieval-Augmented Generation) Service for Pipecat.
Integrates vectorstore retrieval with LLM generation.
"""

import asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path

from pipecat.frames.frames import Frame, TextFrame, ErrorFrame
from pipecat.processors.frame_processor import FrameProcessor
from loguru import logger

try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain not available. Install: pip install langchain langchain-community")


class RAGService(FrameProcessor):
    """
    RAG service that retrieves relevant context before LLM generation.
    
    Features:
    - Multi-subject document retrieval
    - Semantic search with embeddings
    - Context enrichment for LLM
    - Subject-specific routing
    
    Args:
        vectorstore_path: Path to ChromaDB vectorstore
        collection_name: Name of the collection
        embedding_model: HuggingFace embedding model name
        top_k: Number of documents to retrieve
        score_threshold: Minimum similarity score (0.0-1.0)
    """
    
    def __init__(
        self,
        vectorstore_path: str = "./data/vectorstore",
        collection_name: str = "documents",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 4,
        score_threshold: float = 0.5,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        if not LANGCHAIN_AVAILABLE:
            raise RuntimeError("LangChain not installed. Run: pip install langchain langchain-community")
        
        self.vectorstore_path = Path(vectorstore_path)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.top_k = top_k
        self.score_threshold = score_threshold
        
        self._embeddings = None
        self._vectorstore = None
        self._initialized = False
        
        logger.info(f"RAGService initialized (path: {vectorstore_path}, model: {embedding_model})")
    
    async def _initialize(self):
        """Initialize embeddings and vectorstore."""
        if self._initialized:
            return
        
        try:
            logger.info("Loading embedding model...")
            
            # Load embeddings in executor
            loop = asyncio.get_event_loop()
            self._embeddings = await loop.run_in_executor(
                None,
                lambda: HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name,
                    model_kwargs={'device': 'cuda'},  # Use GPU if available
                    encode_kwargs={'normalize_embeddings': True}
                )
            )
            
            # Load or create vectorstore
            if self.vectorstore_path.exists():
                logger.info(f"Loading existing vectorstore from {self.vectorstore_path}")
                self._vectorstore = Chroma(
                    persist_directory=str(self.vectorstore_path),
                    embedding_function=self._embeddings,
                    collection_name=self.collection_name
                )
            else:
                logger.warning(f"Vectorstore not found at {self.vectorstore_path}")
                logger.info("Creating empty vectorstore...")
                self.vectorstore_path.mkdir(parents=True, exist_ok=True)
                self._vectorstore = Chroma(
                    persist_directory=str(self.vectorstore_path),
                    embedding_function=self._embeddings,
                    collection_name=self.collection_name
                )
            
            self._initialized = True
            logger.success("RAG service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG: {e}")
            raise
    
    async def process_frame(self, frame: Frame, direction: str = "down"):
        """
        Process text frames and enrich with retrieved context.
        
        Args:
            frame: Input frame (TextFrame expected)
            direction: Processing direction
        """
        await super().process_frame(frame, direction)
        
        # Only process TextFrame
        if not isinstance(frame, TextFrame):
            await self.push_frame(frame, direction)
            return
        
        # Initialize if needed
        if not self._initialized:
            await self._initialize()
        
        try:
            query = frame.text
            
            # Retrieve relevant context
            context = await self._retrieve(query)
            
            if context:
                # Create enriched frame with context metadata
                enriched_frame = TextFrame(
                    text=query,
                    metadata={"rag_context": context}
                )
                await self.push_frame(enriched_frame, direction)
                logger.debug(f"Retrieved {len(context)} documents for: {query[:50]}...")
            else:
                # No context found, pass through original frame
                await self.push_frame(frame, direction)
                logger.debug("No relevant context found")
                
        except Exception as e:
            logger.error(f"RAG error: {e}")
            # Pass through original frame on error
            await self.push_frame(frame, direction)
    
    async def _retrieve(self, query: str) -> Optional[str]:
        """
        Retrieve relevant documents for the query.
        
        Args:
            query: User's question
            
        Returns:
            Concatenated context from retrieved documents
        """
        if not self._vectorstore:
            return None
        
        try:
            # Perform similarity search in executor
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._vectorstore.similarity_search_with_score(
                    query,
                    k=self.top_k
                )
            )
            
            # Filter by score threshold
            relevant_docs = [
                (doc, score) for doc, score in results
                if score >= self.score_threshold
            ]
            
            if not relevant_docs:
                return None
            
            # Format context
            context_parts = []
            for i, (doc, score) in enumerate(relevant_docs, 1):
                metadata = doc.metadata
                source = metadata.get("source", "Unknown")
                subject = metadata.get("subject", "Unknown")
                
                context_parts.append(
                    f"[Document {i} - {subject} - Score: {score:.2f}]\n"
                    f"Source: {source}\n"
                    f"Contenu: {doc.page_content}\n"
                )
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return None
    
    async def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        subject: Optional[str] = None
    ):
        """
        Add documents to the vectorstore.
        
        Args:
            documents: List of document texts
            metadatas: Optional list of metadata dicts
            subject: Subject label for all documents
        """
        if not self._initialized:
            await self._initialize()
        
        # Prepare metadatas
        if metadatas is None:
            metadatas = [{"subject": subject or "general"} for _ in documents]
        elif subject:
            for meta in metadatas:
                meta.setdefault("subject", subject)
        
        # Add to vectorstore
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._vectorstore.add_texts(documents, metadatas=metadatas)
        )
        
        logger.info(f"Added {len(documents)} documents to vectorstore")
    
    async def load_documents_from_directory(
        self,
        directory: Path,
        subject: Optional[str] = None,
        file_extensions: List[str] = [".txt", ".md", ".pdf"]
    ):
        """
        Load and index documents from a directory.
        
        Args:
            directory: Path to directory containing documents
            subject: Subject label for documents
            file_extensions: List of file extensions to process
        """
        if not self._initialized:
            await self._initialize()
        
        directory = Path(directory)
        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            return
        
        # Find all documents
        doc_files = []
        for ext in file_extensions:
            doc_files.extend(directory.rglob(f"*{ext}"))
        
        if not doc_files:
            logger.warning(f"No documents found in {directory}")
            return
        
        logger.info(f"Found {len(doc_files)} documents in {directory}")
        
        # Load and split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            length_function=len
        )
        
        all_chunks = []
        all_metadatas = []
        
        for doc_file in doc_files:
            try:
                # Read file
                with open(doc_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split into chunks
                chunks = text_splitter.split_text(content)
                
                # Create metadata
                metadata = {
                    "source": str(doc_file),
                    "subject": subject or doc_file.parent.name,
                    "filename": doc_file.name
                }
                
                all_chunks.extend(chunks)
                all_metadatas.extend([metadata.copy() for _ in chunks])
                
                logger.debug(f"Loaded {len(chunks)} chunks from {doc_file.name}")
                
            except Exception as e:
                logger.error(f"Error loading {doc_file}: {e}")
        
        # Add to vectorstore
        if all_chunks:
            await self.add_documents(all_chunks, all_metadatas)
            logger.success(f"Indexed {len(all_chunks)} chunks from {len(doc_files)} documents")
    
    async def cleanup(self):
        """Cleanup resources."""
        if self._vectorstore:
            del self._vectorstore
            self._vectorstore = None
        
        if self._embeddings:
            del self._embeddings
            self._embeddings = None
        
        self._initialized = False
        logger.info("RAGService cleaned up")


class AgenticRAGService(RAGService):
    """
    Advanced RAG service with subject routing and multi-index support.
    Routes queries to specialized subject-specific vectorstores.
    """
    
    def __init__(
        self,
        base_path: str = "./data",
        subjects: List[str] = ["maths", "physique", "anglais"],
        router_keywords: Optional[Dict[str, List[str]]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.base_path = Path(base_path)
        self.subjects = subjects
        self.router_keywords = router_keywords or {}
        
        # Multi-index setup
        self._subject_vectorstores: Dict[str, Any] = {}
        
        logger.info(f"AgenticRAGService initialized with subjects: {subjects}")
    
    async def _initialize(self):
        """Initialize multiple subject-specific vectorstores."""
        if self._initialized:
            return
        
        # Initialize embeddings
        await super()._initialize()
        
        # Load vectorstores for each subject
        for subject in self.subjects:
            subject_path = self.base_path / subject
            
            if subject_path.exists():
                try:
                    vectorstore = Chroma(
                        persist_directory=str(subject_path),
                        embedding_function=self._embeddings,
                        collection_name=subject
                    )
                    self._subject_vectorstores[subject] = vectorstore
                    logger.success(f"Loaded vectorstore for {subject}")
                except Exception as e:
                    logger.error(f"Failed to load vectorstore for {subject}: {e}")
        
        logger.info(f"Loaded {len(self._subject_vectorstores)} subject vectorstores")
    
    def _route_query(self, query: str) -> str:
        """
        Route query to appropriate subject based on keywords.
        
        Args:
            query: User's question
            
        Returns:
            Detected subject name
        """
        query_lower = query.lower()
        
        for subject, keywords in self.router_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                logger.debug(f"Routed to {subject}")
                return subject
        
        # Default to first subject or "general"
        default = self.subjects[0] if self.subjects else "general"
        logger.debug(f"No routing match, using default: {default}")
        return default
    
    async def _retrieve(self, query: str) -> Optional[str]:
        """Retrieve from routed subject-specific vectorstore."""
        subject = self._route_query(query)
        
        vectorstore = self._subject_vectorstores.get(subject)
        if not vectorstore:
            logger.warning(f"No vectorstore for subject: {subject}")
            return None
        
        # Use the subject-specific vectorstore
        old_vectorstore = self._vectorstore
        self._vectorstore = vectorstore
        
        result = await super()._retrieve(query)
        
        self._vectorstore = old_vectorstore
        return result
