"""
RAG Service using FAISS indexes (lightweight, no LangChain dependency).
Compatible with indexes built by scripts/build_indexes_colab.py
"""

import asyncio
import pickle
from typing import Optional, List, Dict, Any
from pathlib import Path

from pipecat.frames.frames import Frame, TextFrame, ErrorFrame
from pipecat.processors.frame_processor import FrameProcessor
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    logger.warning("sentence-transformers or faiss not available")


class RAGServiceFAISS(FrameProcessor):
    """
    RAG service using FAISS indexes (no LangChain).
    
    Features:
    - Multi-subject document retrieval (maths/physique/anglais)
    - FAISS vector search
    - Subject routing
    - Lightweight (no ChromaDB/LangChain)
    
    Args:
        data_path: Path to data folder containing subject folders with FAISS indexes
        embedding_model: HuggingFace embedding model name
        top_k: Number of documents to retrieve per subject
        score_threshold: Minimum similarity score (0.0-1.0)
    """
    
    def __init__(
        self,
        data_path: str = "./data",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 3,
        score_threshold: float = 0.3,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        if not DEPENDENCIES_AVAILABLE:
            raise RuntimeError("Dependencies missing. Run: pip install sentence-transformers faiss-cpu")
        
        self.data_path = Path(data_path)
        self.embedding_model_name = embedding_model
        self.top_k = top_k
        self.score_threshold = score_threshold
        
        self._embedding_model = None
        self._indexes = {}  # subject -> faiss index
        self._metadata = {}  # subject -> metadata list
        self._initialized = False
        
        logger.info(f"RAGServiceFAISS initialized (path: {data_path}, model: {embedding_model})")
    
    async def _initialize(self):
        """Initialize embedding model and load FAISS indexes."""
        if self._initialized:
            return
        
        try:
            logger.info("Loading embedding model...")
            
            # Load embedding model
            loop = asyncio.get_event_loop()
            self._embedding_model = await loop.run_in_executor(
                None,
                lambda: SentenceTransformer(self.embedding_model_name)
            )
            
            logger.info(f"Embedding model loaded: {self.embedding_model_name}")
            
            # Load FAISS indexes for each subject
            subjects = ['maths', 'physique', 'anglais']
            
            for subject in subjects:
                subject_dir = self.data_path / subject
                index_path = subject_dir / "index.faiss"
                metadata_path = subject_dir / "metadata.pkl"
                
                if not index_path.exists():
                    logger.warning(f"Index not found for {subject}: {index_path}")
                    continue
                
                # Load FAISS index
                index = faiss.read_index(str(index_path))
                self._indexes[subject] = index
                
                # Load metadata
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                self._metadata[subject] = metadata
                
                logger.info(f"Loaded {subject} index: {index.ntotal} vectors")
            
            if not self._indexes:
                raise RuntimeError("No FAISS indexes loaded")
            
            self._initialized = True
            logger.info(f"RAG initialized with {len(self._indexes)} subjects")
        
        except Exception as e:
            logger.error(f"Failed to initialize RAG: {e}")
            raise
    
    def _classify_subject(self, query: str) -> str:
        """
        Simple subject classification based on keywords.
        Returns: 'maths', 'physique', or 'anglais'
        """
        query_lower = query.lower()
        
        # Maths keywords
        math_keywords = [
            'équation', 'equation', 'degré', 'résoudre', 'calculer',
            'discriminant', 'delta', 'racine', 'solution', 'x²', 'x2',
            'polynôme', 'algèbre', 'formule', 'nombre', 'mathématique'
        ]
        
        # Physics keywords
        physics_keywords = [
            'newton', 'force', 'masse', 'accélération', 'vitesse',
            'énergie', 'mouvement', 'mécanique', 'loi', 'physique',
            'poids', 'gravit', 'inertie', 'action', 'réaction'
        ]
        
        # English keywords
        english_keywords = [
            'anglais', 'english', 'verbe', 'verb', 'conjugaison',
            'temps', 'tense', 'grammaire', 'grammar', 'present',
            'past', 'futur', 'to be', 'to have', 'présent'
        ]
        
        # Count matches
        math_score = sum(1 for kw in math_keywords if kw in query_lower)
        physics_score = sum(1 for kw in physics_keywords if kw in query_lower)
        english_score = sum(1 for kw in english_keywords if kw in query_lower)
        
        # Return subject with highest score
        scores = {
            'maths': math_score,
            'physique': physics_score,
            'anglais': english_score
        }
        
        subject = max(scores, key=scores.get)
        logger.info(f"Classified query as: {subject} (scores: {scores})")
        
        return subject
    
    async def retrieve(self, query: str, subject: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for query.
        
        Args:
            query: Search query
            subject: Specific subject to search in, or None for auto-detection
        
        Returns:
            List of retrieved documents with metadata
        """
        if not self._initialized:
            await self._initialize()
        
        # Auto-detect subject if not specified
        if subject is None:
            subject = self._classify_subject(query)
        
        if subject not in self._indexes:
            logger.warning(f"Subject {subject} not available")
            return []
        
        # Generate query embedding
        loop = asyncio.get_event_loop()
        query_embedding = await loop.run_in_executor(
            None,
            lambda: self._embedding_model.encode([query])
        )
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        index = self._indexes[subject]
        scores, indices = index.search(query_embedding, self.top_k)
        
        # Retrieve documents above threshold
        results = []
        metadata_list = self._metadata[subject]
        
        for score, idx in zip(scores[0], indices[0]):
            if score >= self.score_threshold and idx < len(metadata_list):
                doc = metadata_list[idx].copy()
                doc['score'] = float(score)
                doc['subject'] = subject
                results.append(doc)
        
        logger.info(f"Retrieved {len(results)} documents for {subject}")
        
        return results
    
    def format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents as context for LLM."""
        if not documents:
            return "Aucun document pertinent trouvé."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            text = doc.get('text', '')
            source = doc.get('source', 'unknown')
            score = doc.get('score', 0.0)
            
            context_parts.append(
                f"[Document {i}] (source: {source}, score: {score:.2f})\n{text}"
            )
        
        return "\n\n".join(context_parts)
    
    async def process_frame(self, frame: Frame, direction: str):
        """Process incoming frame - enrich with RAG context if text."""
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TextFrame):
            try:
                # Retrieve relevant documents
                query = frame.text
                documents = await self.retrieve(query)
                
                # Format context
                context = self.format_context(documents)
                
                # Create enriched prompt
                enriched_text = f"""Contexte pertinent :
{context}

Question : {query}

Réponds en français de manière pédagogique en utilisant le contexte ci-dessus."""
                
                # Push enriched frame
                await self.push_frame(TextFrame(enriched_text))
                
                logger.info(f"Enriched query with {len(documents)} documents")
            
            except Exception as e:
                logger.error(f"RAG error: {e}")
                # Push original frame on error
                await self.push_frame(frame)
        else:
            # Pass through non-text frames
            await self.push_frame(frame)


# Alias for backward compatibility
RAGService = RAGServiceFAISS
