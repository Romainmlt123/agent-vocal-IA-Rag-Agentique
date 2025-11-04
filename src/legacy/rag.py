"""
RAG Retrieval Module - Query and retrieve relevant passages.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from .config import get_config
from .utils import get_logger


logger = get_logger(__name__)


class RetrievalResult:
    """Represents a single retrieval result."""
    
    def __init__(self, text: str, source: str, page: Optional[int], 
                 score: float, subject: str):
        """
        Initialize retrieval result.
        
        Args:
            text: Retrieved text chunk
            source: Source document name
            page: Page number (if applicable)
            score: Similarity score
            subject: Subject category
        """
        self.text = text
        self.source = source
        self.page = page
        self.score = score
        self.subject = subject
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "score": float(self.score),
            "subject": self.subject
        }
    
    def __repr__(self) -> str:
        page_str = f", page {self.page}" if self.page else ""
        return f"RetrievalResult(source={self.source}{page_str}, score={self.score:.3f})"


class RAGRetriever:
    """Retrieves relevant passages from FAISS indexes."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize RAG retriever.
        
        Args:
            config_path: Path to config file
        """
        self.config = get_config(config_path)
        self.embedding_model = None
        self.indexes = {}  # subject -> faiss.Index
        self.metadata = {}  # subject -> List[chunk_dict]
        
        self._load_embedding_model()
        self._load_indexes()
    
    def _load_embedding_model(self):
        """Load the sentence transformer embedding model."""
        logger.info(f"Loading embedding model: {self.config.rag.embedding_model}")
        self.embedding_model = SentenceTransformer(
            self.config.rag.embedding_model,
            device=self.config.rag.embedding_device
        )
        logger.info("Embedding model loaded successfully")
    
    def _load_indexes(self):
        """Load all available FAISS indexes."""
        subjects = ["maths", "physique", "anglais"]
        
        for subject in subjects:
            try:
                self._load_subject_index(subject)
            except Exception as e:
                logger.warning(f"Could not load index for {subject}: {e}")
    
    def _load_subject_index(self, subject: str):
        """
        Load FAISS index and metadata for a subject.
        
        Args:
            subject: Subject name
        """
        index_path = self.config.get_index_path(subject)
        
        if not index_path.exists():
            logger.warning(f"Index file not found: {index_path}")
            return
        
        # Load FAISS index
        index = faiss.read_index(str(index_path))
        
        # Load metadata
        metadata_path = index_path.with_suffix('.pkl')
        if not metadata_path.exists():
            logger.warning(f"Metadata file not found: {metadata_path}")
            return
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.indexes[subject] = index
        self.metadata[subject] = metadata
        
        logger.info(f"Loaded index for {subject}: {index.ntotal} vectors")
    
    def retrieve(self, subject: str, query: str, k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Retrieve relevant passages for a query.
        
        Args:
            subject: Subject to search in (maths, physique, anglais)
            query: Query text
            k: Number of results to retrieve (default from config)
        
        Returns:
            List of RetrievalResult objects, sorted by relevance
        """
        if k is None:
            k = self.config.rag.top_k
        
        # Check if index is available
        if subject not in self.indexes:
            logger.warning(f"No index available for subject: {subject}")
            return []
        
        logger.debug(f"Retrieving from {subject} with query: {query[:100]}...")
        
        # Encode query
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        )
        
        # Search FAISS index
        index = self.indexes[subject]
        distances, indices = index.search(query_embedding, k)
        
        # Convert distances to similarity scores (lower distance = higher similarity)
        # For L2 distance, we can use negative distance or convert to similarity
        # Here we'll use a simple conversion: similarity = 1 / (1 + distance)
        similarities = 1.0 / (1.0 + distances[0])
        
        # Get metadata for results
        results = []
        metadata = self.metadata[subject]
        
        for idx, (doc_idx, score) in enumerate(zip(indices[0], similarities)):
            if doc_idx < 0 or doc_idx >= len(metadata):
                continue
            
            chunk_data = metadata[doc_idx]
            
            # Filter by similarity threshold
            if score < self.config.rag.similarity_threshold:
                continue
            
            result = RetrievalResult(
                text=chunk_data["text"],
                source=chunk_data["source"],
                page=chunk_data.get("page"),
                score=score,
                subject=subject
            )
            results.append(result)
        
        logger.debug(f"Retrieved {len(results)} passages (threshold: {self.config.rag.similarity_threshold})")
        return results
    
    def retrieve_multi_subject(self, query: str, subjects: Optional[List[str]] = None, 
                               k: Optional[int] = None) -> Dict[str, List[RetrievalResult]]:
        """
        Retrieve from multiple subjects.
        
        Args:
            query: Query text
            subjects: List of subjects to search (default: all available)
            k: Number of results per subject
        
        Returns:
            Dictionary mapping subject names to retrieval results
        """
        if subjects is None:
            subjects = list(self.indexes.keys())
        
        results = {}
        for subject in subjects:
            results[subject] = self.retrieve(subject, query, k)
        
        return results
    
    def format_context(self, results: List[RetrievalResult], max_length: Optional[int] = None) -> str:
        """
        Format retrieval results into a context string for LLM.
        
        Args:
            results: List of retrieval results
            max_length: Maximum context length in characters
        
        Returns:
            Formatted context string
        """
        if not results:
            return ""
        
        context_parts = []
        current_length = 0
        
        for i, result in enumerate(results, 1):
            page_info = f" (page {result.page})" if result.page else ""
            source_info = f"[Source {i}: {result.source}{page_info}]\n"
            passage = f"{result.text}\n\n"
            
            part = source_info + passage
            
            if max_length and current_length + len(part) > max_length:
                break
            
            context_parts.append(part)
            current_length += len(part)
        
        return "".join(context_parts)
    
    def is_available(self, subject: str) -> bool:
        """
        Check if a subject index is available.
        
        Args:
            subject: Subject name
        
        Returns:
            True if index is loaded and available
        """
        return subject in self.indexes
    
    def get_available_subjects(self) -> List[str]:
        """
        Get list of available subjects.
        
        Returns:
            List of subject names with loaded indexes
        """
        return list(self.indexes.keys())


# Singleton instance for easy access
_retriever_instance: Optional[RAGRetriever] = None


def get_retriever(config_path: Optional[str] = None) -> RAGRetriever:
    """
    Get or create the global RAG retriever instance.
    
    Args:
        config_path: Path to config file
    
    Returns:
        RAGRetriever instance
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = RAGRetriever(config_path)
    return _retriever_instance
