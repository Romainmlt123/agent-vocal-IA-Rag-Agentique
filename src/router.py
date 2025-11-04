"""
Router Module - Subject Detection and Model Routing.
Uses keyword matching and TF-IDF for intelligent routing to specialized models.
"""

from typing import Optional, Dict, List
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from .config import get_config
from .utils import get_logger


logger = get_logger(__name__)


@dataclass
class ModelSpec:
    """Specification for a subject's model."""
    subject: str
    model_path: str
    confidence: float = 0.0


class SubjectRouter:
    """Routes queries to appropriate subject-specific models."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize router.
        
        Args:
            config_path: Path to config file
        """
        self.config = get_config(config_path)
        self.keywords = self.config.router.keywords
        self.tfidf_vectorizer = None
        self.subject_vectors = {}
        
        if self.config.router.use_tfidf_fallback:
            self._build_tfidf()
    
    def _build_tfidf(self):
        """Build TF-IDF vectorizer from keywords."""
        logger.info("Building TF-IDF vectorizer for fallback routing")
        
        # Create documents from keywords
        documents = []
        subjects = []
        
        for subject, keywords in self.keywords.items():
            # Create a document by joining all keywords
            doc = " ".join(keywords)
            documents.append(doc)
            subjects.append(subject)
        
        # Build TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(lowercase=True)
        vectors = self.tfidf_vectorizer.fit_transform(documents)
        
        # Store subject vectors
        for subject, vector in zip(subjects, vectors):
            self.subject_vectors[subject] = vector
        
        logger.info(f"TF-IDF vectorizer built with {len(subjects)} subjects")
    
    def detect_subject_keywords(self, query: str) -> Optional[str]:
        """
        Detect subject using keyword matching.
        
        Args:
            query: Query text
        
        Returns:
            Subject name if detected, None otherwise
        """
        query_lower = query.lower()
        
        # Count keyword matches for each subject
        matches = {}
        for subject, keywords in self.keywords.items():
            count = sum(1 for keyword in keywords if keyword.lower() in query_lower)
            if count > 0:
                matches[subject] = count
        
        if not matches:
            return None
        
        # Return subject with most matches
        best_subject = max(matches.items(), key=lambda x: x[1])[0]
        logger.debug(f"Keyword detection: {best_subject} ({matches[best_subject]} matches)")
        
        return best_subject
    
    def detect_subject_tfidf(self, query: str) -> Optional[str]:
        """
        Detect subject using TF-IDF similarity.
        
        Args:
            query: Query text
        
        Returns:
            Subject name if detected, None otherwise
        """
        if self.tfidf_vectorizer is None:
            return None
        
        # Vectorize query
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Calculate similarities
        similarities = {}
        for subject, subject_vector in self.subject_vectors.items():
            # Cosine similarity
            similarity = (query_vector * subject_vector.T).toarray()[0, 0]
            similarities[subject] = similarity
        
        if not similarities:
            return None
        
        # Get best match
        best_subject = max(similarities.items(), key=lambda x: x[1])[0]
        best_score = similarities[best_subject]
        
        logger.debug(f"TF-IDF detection: {best_subject} (score: {best_score:.3f})")
        
        # Only return if confidence is reasonable
        if best_score > 0.1:
            return best_subject
        
        return None
    
    def detect_subject(self, query: str) -> str:
        """
        Detect subject from query text.
        
        Args:
            query: Query text
        
        Returns:
            Subject name (maths, physique, anglais) or 'default'
        """
        logger.debug(f"Detecting subject for query: {query[:100]}...")
        
        # Try keyword detection first
        subject = self.detect_subject_keywords(query)
        
        # Fallback to TF-IDF if enabled
        if subject is None and self.config.router.use_tfidf_fallback:
            subject = self.detect_subject_tfidf(query)
        
        # Default fallback
        if subject is None:
            logger.debug("No subject detected, using default")
            subject = "default"
        
        logger.info(f"Routed to subject: {subject}")
        return subject
    
    def pick_model(self, query: str, subject_hint: Optional[str] = None) -> ModelSpec:
        """
        Pick the appropriate model for a query.
        
        Args:
            query: Query text
            subject_hint: Optional subject hint to override detection
        
        Returns:
            ModelSpec with subject and model path
        """
        # Use hint if provided, otherwise detect
        if subject_hint:
            subject = subject_hint
            confidence = 1.0
        else:
            subject = self.detect_subject(query)
            
            # Calculate confidence (simple heuristic)
            if self.detect_subject_keywords(query):
                confidence = 0.8
            else:
                confidence = 0.5
        
        # Get model path
        model_path = str(self.config.get_model_path(subject))
        
        return ModelSpec(
            subject=subject,
            model_path=model_path,
            confidence=confidence
        )
    
    def get_all_subjects(self) -> List[str]:
        """
        Get list of all available subjects.
        
        Returns:
            List of subject names
        """
        return list(self.keywords.keys())
    
    def get_subject_keywords(self, subject: str) -> List[str]:
        """
        Get keywords for a subject.
        
        Args:
            subject: Subject name
        
        Returns:
            List of keywords
        """
        return self.keywords.get(subject, [])


# Singleton instance
_router_instance: Optional[SubjectRouter] = None


def get_router(config_path: Optional[str] = None) -> SubjectRouter:
    """
    Get or create the global router instance.
    
    Args:
        config_path: Path to config file
    
    Returns:
        SubjectRouter instance
    """
    global _router_instance
    if _router_instance is None:
        _router_instance = SubjectRouter(config_path)
    return _router_instance
