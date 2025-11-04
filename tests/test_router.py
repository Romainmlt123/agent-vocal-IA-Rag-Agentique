"""
Tests for Router module.
"""

import pytest

from src.router import SubjectRouter, ModelSpec


def test_router_initialization():
    """Test router initialization."""
    router = SubjectRouter()
    
    assert router.config is not None
    assert router.keywords is not None
    assert len(router.keywords) > 0


def test_detect_subject_keywords():
    """Test keyword-based subject detection."""
    router = SubjectRouter()
    
    # Math question
    math_query = "Comment résoudre une équation du second degré?"
    subject = router.detect_subject_keywords(math_query)
    assert subject == "maths"
    
    # Physics question
    physics_query = "Quelle est la formule de l'énergie cinétique?"
    subject = router.detect_subject_keywords(physics_query)
    assert subject == "physique"
    
    # English question
    english_query = "How do you conjugate irregular verbs in English?"
    subject = router.detect_subject_keywords(english_query)
    assert subject == "anglais"
    
    # Unknown subject
    unknown_query = "What is the capital of France?"
    subject = router.detect_subject_keywords(unknown_query)
    # May return None or a subject depending on keywords
    assert subject is None or isinstance(subject, str)


def test_detect_subject():
    """Test complete subject detection."""
    router = SubjectRouter()
    
    # Math
    query = "Aide-moi à calculer une dérivée"
    subject = router.detect_subject(query)
    assert subject in ["maths", "default"]
    
    # Physics
    query = "Explique-moi la loi de Newton"
    subject = router.detect_subject(query)
    assert subject in ["physique", "default"]


def test_pick_model():
    """Test model selection."""
    router = SubjectRouter()
    
    query = "Comment résoudre une intégrale?"
    model_spec = router.pick_model(query)
    
    assert isinstance(model_spec, ModelSpec)
    assert model_spec.subject in ["maths", "physique", "anglais", "default"]
    assert model_spec.model_path is not None
    assert 0 <= model_spec.confidence <= 1.0


def test_get_all_subjects():
    """Test getting all subjects."""
    router = SubjectRouter()
    
    subjects = router.get_all_subjects()
    
    assert isinstance(subjects, list)
    assert len(subjects) > 0
    assert "maths" in subjects
    assert "physique" in subjects
    assert "anglais" in subjects


def test_get_subject_keywords():
    """Test getting keywords for a subject."""
    router = SubjectRouter()
    
    keywords = router.get_subject_keywords("maths")
    
    assert isinstance(keywords, list)
    assert len(keywords) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
