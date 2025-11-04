"""
Tests for RAG module.
"""

import pytest
import tempfile
from pathlib import Path

from src.rag_build import RAGBuilder, Document, DocumentChunk
from src.rag import RAGRetriever
from src.utils import chunk_text


def test_document_creation():
    """Test Document creation."""
    doc = Document(
        text="Test document content",
        source="test.pdf",
        page=1,
        subject="maths"
    )
    
    assert doc.text == "Test document content"
    assert doc.source == "test.pdf"
    assert doc.page == 1
    assert doc.subject == "maths"
    
    doc_dict = doc.to_dict()
    assert doc_dict["text"] == "Test document content"


def test_document_chunk_creation():
    """Test DocumentChunk creation."""
    chunk = DocumentChunk(
        text="Chunk content",
        doc_source="test.pdf",
        page=1,
        chunk_idx=0,
        subject="physique"
    )
    
    assert chunk.text == "Chunk content"
    assert chunk.doc_source == "test.pdf"
    assert chunk.chunk_idx == 0


def test_text_chunking():
    """Test text chunking utility."""
    text = "This is a test. " * 100  # Long text
    
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    
    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)  # Allow some variance


def test_rag_builder_initialization():
    """Test RAG builder initialization."""
    builder = RAGBuilder()
    
    assert builder.config is not None
    assert builder.embedding_model is not None


def test_rag_retriever_initialization():
    """Test RAG retriever initialization."""
    retriever = RAGRetriever()
    
    assert retriever.config is not None
    assert retriever.embedding_model is not None


def test_retrieval_result():
    """Test retrieval result creation."""
    from src.rag import RetrievalResult
    
    result = RetrievalResult(
        text="Sample text",
        source="test.pdf",
        page=1,
        score=0.95,
        subject="maths"
    )
    
    assert result.text == "Sample text"
    assert result.score == 0.95
    
    result_dict = result.to_dict()
    assert "text" in result_dict
    assert "score" in result_dict


def test_rag_builder_load_text_file():
    """Test loading text files."""
    builder = RAGBuilder()
    
    # Create temporary text file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is test content for RAG.\nSecond line.")
        temp_path = Path(f.name)
    
    try:
        # Create a directory structure
        temp_dir = temp_path.parent / "test_subject"
        temp_dir.mkdir(exist_ok=True)
        final_path = temp_dir / temp_path.name
        temp_path.rename(final_path)
        
        doc = builder.load_text_file(final_path)
        
        assert doc.text == "This is test content for RAG.\nSecond line."
        assert doc.source == final_path.name
        assert doc.subject == "test_subject"
    
    finally:
        if final_path.exists():
            final_path.unlink()
        if temp_dir.exists():
            temp_dir.rmdir()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
