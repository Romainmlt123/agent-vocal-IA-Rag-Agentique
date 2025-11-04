"""
Tests for Orchestrator module.
"""

import pytest

from src.orchestrator import TutoringOrchestrator, Session, SessionState, TutoringEvent


def test_tutoring_event_creation():
    """Test TutoringEvent creation."""
    event = TutoringEvent(type="transcript", data="Test transcript")
    
    assert event.type == "transcript"
    assert event.data == "Test transcript"
    assert event.timestamp > 0
    
    event_dict = event.to_dict()
    assert "type" in event_dict
    assert "data" in event_dict
    assert "timestamp" in event_dict


def test_session_creation():
    """Test Session creation."""
    session = Session(session_id="test_123")
    
    assert session.session_id == "test_123"
    assert session.state == SessionState.IDLE
    assert session.transcript == ""
    assert session.subject is None


def test_session_add_event():
    """Test adding events to session."""
    session = Session(session_id="test_123")
    
    event = session.add_event("test", {"key": "value"})
    
    assert len(session.events) == 1
    assert event.type == "test"
    assert event.data == {"key": "value"}


def test_session_elapsed_time():
    """Test session elapsed time."""
    import time
    
    session = Session(session_id="test_123")
    time.sleep(0.1)
    
    elapsed = session.elapsed_time()
    assert elapsed >= 0.1


def test_session_to_dict():
    """Test session serialization."""
    session = Session(session_id="test_123")
    session.subject = "maths"
    session.transcript = "Test question"
    
    session_dict = session.to_dict()
    
    assert session_dict["session_id"] == "test_123"
    assert session_dict["subject"] == "maths"
    assert session_dict["transcript"] == "Test question"


def test_orchestrator_initialization():
    """Test orchestrator initialization."""
    orchestrator = TutoringOrchestrator()
    
    assert orchestrator.config is not None
    assert orchestrator.asr is not None
    assert orchestrator.retriever is not None
    assert orchestrator.router is not None
    assert orchestrator.llm is not None
    assert orchestrator.tts is not None


def test_create_session():
    """Test session creation."""
    orchestrator = TutoringOrchestrator()
    
    session = orchestrator.create_session()
    
    assert session is not None
    assert session.session_id is not None
    assert session.state == SessionState.IDLE


def test_get_session():
    """Test getting session by ID."""
    orchestrator = TutoringOrchestrator()
    
    session = orchestrator.create_session("test_456")
    retrieved = orchestrator.get_session("test_456")
    
    assert retrieved is not None
    assert retrieved.session_id == "test_456"


def test_get_session_summary():
    """Test getting session summary."""
    orchestrator = TutoringOrchestrator()
    
    session = orchestrator.create_session("test_789")
    summary = orchestrator.get_session_summary("test_789")
    
    assert summary is not None
    assert "session_id" in summary
    assert summary["session_id"] == "test_789"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
