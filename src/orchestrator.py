"""
Orchestrator - Coordinates the complete tutoring pipeline.
Manages state, error handling, and event flow through ASR→RAG→LLM→TTS.
"""

import time
from typing import Dict, Any, Optional, Iterator, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .config import get_config
from .utils import get_logger
from .asr import get_asr_engine
from .rag import get_retriever
from .router import get_router
from .llm import get_llm_engine, HintLadder
from .tts import get_tts_engine


logger = get_logger(__name__)


class SessionState(Enum):
    """Session states."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    ERROR = "error"
    FINISHED = "finished"


@dataclass
class TutoringEvent:
    """Represents an event in the tutoring pipeline."""
    type: str  # transcript, rag_results, hints, audio, error, state_change
    data: Any
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp
        }


@dataclass
class Session:
    """Represents a tutoring session."""
    session_id: str
    state: SessionState = SessionState.IDLE
    start_time: float = field(default_factory=time.time)
    transcript: str = ""
    subject: Optional[str] = None
    rag_results: List = field(default_factory=list)
    hints: Optional[HintLadder] = None
    events: List[TutoringEvent] = field(default_factory=list)
    
    def add_event(self, event_type: str, data: Any):
        """Add an event to the session."""
        event = TutoringEvent(type=event_type, data=data)
        self.events.append(event)
        return event
    
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "elapsed_time": self.elapsed_time(),
            "transcript": self.transcript,
            "subject": self.subject,
            "has_hints": self.hints is not None,
            "num_events": len(self.events)
        }


class TutoringOrchestrator:
    """Orchestrates the complete tutoring pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize orchestrator.
        
        Args:
            config_path: Path to config file
        """
        self.config = get_config(config_path)
        
        # Initialize components
        logger.info("Initializing tutoring orchestrator components...")
        self.asr = get_asr_engine(config_path)
        self.retriever = get_retriever(config_path)
        self.router = get_router(config_path)
        self.llm = get_llm_engine(config_path)
        self.tts = get_tts_engine(config_path)
        
        self.sessions: Dict[str, Session] = {}
        
        logger.info("Orchestrator initialized successfully")
    
    def create_session(self, session_id: Optional[str] = None) -> Session:
        """
        Create a new tutoring session.
        
        Args:
            session_id: Optional session ID (generated if not provided)
        
        Returns:
            Session object
        """
        if session_id is None:
            session_id = f"session_{int(time.time() * 1000)}"
        
        session = Session(session_id=session_id)
        self.sessions[session_id] = session
        
        logger.info(f"Created session: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        return self.sessions.get(session_id)
    
    def process_audio(self, session: Session, audio_data, 
                     language: Optional[str] = None) -> Iterator[TutoringEvent]:
        """
        Process audio through the complete pipeline.
        
        Args:
            session: Session object
            audio_data: Audio data (numpy array or file path)
            language: Language code (optional)
        
        Yields:
            TutoringEvent objects
        """
        try:
            session.state = SessionState.LISTENING
            yield session.add_event("state_change", SessionState.LISTENING.value)
            
            # Step 1: ASR - Transcribe audio
            logger.info(f"[{session.session_id}] Starting ASR...")
            transcript = self.asr.transcribe_audio(audio_data, language)
            
            if not transcript:
                logger.warning(f"[{session.session_id}] Empty transcript")
                yield session.add_event("error", "No speech detected")
                session.state = SessionState.ERROR
                return
            
            session.transcript = transcript
            logger.info(f"[{session.session_id}] Transcript: {transcript}")
            yield session.add_event("transcript", transcript)
            
            # Step 2: Router - Detect subject
            session.state = SessionState.PROCESSING
            yield session.add_event("state_change", SessionState.PROCESSING.value)
            
            logger.info(f"[{session.session_id}] Detecting subject...")
            model_spec = self.router.pick_model(transcript)
            session.subject = model_spec.subject
            
            logger.info(f"[{session.session_id}] Subject: {session.subject}")
            yield session.add_event("subject_detected", {
                "subject": session.subject,
                "confidence": model_spec.confidence
            })
            
            # Step 3: RAG - Retrieve relevant context
            logger.info(f"[{session.session_id}] Retrieving RAG context...")
            rag_results = self.retriever.retrieve(session.subject, transcript)
            session.rag_results = rag_results
            
            logger.info(f"[{session.session_id}] Retrieved {len(rag_results)} passages")
            yield session.add_event("rag_results", [r.to_dict() for r in rag_results])
            
            # Format context for LLM
            context = self.retriever.format_context(rag_results, max_length=2000)
            
            # Step 4: LLM - Generate hints
            logger.info(f"[{session.session_id}] Generating hints...")
            hints = self.llm.generate_tutoring_response(
                transcript, 
                context, 
                session.subject
            )
            session.hints = hints
            
            logger.info(f"[{session.session_id}] Hints generated")
            yield session.add_event("hints", hints.to_dict())
            
            # Step 5: TTS - Synthesize response
            session.state = SessionState.RESPONDING
            yield session.add_event("state_change", SessionState.RESPONDING.value)
            
            # Detect language for TTS
            tts_language = self.tts.detect_language(transcript)
            logger.info(f"[{session.session_id}] TTS language: {tts_language}")
            
            # Synthesize each hint level
            for level_num, hint_text in enumerate([hints.level1, hints.level2, hints.level3], 1):
                if hint_text:
                    logger.info(f"[{session.session_id}] Synthesizing hint level {level_num}...")
                    audio_data = self.tts.synthesize(hint_text, tts_language)
                    
                    if audio_data:
                        yield session.add_event("audio", {
                            "level": level_num,
                            "audio_data": audio_data
                        })
            
            # Mark as finished
            session.state = SessionState.FINISHED
            yield session.add_event("state_change", SessionState.FINISHED.value)
            
            logger.info(f"[{session.session_id}] Session complete")
        
        except Exception as e:
            logger.error(f"[{session.session_id}] Pipeline error: {e}", exc_info=True)
            session.state = SessionState.ERROR
            yield session.add_event("error", str(e))
    
    def process_text_query(self, session: Session, query: str) -> Iterator[TutoringEvent]:
        """
        Process a text query (bypass ASR).
        
        Args:
            session: Session object
            query: Text query
        
        Yields:
            TutoringEvent objects
        """
        try:
            session.state = SessionState.PROCESSING
            session.transcript = query
            
            yield session.add_event("transcript", query)
            yield session.add_event("state_change", SessionState.PROCESSING.value)
            
            # Detect subject
            model_spec = self.router.pick_model(query)
            session.subject = model_spec.subject
            
            yield session.add_event("subject_detected", {
                "subject": session.subject,
                "confidence": model_spec.confidence
            })
            
            # Retrieve context
            rag_results = self.retriever.retrieve(session.subject, query)
            session.rag_results = rag_results
            
            yield session.add_event("rag_results", [r.to_dict() for r in rag_results])
            
            context = self.retriever.format_context(rag_results, max_length=2000)
            
            # Generate hints
            hints = self.llm.generate_tutoring_response(query, context, session.subject)
            session.hints = hints
            
            yield session.add_event("hints", hints.to_dict())
            
            # TTS
            session.state = SessionState.RESPONDING
            yield session.add_event("state_change", SessionState.RESPONDING.value)
            
            tts_language = self.tts.detect_language(query)
            
            for level_num, hint_text in enumerate([hints.level1, hints.level2, hints.level3], 1):
                if hint_text:
                    audio_data = self.tts.synthesize(hint_text, tts_language)
                    if audio_data:
                        yield session.add_event("audio", {
                            "level": level_num,
                            "audio_data": audio_data
                        })
            
            session.state = SessionState.FINISHED
            yield session.add_event("state_change", SessionState.FINISHED.value)
        
        except Exception as e:
            logger.error(f"[{session.session_id}] Error: {e}", exc_info=True)
            session.state = SessionState.ERROR
            yield session.add_event("error", str(e))
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of a session.
        
        Args:
            session_id: Session ID
        
        Returns:
            Session summary dictionary
        """
        session = self.get_session(session_id)
        if session is None:
            return None
        
        return {
            **session.to_dict(),
            "rag_results": [r.to_dict() for r in session.rag_results],
            "hints": session.hints.to_dict() if session.hints else None
        }


# Singleton instance
_orchestrator_instance: Optional[TutoringOrchestrator] = None


def get_orchestrator(config_path: Optional[str] = None) -> TutoringOrchestrator:
    """
    Get or create the global orchestrator instance.
    
    Args:
        config_path: Path to config file
    
    Returns:
        TutoringOrchestrator instance
    """
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = TutoringOrchestrator(config_path)
    return _orchestrator_instance
