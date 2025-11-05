"""
Voice Pipeline pour Agent IA avec Pipecat Framework
Optimisé pour Google Colab et interface Gradio
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.aggregators.llm_response import LLMAssistantResponseAggregator, LLMUserResponseAggregator
from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    TextFrame,
    TranscriptionFrame,
    EndFrame,
    StartFrame,
    LLMMessagesFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection

# Import local services
import sys
sys.path.append(str(Path(__file__).parent.parent))
from services.local_stt import LocalSTTService
from services.local_llm import LocalLLMService
from services.local_tts import LocalTTSService
from services.rag_service import RAGService

logger = logging.getLogger(__name__)


class AudioBufferProcessor(FrameProcessor):
    """Processeur pour collecter l'audio de sortie"""
    
    def __init__(self):
        super().__init__()
        self.audio_buffer = []
        self.sample_rate = 22050
        self.is_collecting = False
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TTSStartedFrame):
            self.is_collecting = True
            self.audio_buffer = []
            logger.debug("Started collecting audio")
            
        elif isinstance(frame, TTSAudioRawFrame):
            if self.is_collecting:
                self.audio_buffer.append(frame.audio)
                logger.debug(f"Collected {len(frame.audio)} bytes of audio")
                
        elif isinstance(frame, TTSStoppedFrame):
            self.is_collecting = False
            logger.debug(f"Stopped collecting audio, total buffers: {len(self.audio_buffer)}")
        
        await self.push_frame(frame, direction)
    
    def get_audio_bytes(self) -> bytes:
        """Récupère l'audio collecté"""
        if not self.audio_buffer:
            return b''
        return b''.join(self.audio_buffer)
    
    def clear_buffer(self):
        """Vide le buffer audio"""
        self.audio_buffer = []


class TranscriptionCollector(FrameProcessor):
    """Collecteur pour la transcription STT"""
    
    def __init__(self):
        super().__init__()
        self.transcription = ""
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TranscriptionFrame):
            self.transcription = frame.text
            logger.info(f"Transcription: {self.transcription}")
        
        await self.push_frame(frame, direction)
    
    def get_transcription(self) -> str:
        """Récupère la dernière transcription"""
        return self.transcription
    
    def clear(self):
        """Efface la transcription"""
        self.transcription = ""


class ResponseCollector(FrameProcessor):
    """Collecteur pour la réponse du LLM"""
    
    def __init__(self):
        super().__init__()
        self.response = ""
        self.is_collecting = False
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TextFrame) and not isinstance(frame, TranscriptionFrame):
            # C'est une réponse du LLM ou du RAG
            if frame.text:
                self.response += frame.text
                logger.debug(f"Response chunk: {frame.text}")
        
        await self.push_frame(frame, direction)
    
    def get_response(self) -> str:
        """Récupère la réponse complète"""
        return self.response
    
    def clear(self):
        """Efface la réponse"""
        self.response = ""


class VoicePipeline:
    """
    Pipeline vocal complet pour Google Colab
    Gère : Audio Input → STT → RAG → LLM → TTS → Audio Output
    """
    
    def __init__(
        self,
        whisper_model: str = "base",
        ollama_model: str = "qwen2:1.5b",
        device: str = "cuda",
        rag_data_path: str = "data"
    ):
        """
        Initialize Voice Pipeline
        
        Args:
            whisper_model: Taille du modèle Whisper (tiny, base, small, medium, large)
            ollama_model: Modèle Ollama à utiliser
            device: Device pour les modèles (cuda, cpu)
            rag_data_path: Chemin vers les données RAG
        """
        self.whisper_model = whisper_model
        self.ollama_model = ollama_model
        self.device = device
        self.rag_data_path = rag_data_path
        
        # Services
        self.stt_service: Optional[LocalSTTService] = None
        self.llm_service: Optional[LocalLLMService] = None
        self.tts_service: Optional[LocalTTSService] = None
        self.rag_service: Optional[RAGService] = None
        
        # Collectors
        self.transcription_collector = TranscriptionCollector()
        self.response_collector = ResponseCollector()
        self.audio_buffer = AudioBufferProcessor()
        
        # Pipeline components
        self.pipeline: Optional[Pipeline] = None
        self.task: Optional[PipelineTask] = None
        self.runner: Optional[PipelineRunner] = None
        
        logger.info(f"VoicePipeline initialized with Whisper={whisper_model}, Ollama={ollama_model}")
    
    async def initialize_services(self):
        """Initialize all services"""
        logger.info("Initializing services...")
        
        # STT Service
        self.stt_service = LocalSTTService(
            model_size=self.whisper_model,
            device=self.device,
            language="fr"
        )
        logger.info(f"✓ STT Service initialized (Whisper {self.whisper_model})")
        
        # RAG Service
        self.rag_service = RAGService(
            data_path=self.rag_data_path,
            device=self.device
        )
        logger.info(f"✓ RAG Service initialized")
        
        # LLM Service
        self.llm_service = LocalLLMService(
            model=self.ollama_model,
            temperature=0.7,
            max_tokens=512
        )
        logger.info(f"✓ LLM Service initialized (Ollama {self.ollama_model})")
        
        # TTS Service
        self.tts_service = LocalTTSService(
            model_name="fr_FR-siwis-medium",
            device=self.device
        )
        logger.info(f"✓ TTS Service initialized (Piper)")
        
        logger.info("✅ All services initialized successfully")
    
    def build_pipeline(self):
        """Build the Pipecat pipeline"""
        logger.info("Building pipeline...")
        
        if not all([self.stt_service, self.rag_service, self.llm_service, self.tts_service]):
            raise RuntimeError("Services not initialized. Call initialize_services() first.")
        
        # Create pipeline with all processors in order
        self.pipeline = Pipeline([
            self.stt_service,              # Audio → Text transcription
            self.transcription_collector,  # Collect transcription
            self.rag_service,              # Add RAG context
            self.llm_service,              # Generate response
            self.response_collector,       # Collect response
            self.tts_service,              # Text → Audio synthesis
            self.audio_buffer              # Collect output audio
        ])
        
        # Create task
        self.task = PipelineTask(
            self.pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True
            )
        )
        
        # Create runner
        self.runner = PipelineRunner(handle_sigint=False)
        
        logger.info("✅ Pipeline built successfully")
    
    async def process_audio(self, audio_bytes: bytes, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Process audio input and return results
        
        Args:
            audio_bytes: Raw audio bytes (PCM)
            sample_rate: Sample rate of input audio
            
        Returns:
            Dict with transcription, response, subject, and audio output
        """
        if not self.pipeline or not self.task:
            raise RuntimeError("Pipeline not built. Call build_pipeline() first.")
        
        logger.info(f"Processing audio ({len(audio_bytes)} bytes, {sample_rate}Hz)")
        
        # Clear collectors
        self.transcription_collector.clear()
        self.response_collector.clear()
        self.audio_buffer.clear_buffer()
        
        try:
            # Create audio frame
            audio_frame = AudioRawFrame(
                audio=audio_bytes,
                sample_rate=sample_rate,
                num_channels=1
            )
            
            # Queue frames
            await self.task.queue_frames([StartFrame(), audio_frame, EndFrame()])
            
            # Run pipeline (with timeout)
            try:
                await asyncio.wait_for(self.runner.run(self.task), timeout=30.0)
            except asyncio.TimeoutError:
                logger.error("Pipeline execution timeout")
                raise
            
            # Collect results
            transcription = self.transcription_collector.get_transcription()
            response_text = self.response_collector.get_response()
            output_audio = self.audio_buffer.get_audio_bytes()
            
            # Get subject from RAG service
            subject = getattr(self.rag_service, 'last_detected_subject', 'unknown')
            
            results = {
                'transcription': transcription,
                'response': response_text,
                'subject': subject,
                'audio_output': output_audio,
                'sample_rate': self.audio_buffer.sample_rate
            }
            
            logger.info(f"✅ Processing complete - Transcription: '{transcription[:50]}...', Subject: {subject}")
            return results
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            raise
    
    async def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process text input directly (without STT)
        
        Args:
            text: Input text question
            
        Returns:
            Dict with response, subject, and audio output
        """
        if not self.pipeline or not self.task:
            raise RuntimeError("Pipeline not built. Call build_pipeline() first.")
        
        logger.info(f"Processing text: '{text}'")
        
        # Clear collectors
        self.response_collector.clear()
        self.audio_buffer.clear_buffer()
        
        try:
            # Create text frame (simulate transcription)
            text_frame = TranscriptionFrame(text=text, user_id="user")
            
            # Queue frames
            await self.task.queue_frames([StartFrame(), text_frame, EndFrame()])
            
            # Run pipeline
            try:
                await asyncio.wait_for(self.runner.run(self.task), timeout=30.0)
            except asyncio.TimeoutError:
                logger.error("Pipeline execution timeout")
                raise
            
            # Collect results
            response_text = self.response_collector.get_response()
            output_audio = self.audio_buffer.get_audio_bytes()
            
            # Get subject from RAG service
            subject = getattr(self.rag_service, 'last_detected_subject', 'unknown')
            
            results = {
                'transcription': text,
                'response': response_text,
                'subject': subject,
                'audio_output': output_audio,
                'sample_rate': self.audio_buffer.sample_rate
            }
            
            logger.info(f"✅ Text processing complete - Subject: {subject}")
            return results
            
        except Exception as e:
            logger.error(f"Error processing text: {e}", exc_info=True)
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up pipeline...")
        
        if self.task:
            try:
                await self.task.cancel()
            except:
                pass
        
        # Cleanup services
        if self.stt_service:
            # Add cleanup if needed
            pass
        
        logger.info("✅ Pipeline cleaned up")


# Factory function for easy instantiation
async def create_voice_pipeline(
    whisper_model: str = "base",
    ollama_model: str = "qwen2:1.5b",
    device: str = "cuda",
    rag_data_path: str = "data"
) -> VoicePipeline:
    """
    Create and initialize a voice pipeline
    
    Args:
        whisper_model: Whisper model size
        ollama_model: Ollama model name
        device: Device (cuda/cpu)
        rag_data_path: Path to RAG data
        
    Returns:
        Initialized VoicePipeline
    """
    pipeline = VoicePipeline(
        whisper_model=whisper_model,
        ollama_model=ollama_model,
        device=device,
        rag_data_path=rag_data_path
    )
    
    await pipeline.initialize_services()
    pipeline.build_pipeline()
    
    return pipeline


# Example usage
if __name__ == "__main__":
    import numpy as np
    
    async def test_pipeline():
        """Test the pipeline with dummy data"""
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
        # Create pipeline
        pipeline = await create_voice_pipeline(
            whisper_model="tiny",  # Use tiny for testing
            ollama_model="qwen2:1.5b",
            device="cuda"
        )
        
        # Test with text
        print("\n=== Testing with text input ===")
        result = await pipeline.process_text("Comment résoudre une équation du second degré ?")
        print(f"Transcription: {result['transcription']}")
        print(f"Subject: {result['subject']}")
        print(f"Response: {result['response'][:100]}...")
        print(f"Audio output: {len(result['audio_output'])} bytes")
        
        # Cleanup
        await pipeline.cleanup()
        print("\n✅ Test completed")
    
    # Run test
    asyncio.run(test_pipeline())
