"""
Local Speech-to-Text Service for Pipecat using Whisper.
Provides fast, accurate speech recognition with multiple model sizes.
"""

import asyncio
import numpy as np
from typing import Optional, Literal
from pathlib import Path

from pipecat.frames.frames import Frame, AudioRawFrame, TextFrame, ErrorFrame
from pipecat.processors.frame_processor import FrameProcessor
from loguru import logger

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    logger.warning("faster-whisper not available. Install with: pip install faster-whisper")


class LocalSTTService(FrameProcessor):
    """
    Local STT service using Faster-Whisper for real-time speech recognition.
    
    Features:
    - Multiple model sizes (tiny, base, small, medium, large)
    - GPU acceleration (CUDA)
    - Multilingual support
    - Word-level timestamps
    - No API calls required
    
    Args:
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        device: Computation device ('cpu', 'cuda', 'auto')
        compute_type: Precision type ('float16', 'float32', 'int8')
        language: Target language ('fr', 'en', None for auto-detection)
        vad_filter: Enable voice activity detection
        beam_size: Beam search size (higher = more accurate but slower)
    """
    
    def __init__(
        self,
        model_size: Literal["tiny", "base", "small", "medium", "large"] = "base",
        device: Literal["cpu", "cuda", "auto"] = "auto",
        compute_type: str = "float16",
        language: Optional[str] = "fr",
        vad_filter: bool = True,
        beam_size: int = 5,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        if not FASTER_WHISPER_AVAILABLE:
            raise RuntimeError("faster-whisper not installed. Run: pip install faster-whisper")
        
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.vad_filter = vad_filter
        self.beam_size = beam_size
        
        self._model: Optional[WhisperModel] = None
        self._initialized = False
        self._audio_buffer = bytearray()
        self._sample_rate = 16000  # Whisper expects 16kHz
        
        logger.info(f"LocalSTTService initialized (model: {model_size}, device: {device})")
    
    async def _initialize(self):
        """Initialize Whisper model asynchronously."""
        if self._initialized:
            return
        
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            
            # Load model in executor to avoid blocking
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type
                )
            )
            
            self._initialized = True
            logger.success(f"Whisper model loaded: {self.model_size} on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize STT: {e}")
            raise
    
    async def process_frame(self, frame: Frame, direction: str = "down"):
        """
        Process audio frames and convert to text.
        
        Args:
            frame: Input frame (AudioRawFrame expected)
            direction: Processing direction
        """
        await super().process_frame(frame, direction)
        
        # Only process AudioRawFrame
        if not isinstance(frame, AudioRawFrame):
            await self.push_frame(frame, direction)
            return
        
        # Initialize if needed
        if not self._initialized:
            await self._initialize()
        
        try:
            # Accumulate audio
            self._audio_buffer.extend(frame.audio)
            
            # Process when we have enough audio (e.g., 3 seconds)
            min_audio_length = self._sample_rate * 3 * 2  # 3 seconds, 16-bit audio
            
            if len(self._audio_buffer) >= min_audio_length:
                # Convert to numpy array
                audio_np = np.frombuffer(self._audio_buffer, dtype=np.int16)
                audio_float = audio_np.astype(np.float32) / 32768.0
                
                # Transcribe
                text = await self._transcribe(audio_float)
                
                if text.strip():
                    # Create text frame
                    text_frame = TextFrame(text=text)
                    await self.push_frame(text_frame, direction)
                    logger.debug(f"Transcribed: {text}")
                
                # Clear buffer
                self._audio_buffer.clear()
                
        except Exception as e:
            logger.error(f"STT error: {e}")
            error_frame = ErrorFrame(error=str(e))
            await self.push_frame(error_frame, direction)
    
    async def _transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe audio using Whisper.
        
        Args:
            audio: Audio data as numpy array (float32, 16kHz)
            
        Returns:
            Transcribed text
        """
        if not self._model:
            raise RuntimeError("Whisper model not initialized")
        
        # Run transcription in executor
        loop = asyncio.get_event_loop()
        segments, info = await loop.run_in_executor(
            None,
            lambda: self._model.transcribe(
                audio,
                language=self.language,
                vad_filter=self.vad_filter,
                beam_size=self.beam_size
            )
        )
        
        # Concatenate all segments
        text = " ".join([segment.text for segment in segments])
        return text.strip()
    
    async def cleanup(self):
        """Cleanup resources."""
        if self._model:
            del self._model
            self._model = None
        
        self._audio_buffer.clear()
        self._initialized = False
        logger.info("LocalSTTService cleaned up")


class StreamingSTTService(FrameProcessor):
    """
    Advanced streaming STT service with VAD and chunk processing.
    Optimized for real-time conversation with minimal latency.
    """
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        chunk_length_s: float = 3.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.model_size = model_size
        self.device = device
        self.chunk_length_s = chunk_length_s
        self._model = None
        self._sample_rate = 16000
        
        # Streaming buffer
        self._stream_buffer = bytearray()
        self._min_chunk_size = int(self._sample_rate * chunk_length_s * 2)
        
        logger.info(f"StreamingSTTService initialized (chunk: {chunk_length_s}s)")
    
    async def _initialize(self):
        """Initialize streaming model."""
        if self._model:
            return
        
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(
            None,
            lambda: WhisperModel(self.model_size, device=self.device)
        )
        logger.success("Streaming Whisper model loaded")
    
    async def process_frame(self, frame: Frame, direction: str = "down"):
        """Process streaming audio frames."""
        await super().process_frame(frame, direction)
        
        if not isinstance(frame, AudioRawFrame):
            await self.push_frame(frame, direction)
            return
        
        if not self._model:
            await self._initialize()
        
        # Add to streaming buffer
        self._stream_buffer.extend(frame.audio)
        
        # Process chunks when buffer is full
        while len(self._stream_buffer) >= self._min_chunk_size:
            # Extract chunk
            chunk = bytes(self._stream_buffer[:self._min_chunk_size])
            self._stream_buffer = self._stream_buffer[self._min_chunk_size:]
            
            # Transcribe chunk
            audio_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            
            loop = asyncio.get_event_loop()
            segments, _ = await loop.run_in_executor(
                None,
                lambda: self._model.transcribe(audio_np, language="fr", beam_size=3)
            )
            
            text = " ".join([seg.text for seg in segments]).strip()
            
            if text:
                text_frame = TextFrame(text=text)
                await self.push_frame(text_frame, direction)
                logger.debug(f"Streamed: {text}")
    
    async def cleanup(self):
        """Cleanup streaming resources."""
        if self._model:
            del self._model
        self._stream_buffer.clear()
        logger.info("StreamingSTTService cleaned up")
