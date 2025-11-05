"""
Local Text-to-Speech Service for Pipecat using Piper TTS.
Provides fast, local speech synthesis with multiple voice options.
"""

import asyncio
import io
import subprocess
import tempfile
from typing import Optional, AsyncGenerator
from pathlib import Path

from pipecat.frames.frames import Frame, AudioRawFrame, TextFrame, ErrorFrame, TTSAudioRawFrame
from pipecat.processors.frame_processor import FrameProcessor
from loguru import logger

try:
    from piper import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False
    logger.warning("Piper TTS not available. Install with: pip install piper-tts")


class LocalTTSService(FrameProcessor):
    """
    Local TTS service using Piper for fast, high-quality speech synthesis.
    
    Features:
    - Multiple voice models (FR/EN)
    - Low latency streaming
    - GPU acceleration support
    - No API calls required
    
    Args:
        voice_model: Path to Piper voice model (.onnx file)
        language: Language code ('fr-FR', 'en-US', etc.)
        sample_rate: Output audio sample rate (default: 22050)
        speed: Speech speed multiplier (default: 1.0)
    """
    
    def __init__(
        self,
        voice_model: str = "fr_FR-siwis-medium",
        language: str = "fr-FR",
        sample_rate: int = 22050,
        speed: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        if not PIPER_AVAILABLE:
            raise RuntimeError("Piper TTS is not installed. Run: pip install piper-tts")
        
        self.voice_model = voice_model
        self.language = language
        self.sample_rate = sample_rate
        self.speed = speed
        self._voice: Optional[PiperVoice] = None
        self._initialized = False
        
        logger.info(f"LocalTTSService initialized with model: {voice_model}")
    
    async def _initialize(self):
        """Initialize Piper voice model asynchronously."""
        if self._initialized:
            return
        
        try:
            # Try project-local models first (for Colab)
            project_model_path = Path("models/voices") / f"{self.voice_model}.onnx"
            
            if project_model_path.exists():
                logger.info(f"Using project voice model: {project_model_path}")
                model_path = project_model_path
            else:
                # Fallback to default Piper location
                model_path = Path.home() / ".local" / "share" / "piper" / "voices" / f"{self.voice_model}.onnx"
                
                if not model_path.exists():
                    logger.info(f"Downloading voice model: {self.voice_model}")
                    await self._download_voice_model()
            
            # Load voice model
            logger.info(f"Loading voice from: {model_path}")
            self._voice = PiperVoice.load(str(model_path))
            self._initialized = True
            logger.success(f"Voice model loaded: {self.voice_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS: {e}")
            raise
    
    async def _download_voice_model(self):
        """Download Piper voice model from official repository."""
        voice_url = f"https://github.com/rhasspy/piper/releases/download/v1.2.0/{self.voice_model}.onnx"
        model_path = Path.home() / ".local" / "share" / "piper" / "voices"
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Use wget or curl to download
        cmd = [
            "wget",
            "-O",
            str(model_path / f"{self.voice_model}.onnx"),
            voice_url
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()
    
    async def process_frame(self, frame: Frame, direction: str = "down"):
        """
        Process text frames and convert to audio.
        
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
            # Synthesize speech
            text = frame.text
            
            if not text.strip():
                return
            
            logger.debug(f"Synthesizing: {text[:50]}...")
            
            # Generate audio
            audio_bytes = await self._synthesize(text)
            
            # Skip if no audio generated
            if not audio_bytes or len(audio_bytes) == 0:
                logger.warning("No audio generated, skipping frame")
                return
            
            # Create audio frame using TTSAudioRawFrame (has proper ID handling)
            audio_frame = TTSAudioRawFrame(
                audio=audio_bytes,
                sample_rate=self.sample_rate,
                num_channels=1
            )
            
            # Push audio frame downstream
            await self.push_frame(audio_frame, direction)
            logger.debug(f"Audio generated: {len(audio_bytes)} bytes")
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            error_frame = ErrorFrame(error=str(e))
            await self.push_frame(error_frame, direction)
    
    async def _synthesize(self, text: str) -> bytes:
        """
        Synthesize speech from text using Piper.
        
        Args:
            text: Input text to synthesize
            
        Returns:
            Audio data as bytes (PCM 16-bit)
        """
        if not self._voice:
            raise RuntimeError("Voice model not initialized")
        
        # Run synthesis in executor to avoid blocking
        loop = asyncio.get_event_loop()
        
        def _synthesize_blocking(voice, text_input):
            """Blocking synthesis that consumes the generator."""
            import numpy as np
            import wave
            import io
            
            # Collect all audio from Piper generator
            audio_chunks = []
            chunk_count = 0
            
            for audio_chunk in voice.synthesize(text_input):
                chunk_count += 1
                
                # Debug: log chunk type
                chunk_type = type(audio_chunk).__name__
                logger.debug(f"Chunk {chunk_count}: type={chunk_type}")
                
                # Try different extraction methods
                if isinstance(audio_chunk, bytes):
                    audio_chunks.append(audio_chunk)
                elif isinstance(audio_chunk, np.ndarray):
                    # Convert numpy array to PCM bytes
                    if audio_chunk.dtype == np.float32:
                        # Convert float32 [-1, 1] to int16 PCM
                        audio_int16 = (audio_chunk * 32767).astype(np.int16)
                    else:
                        audio_int16 = audio_chunk.astype(np.int16)
                    audio_chunks.append(audio_int16.tobytes())
                elif hasattr(audio_chunk, 'audio'):
                    # Extract audio attribute
                    audio_data = audio_chunk.audio
                    if isinstance(audio_data, bytes):
                        audio_chunks.append(audio_data)
                    elif isinstance(audio_data, np.ndarray):
                        if audio_data.dtype == np.float32:
                            audio_int16 = (audio_data * 32767).astype(np.int16)
                        else:
                            audio_int16 = audio_data.astype(np.int16)
                        audio_chunks.append(audio_int16.tobytes())
            
            logger.debug(f"Collected {chunk_count} chunks, total {len(audio_chunks)} byte arrays")
            
            # Concatenate all chunks
            if not audio_chunks:
                logger.warning("No audio chunks collected!")
                return b''
            
            result = b''.join(audio_chunks)
            logger.debug(f"Final audio size: {len(result)} bytes")
            return result
        
        audio_data = await loop.run_in_executor(
            None,
            _synthesize_blocking,
            self._voice,
            text
        )
        
        return audio_data
    
    async def cleanup(self):
        """Cleanup resources."""
        if self._voice:
            del self._voice
            self._voice = None
        
        self._initialized = False
        logger.info("LocalTTSService cleaned up")


class CoquiTTSService(FrameProcessor):
    """
    Alternative TTS service using Coqui TTS (better quality, slower).
    
    Note: Heavier than Piper but more natural sounding.
    Install with: pip install coqui-tts
    """
    
    def __init__(
        self,
        model_name: str = "tts_models/fr/css10/vits",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self._tts = None
        
        try:
            from TTS.api import TTS
            self._tts = TTS(model_name)
            logger.info(f"Coqui TTS initialized: {model_name}")
        except ImportError:
            raise RuntimeError("Coqui TTS not installed. Run: pip install coqui-tts")
    
    async def process_frame(self, frame: Frame, direction: str = "down"):
        """Process text frames and convert to audio using Coqui TTS."""
        await super().process_frame(frame, direction)
        
        if not isinstance(frame, TextFrame):
            await self.push_frame(frame, direction)
            return
        
        try:
            text = frame.text
            if not text.strip():
                return
            
            # Generate audio using Coqui
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                self._tts.tts_to_file(text=text, file_path=tmp_file.name)
                
                # Read audio file
                import soundfile as sf
                audio_data, sample_rate = sf.read(tmp_file.name)
                
                # Convert to bytes
                audio_bytes = (audio_data * 32767).astype('int16').tobytes()
                
                # Create audio frame
                audio_frame = AudioRawFrame(
                    audio=audio_bytes,
                    sample_rate=sample_rate,
                    num_channels=1
                )
                
                await self.push_frame(audio_frame, direction)
                
        except Exception as e:
            logger.error(f"Coqui TTS error: {e}")
            error_frame = ErrorFrame(error=str(e))
            await self.push_frame(error_frame, direction)
