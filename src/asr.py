"""
ASR Module - Automatic Speech Recognition with VAD and Streaming.
Uses Silero VAD for voice activity detection and Faster-Whisper for transcription.
"""

import wave
import numpy as np
from typing import Iterator, Optional, Union
from pathlib import Path
import torch

from .config import get_config
from .utils import get_logger


logger = get_logger(__name__)


class ASREngine:
    """
    Streaming ASR engine with voice activity detection.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ASR engine.
        
        Args:
            config_path: Path to config file
        """
        self.config = get_config(config_path)
        self.whisper_model = None
        self.vad_model = None
        
        self._load_models()
    
    def _load_models(self):
        """Load Whisper and VAD models."""
        logger.info("Loading ASR models...")
        
        # Load Faster-Whisper
        try:
            from faster_whisper import WhisperModel
            
            model_size = self.config.asr.model
            device = self.config.asr.device
            compute_type = self.config.asr.compute_type
            
            logger.info(f"Loading Whisper model: {model_size} on {device}")
            self.whisper_model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type
            )
            logger.info("Whisper model loaded successfully")
        
        except ImportError:
            logger.error("faster-whisper not installed. Install with: pip install faster-whisper")
            raise
        
        # Load Silero VAD
        try:
            logger.info("Loading Silero VAD model...")
            self.vad_model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self.get_speech_timestamps = utils[0]
            logger.info("VAD model loaded successfully")
        
        except Exception as e:
            logger.warning(f"Could not load VAD model: {e}. Continuing without VAD.")
            self.vad_model = None
    
    def detect_voice_activity(self, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Detect if audio contains speech.
        
        Args:
            audio: Audio samples as numpy array
            sample_rate: Sample rate in Hz
        
        Returns:
            True if speech is detected, False otherwise
        """
        if self.vad_model is None:
            # If VAD not available, assume speech is present
            return True
        
        try:
            # Ensure audio is float32 and normalized
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Normalize to [-1, 1] if not already
            if audio.max() > 1.0 or audio.min() < -1.0:
                audio = audio / np.abs(audio).max()
            
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio)
            
            # Get speech timestamps
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.vad_model,
                threshold=self.config.asr.vad_threshold,
                sampling_rate=sample_rate,
                min_speech_duration_ms=self.config.asr.vad_min_speech_duration_ms,
                min_silence_duration_ms=self.config.asr.vad_min_silence_duration_ms
            )
            
            return len(speech_timestamps) > 0
        
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return True  # Default to assuming speech on error
    
    def transcribe_audio(self, audio: Union[np.ndarray, str, Path], 
                        language: Optional[str] = None) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio as numpy array, or path to audio file
            language: Language code (default from config)
        
        Returns:
            Transcribed text
        """
        if language is None:
            language = self.config.asr.language
        
        logger.debug(f"Transcribing audio in language: {language}")
        
        try:
            # Transcribe
            segments, info = self.whisper_model.transcribe(
                audio,
                language=language,
                beam_size=5,
                vad_filter=False  # We handle VAD separately
            )
            
            # Combine segments
            text = " ".join([segment.text for segment in segments])
            
            logger.debug(f"Transcription complete: {len(text)} characters")
            return text.strip()
        
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
    
    def transcribe_stream(self, audio_source: Iterator[np.ndarray], 
                         sample_rate: int = 16000,
                         language: Optional[str] = None) -> Iterator[str]:
        """
        Transcribe streaming audio chunks.
        
        Args:
            audio_source: Iterator yielding audio chunks as numpy arrays
            sample_rate: Sample rate in Hz
            language: Language code
        
        Yields:
            Transcribed text segments
        """
        if language is None:
            language = self.config.asr.language
        
        logger.info("Starting streaming transcription")
        
        buffer = np.array([], dtype=np.float32)
        min_chunk_size = sample_rate * 2  # 2 seconds minimum
        
        for audio_chunk in audio_source:
            # Ensure float32
            if audio_chunk.dtype != np.float32:
                audio_chunk = audio_chunk.astype(np.float32)
            
            # Add to buffer
            buffer = np.concatenate([buffer, audio_chunk])
            
            # Process if buffer is large enough
            if len(buffer) >= min_chunk_size:
                # Check for speech
                if self.detect_voice_activity(buffer, sample_rate):
                    # Transcribe
                    text = self.transcribe_audio(buffer, language)
                    if text:
                        yield text
                
                # Clear buffer
                buffer = np.array([], dtype=np.float32)
        
        # Process remaining buffer
        if len(buffer) > 0:
            if self.detect_voice_activity(buffer, sample_rate):
                text = self.transcribe_audio(buffer, language)
                if text:
                    yield text
        
        logger.info("Streaming transcription complete")
    
    def transcribe_file(self, audio_path: Union[str, Path], 
                       language: Optional[str] = None) -> str:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to audio file
            language: Language code
        
        Returns:
            Transcribed text
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Transcribing file: {audio_path}")
        return self.transcribe_audio(str(audio_path), language)
    
    def load_audio_file(self, audio_path: Union[str, Path]) -> tuple[np.ndarray, int]:
        """
        Load audio file as numpy array.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        audio_path = Path(audio_path)
        
        try:
            import soundfile as sf
            audio, sample_rate = sf.read(str(audio_path), dtype='float32')
            
            # Convert stereo to mono if needed
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            return audio, sample_rate
        
        except ImportError:
            # Fallback to wave for WAV files
            if audio_path.suffix.lower() == '.wav':
                with wave.open(str(audio_path), 'rb') as wav_file:
                    sample_rate = wav_file.getframerate()
                    n_frames = wav_file.getnframes()
                    audio_bytes = wav_file.readframes(n_frames)
                    
                    # Convert to numpy array
                    audio = np.frombuffer(audio_bytes, dtype=np.int16)
                    audio = audio.astype(np.float32) / 32768.0
                    
                    # Convert stereo to mono
                    if wav_file.getnchannels() == 2:
                        audio = audio.reshape(-1, 2).mean(axis=1)
                    
                    return audio, sample_rate
            else:
                raise ImportError("soundfile not installed and file is not WAV")


# Singleton instance
_asr_instance: Optional[ASREngine] = None


def get_asr_engine(config_path: Optional[str] = None) -> ASREngine:
    """
    Get or create the global ASR engine instance.
    
    Args:
        config_path: Path to config file
    
    Returns:
        ASREngine instance
    """
    global _asr_instance
    if _asr_instance is None:
        _asr_instance = ASREngine(config_path)
    return _asr_instance
