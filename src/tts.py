"""
TTS Module - Text-to-Speech with Piper.
Supports French and English voice synthesis with streaming.
"""

import wave
import io
from typing import Iterator, Optional, Union
from pathlib import Path
import numpy as np

from .config import get_config
from .utils import get_logger


logger = get_logger(__name__)


class TTSEngine:
    """Text-to-Speech engine using Piper."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize TTS engine.
        
        Args:
            config_path: Path to config file
        """
        self.config = get_config(config_path)
        self.voices = {}  # language -> voice instance
    
    def _load_voice(self, language: str):
        """
        Load a Piper voice model.
        
        Args:
            language: Language code (fr, en)
        
        Returns:
            Voice instance
        """
        try:
            # Note: Piper-TTS installation varies by platform
            # This is a simplified version - actual implementation may vary
            logger.info(f"Loading TTS voice for language: {language}")
            
            voice_path = self.config.get_voice_path(language)
            
            if not voice_path.exists():
                logger.warning(f"Voice model not found: {voice_path}")
                logger.warning("TTS will be simulated. Download Piper voices from: https://github.com/rhasspy/piper/releases")
                return None
            
            # Lazy import of piper
            try:
                from piper import PiperVoice
                
                voice = PiperVoice.load(
                    str(voice_path),
                    use_cuda=False  # Use CPU for compatibility
                )
                
                logger.info(f"Voice loaded successfully: {voice_path.name}")
                return voice
            
            except ImportError:
                logger.warning("piper-tts not available. Install with: pip install piper-tts")
                return None
        
        except Exception as e:
            logger.error(f"Error loading voice: {e}")
            return None
    
    def get_voice(self, language: str):
        """
        Get or load voice for a language.
        
        Args:
            language: Language code
        
        Returns:
            Voice instance or None
        """
        if language not in self.voices:
            self.voices[language] = self._load_voice(language)
        
        return self.voices[language]
    
    def synthesize(self, text: str, language: str = "fr", 
                  output_path: Optional[Union[str, Path]] = None) -> Optional[bytes]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            language: Language code
            output_path: Optional path to save WAV file
        
        Returns:
            Audio data as bytes, or None if voice not available
        """
        logger.debug(f"Synthesizing text ({len(text)} chars) in {language}")
        
        voice = self.get_voice(language)
        
        if voice is None:
            logger.warning("Voice not available, returning None")
            return None
        
        try:
            # Synthesize
            # Note: Actual Piper API may differ
            audio_bytes = io.BytesIO()
            
            # Synthesize with Piper
            voice.synthesize(
                text,
                audio_bytes,
                rate=self.config.tts.speed
            )
            
            audio_data = audio_bytes.getvalue()
            
            # Save to file if requested
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'wb') as f:
                    f.write(audio_data)
                
                logger.info(f"Audio saved to: {output_path}")
            
            return audio_data
        
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return None
    
    def synthesize_to_file(self, text: str, output_path: Union[str, Path],
                          language: str = "fr") -> bool:
        """
        Synthesize speech and save to file.
        
        Args:
            text: Text to synthesize
            output_path: Path to save WAV file
            language: Language code
        
        Returns:
            True if successful, False otherwise
        """
        audio_data = self.synthesize(text, language, output_path)
        return audio_data is not None
    
    def speak_stream(self, text_stream: Iterator[str], 
                    language: str = "fr") -> Iterator[bytes]:
        """
        Synthesize speech from streaming text.
        
        Args:
            text_stream: Iterator yielding text chunks
            language: Language code
        
        Yields:
            Audio data chunks as bytes
        """
        logger.info("Starting streaming TTS")
        
        buffer = ""
        sentence_terminators = {'.', '!', '?', '\n'}
        
        for text_chunk in text_stream:
            buffer += text_chunk
            
            # Check for sentence boundaries
            while any(term in buffer for term in sentence_terminators):
                # Find first terminator
                min_idx = len(buffer)
                for term in sentence_terminators:
                    idx = buffer.find(term)
                    if idx != -1 and idx < min_idx:
                        min_idx = idx
                
                # Extract sentence
                sentence = buffer[:min_idx + 1].strip()
                buffer = buffer[min_idx + 1:]
                
                if sentence:
                    # Synthesize sentence
                    audio_data = self.synthesize(sentence, language)
                    if audio_data:
                        yield audio_data
        
        # Process remaining buffer
        if buffer.strip():
            audio_data = self.synthesize(buffer.strip(), language)
            if audio_data:
                yield audio_data
        
        logger.info("Streaming TTS complete")
    
    def create_silence(self, duration_seconds: float, sample_rate: int = 22050) -> bytes:
        """
        Create silence audio data.
        
        Args:
            duration_seconds: Duration in seconds
            sample_rate: Sample rate in Hz
        
        Returns:
            WAV audio bytes
        """
        n_samples = int(duration_seconds * sample_rate)
        silence = np.zeros(n_samples, dtype=np.int16)
        
        # Create WAV bytes
        wav_io = io.BytesIO()
        with wave.open(wav_io, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(silence.tobytes())
        
        return wav_io.getvalue()
    
    def detect_language(self, text: str) -> str:
        """
        Simple language detection based on character patterns.
        
        Args:
            text: Text to analyze
        
        Returns:
            Language code (fr or en)
        """
        # Simple heuristic: check for French-specific characters
        french_chars = {'é', 'è', 'ê', 'à', 'â', 'ô', 'û', 'ù', 'ç', 'î', 'ï', 'ë'}
        
        text_lower = text.lower()
        french_count = sum(1 for char in text_lower if char in french_chars)
        
        # If >1% of characters are French-specific, assume French
        if len(text) > 0 and (french_count / len(text)) > 0.01:
            return "fr"
        
        # Check for common French words
        french_words = {'le', 'la', 'les', 'un', 'une', 'des', 'est', 'sont', 'dans', 'pour'}
        words = set(text_lower.split())
        
        if len(words & french_words) >= 2:
            return "fr"
        
        # Default to English
        return "en"


# Singleton instance
_tts_instance: Optional[TTSEngine] = None


def get_tts_engine(config_path: Optional[str] = None) -> TTSEngine:
    """
    Get or create the global TTS engine instance.
    
    Args:
        config_path: Path to config file
    
    Returns:
        TTSEngine instance
    """
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = TTSEngine(config_path)
    return _tts_instance
