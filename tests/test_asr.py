"""
Tests for ASR module.
"""

import pytest
import numpy as np
import tempfile
import wave
from pathlib import Path

from src.asr import ASREngine


def test_asr_engine_initialization():
    """Test ASR engine initialization."""
    engine = ASREngine()
    
    assert engine.config is not None
    assert engine.whisper_model is not None


def test_detect_voice_activity():
    """Test VAD on synthetic audio."""
    engine = ASREngine()
    
    # Create synthetic speech-like audio (sine wave)
    sample_rate = 16000
    duration = 1.0
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32) * 0.5
    
    # Test VAD (may or may not detect as speech, but should not crash)
    has_speech = engine.detect_voice_activity(audio, sample_rate)
    assert isinstance(has_speech, bool)


def test_load_audio_file():
    """Test loading audio file."""
    engine = ASREngine()
    
    # Create temporary WAV file
    sample_rate = 16000
    duration = 1.0
    audio_data = np.random.randn(int(sample_rate * duration)).astype(np.float32)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        temp_path = Path(f.name)
    
    try:
        # Write WAV file
        audio_int16 = (audio_data * 32767).astype(np.int16)
        with wave.open(str(temp_path), 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        # Load the file
        loaded_audio, loaded_sr = engine.load_audio_file(temp_path)
        
        assert loaded_sr == sample_rate
        assert len(loaded_audio) > 0
        assert loaded_audio.dtype == np.float32
    
    finally:
        if temp_path.exists():
            temp_path.unlink()


def test_transcribe_empty_audio():
    """Test transcription with empty/silent audio."""
    engine = ASREngine()
    
    # Create silent audio
    sample_rate = 16000
    duration = 0.5
    audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
    
    # Transcribe (should return empty or minimal text)
    transcript = engine.transcribe_audio(audio)
    assert isinstance(transcript, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
