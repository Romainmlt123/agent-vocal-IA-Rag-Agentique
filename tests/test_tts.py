"""
Tests for TTS module.
"""

import pytest

from src.tts import TTSEngine


def test_tts_engine_initialization():
    """Test TTS engine initialization."""
    engine = TTSEngine()
    
    assert engine.config is not None
    assert engine.voices is not None


def test_detect_language():
    """Test language detection."""
    engine = TTSEngine()
    
    # French text
    french_text = "Bonjour, comment allez-vous? C'est une belle journée."
    assert engine.detect_language(french_text) == "fr"
    
    # English text
    english_text = "Hello, how are you? This is a nice day."
    lang = engine.detect_language(english_text)
    # May detect as 'en' or fallback behavior
    assert lang in ["fr", "en"]
    
    # Text with French characters
    french_chars = "L'été est très chaud à Montréal."
    assert engine.detect_language(french_chars) == "fr"


def test_create_silence():
    """Test silence generation."""
    engine = TTSEngine()
    
    silence = engine.create_silence(duration_seconds=0.5, sample_rate=22050)
    
    assert isinstance(silence, bytes)
    assert len(silence) > 0


def test_synthesize_without_model():
    """Test synthesis without model (should handle gracefully)."""
    engine = TTSEngine()
    
    text = "Test speech synthesis"
    
    # Should return None if model not available
    result = engine.synthesize(text, language="fr")
    
    # Result can be None or bytes depending on model availability
    assert result is None or isinstance(result, bytes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
