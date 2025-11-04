"""
Configuration management for Agent Vocal Prof.
Loads and validates config.yaml settings.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ASRConfig:
    """ASR configuration."""
    model: str = "base"
    language: str = "fr"
    device: str = "auto"
    compute_type: str = "int8"
    vad_threshold: float = 0.5
    vad_min_speech_duration_ms: int = 250
    vad_min_silence_duration_ms: int = 500


@dataclass
class RAGConfig:
    """RAG configuration."""
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 4
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    similarity_threshold: float = 0.3
    indexes: Dict[str, str] = field(default_factory=dict)


@dataclass
class LLMConfig:
    """LLM configuration."""
    models: Dict[str, str] = field(default_factory=dict)
    n_ctx: int = 4096
    n_threads: int = 4
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int = 512
    stream: bool = True
    system_prompt: str = ""


@dataclass
class TTSConfig:
    """TTS configuration."""
    voices: Dict[str, str] = field(default_factory=dict)
    speed: float = 1.0
    noise_scale: float = 0.667
    noise_w: float = 0.8
    sample_rate: int = 22050


@dataclass
class RouterConfig:
    """Router configuration."""
    keywords: Dict[str, list] = field(default_factory=dict)
    use_tfidf_fallback: bool = True
    tfidf_top_n: int = 3


@dataclass
class OrchestratorConfig:
    """Orchestrator configuration."""
    max_session_duration_seconds: int = 3600
    max_retries: int = 3
    timeout_seconds: int = 30
    enable_logging: bool = True
    log_level: str = "INFO"


@dataclass
class UIConfig:
    """UI configuration."""
    server_name: str = "0.0.0.0"
    server_port: int = 7860
    share: bool = False
    enable_queue: bool = True
    max_file_size_mb: int = 10
    theme: str = "default"
    audio_chunk_duration_ms: int = 500
    show_transcript_live: bool = True
    show_rag_sources: bool = True
    show_hint_ladder: bool = True


class Config:
    """Main configuration class that loads and provides access to all settings."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to config.yaml. If None, uses default location.
        """
        if config_path is None:
            # Default to config/config.yaml relative to project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._raw_config = self._load_yaml()
        
        # Initialize all config sections
        self.asr = self._init_asr_config()
        self.rag = self._init_rag_config()
        self.llm = self._init_llm_config()
        self.tts = self._init_tts_config()
        self.router = self._init_router_config()
        self.orchestrator = self._init_orchestrator_config()
        self.ui = self._init_ui_config()
        
        # Paths
        self.project_root = Path(__file__).parent.parent
        self.data_dir = self.project_root / self._raw_config.get("paths", {}).get("data_dir", "data")
        self.models_dir = self.project_root / self._raw_config.get("paths", {}).get("models_dir", "models")
        self.logs_dir = self.project_root / self._raw_config.get("paths", {}).get("logs_dir", "logs")
        self.cache_dir = self.project_root / self._raw_config.get("paths", {}).get("cache_dir", ".cache")
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _init_asr_config(self) -> ASRConfig:
        """Initialize ASR configuration."""
        asr_dict = self._raw_config.get("asr", {})
        return ASRConfig(**asr_dict)
    
    def _init_rag_config(self) -> RAGConfig:
        """Initialize RAG configuration."""
        rag_dict = self._raw_config.get("rag", {})
        return RAGConfig(**rag_dict)
    
    def _init_llm_config(self) -> LLMConfig:
        """Initialize LLM configuration."""
        llm_dict = self._raw_config.get("llm", {})
        return LLMConfig(**llm_dict)
    
    def _init_tts_config(self) -> TTSConfig:
        """Initialize TTS configuration."""
        tts_dict = self._raw_config.get("tts", {})
        return TTSConfig(**tts_dict)
    
    def _init_router_config(self) -> RouterConfig:
        """Initialize Router configuration."""
        router_dict = self._raw_config.get("router", {})
        return RouterConfig(**router_dict)
    
    def _init_orchestrator_config(self) -> OrchestratorConfig:
        """Initialize Orchestrator configuration."""
        orch_dict = self._raw_config.get("orchestrator", {})
        return OrchestratorConfig(**orch_dict)
    
    def _init_ui_config(self) -> UIConfig:
        """Initialize UI configuration."""
        ui_dict = self._raw_config.get("ui", {})
        return UIConfig(**ui_dict)
    
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        for directory in [self.data_dir, self.models_dir, self.logs_dir, self.cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create subject subdirectories
        for subject in ["maths", "physique", "anglais"]:
            (self.data_dir / subject).mkdir(parents=True, exist_ok=True)
        
        # Create model subdirectories
        (self.models_dir / "llm").mkdir(parents=True, exist_ok=True)
        (self.models_dir / "voices").mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, subject: str) -> Path:
        """
        Get the LLM model path for a given subject.
        
        Args:
            subject: Subject name (maths, physique, anglais)
        
        Returns:
            Path to the model file
        """
        model_rel_path = self.llm.models.get(subject, self.llm.models.get("default"))
        return self.project_root / model_rel_path
    
    def get_index_path(self, subject: str) -> Path:
        """
        Get the FAISS index path for a given subject.
        
        Args:
            subject: Subject name (maths, physique, anglais)
        
        Returns:
            Path to the index file
        """
        index_rel_path = self.rag.indexes.get(subject, f"data/{subject}/index.faiss")
        return self.project_root / index_rel_path
    
    def get_voice_path(self, language: str) -> Path:
        """
        Get the TTS voice model path for a given language.
        
        Args:
            language: Language code (fr, en)
        
        Returns:
            Path to the voice model file
        """
        voice_rel_path = self.tts.voices.get(language, self.tts.voices.get("fr"))
        return self.project_root / voice_rel_path
    
    def __repr__(self) -> str:
        return f"Config(config_path={self.config_path})"


# Global config instance
_config_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get or create the global configuration instance.
    
    Args:
        config_path: Path to config.yaml. If None, uses default.
    
    Returns:
        Config instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance
