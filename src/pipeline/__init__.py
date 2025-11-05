"""
Pipeline module for voice AI agent.

This module provides the orchestration layer that connects all services
(STT, RAG, LLM, TTS) into a complete voice AI pipeline using Pipecat framework.
"""

from src.pipeline.voice_pipeline import (
    VoicePipelineOrchestrator,
    RAGFrameProcessor,
    create_voice_pipeline,
)

__all__ = [
    "VoicePipelineOrchestrator",
    "RAGFrameProcessor",
    "create_voice_pipeline",
]
