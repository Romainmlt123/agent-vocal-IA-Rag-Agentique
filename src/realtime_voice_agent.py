"""
Agent Vocal IA Local en Temps Réel avec RAG Agentique
Basé sur l'architecture Pipecat simple-chatbot
100% Local - Pas d'API externe

Fonctionnalités:
- Conversation continue en temps réel
- VAD (Voice Activity Detection) avec Silero
- STT local avec Whisper (faster-whisper)
- RAG local avec FAISS
- LLM local avec Ollama
- TTS local avec Piper
- Gestion de session (connexion/déconnexion)
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    TextFrame,
    EndFrame,
    StartFrame,
    TranscriptionFrame,
    LLMMessagesFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

# Import local services
from src.services.local_stt import LocalSTTService
from src.services.local_llm import LocalLLMService
from src.services.local_tts import LocalTTSService
from src.services.rag_service_faiss import RAGService

# Setup logging
logger.remove(0)
logger.add(sys.stderr, level="INFO")


class ConversationManager(FrameProcessor):
    """
    Gère le contexte de conversation et l'historique des messages.
    Similaire au context_aggregator dans simple-chatbot.
    """
    
    def __init__(self, rag_service: RAGService, llm_service: LocalLLMService):
        super().__init__()
        self.rag_service = rag_service
        self.llm_service = llm_service
        self.conversation_history = []
        self._current_transcription = ""
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and manage conversation context"""
        await super().process_frame(frame, direction)
        
        # Collect user transcription
        if isinstance(frame, TranscriptionFrame):
            self._current_transcription = frame.text
            logger.info(f"👤 User: {frame.text}")
            
            # Get RAG context
            subject, context = self.rag_service.retrieve(frame.text)
            logger.info(f"📚 RAG Subject: {subject}")
            
            # Build prompt with RAG context
            system_prompt = f"""Tu es un tuteur IA spécialisé en {subject}.
Utilise le contexte suivant pour répondre de manière précise et pédagogique.

Contexte:
{context}

Réponds de manière claire et concise (2-3 phrases maximum).
N'utilise pas de caractères spéciaux car ta réponse sera convertie en audio."""

            user_message = frame.text
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            
            # Create messages for LLM
            messages = [
                {"role": "system", "content": system_prompt}
            ] + self.conversation_history
            
            # Send to LLM via frame
            llm_frame = LLMMessagesFrame(messages)
            await self.push_frame(llm_frame, direction)
            
            # Don't pass the original TranscriptionFrame
            return
        
        # Collect assistant response
        if isinstance(frame, TextFrame):
            # This is the LLM response
            response_text = frame.text
            if response_text and response_text.strip():
                logger.info(f"🤖 Assistant: {response_text}")
                
                # Add to conversation history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response_text
                })
        
        # Pass frame along
        await self.push_frame(frame, direction)


class AudioCollector(FrameProcessor):
    """
    Collecte l'audio de sortie pour le retour à l'utilisateur.
    """
    
    def __init__(self):
        super().__init__()
        self.audio_buffer = bytearray()
        self.sample_rate = 22050
        self.is_collecting = False
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Collect audio frames"""
        await super().process_frame(frame, direction)
        
        # Start collecting on TTS start
        if isinstance(frame, (TTSStartedFrame, BotStartedSpeakingFrame)):
            self.is_collecting = True
            self.audio_buffer.clear()
            logger.debug("🎙️ Started collecting audio")
        
        # Collect audio data
        if self.is_collecting and isinstance(frame, AudioRawFrame):
            self.audio_buffer.extend(frame.audio)
        
        # Stop collecting on TTS stop
        if isinstance(frame, (TTSStoppedFrame, BotStoppedSpeakingFrame)):
            self.is_collecting = False
            logger.debug(f"🎙️ Stopped collecting audio: {len(self.audio_buffer)} bytes")
        
        # Pass frame along
        await self.push_frame(frame, direction)
    
    def get_audio(self) -> bytes:
        """Get collected audio"""
        return bytes(self.audio_buffer)
    
    def clear(self):
        """Clear audio buffer"""
        self.audio_buffer.clear()


class RealtimeVoiceAgent:
    """
    Agent vocal en temps réel 100% local avec RAG.
    """
    
    def __init__(
        self,
        whisper_model: str = "base",
        ollama_model: str = "qwen2:1.5b",
        piper_voice: str = "fr_FR-siwis-medium",
        device: str = "cuda",
        rag_data_path: str = "data"
    ):
        """
        Initialize Realtime Voice Agent
        
        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            ollama_model: Ollama model name
            piper_voice: Piper voice model
            device: Device for models (cuda, cpu)
            rag_data_path: Path to RAG data
        """
        self.whisper_model = whisper_model
        self.ollama_model = ollama_model
        self.piper_voice = piper_voice
        self.device = device
        self.rag_data_path = rag_data_path
        
        # Services (will be initialized)
        self.stt_service: Optional[LocalSTTService] = None
        self.llm_service: Optional[LocalLLMService] = None
        self.tts_service: Optional[LocalTTSService] = None
        self.rag_service: Optional[RAGService] = None
        
        # Pipeline components
        self.pipeline: Optional[Pipeline] = None
        self.task: Optional[PipelineTask] = None
        self.runner: Optional[PipelineRunner] = None
        
        # Processors
        self.conversation_manager: Optional[ConversationManager] = None
        self.audio_collector: Optional[AudioCollector] = None
        
        # State
        self.is_running = False
        
        logger.info(f"🎯 Realtime Voice Agent created")
        logger.info(f"   Whisper: {whisper_model}")
        logger.info(f"   Ollama: {ollama_model}")
        logger.info(f"   Piper: {piper_voice}")
        logger.info(f"   Device: {device}")
    
    async def initialize(self):
        """Initialize all services"""
        logger.info("🔧 Initializing services...")
        
        # STT Service - Whisper with VAD
        self.stt_service = LocalSTTService(
            model_size=self.whisper_model,
            device=self.device,
            language="fr"
        )
        logger.info("✅ STT Service (Whisper + VAD)")
        
        # RAG Service
        self.rag_service = RAGService(
            data_path=self.rag_data_path,
            device=self.device
        )
        logger.info("✅ RAG Service (FAISS)")
        
        # LLM Service
        self.llm_service = LocalLLMService(
            model=self.ollama_model,
            temperature=0.7,
            max_tokens=512
        )
        logger.info("✅ LLM Service (Ollama)")
        
        # TTS Service
        self.tts_service = LocalTTSService(
            model_name=self.piper_voice,
            device=self.device
        )
        logger.info("✅ TTS Service (Piper)")
        
        # Conversation Manager
        self.conversation_manager = ConversationManager(
            rag_service=self.rag_service,
            llm_service=self.llm_service
        )
        logger.info("✅ Conversation Manager")
        
        # Audio Collector
        self.audio_collector = AudioCollector()
        logger.info("✅ Audio Collector")
        
        logger.info("🎉 All services initialized!\n")
    
    def build_pipeline(self):
        """Build the realtime pipeline"""
        logger.info("🏗️ Building pipeline...")
        
        if not all([
            self.stt_service,
            self.rag_service,
            self.llm_service,
            self.tts_service,
            self.conversation_manager,
            self.audio_collector
        ]):
            raise RuntimeError("Services not initialized. Call initialize() first.")
        
        # Pipeline flow (similar to simple-chatbot):
        # Input Audio → STT → ConversationManager → LLM → TTS → Output Audio
        self.pipeline = Pipeline([
            self.stt_service,           # Audio → Transcription
            self.conversation_manager,  # Transcription → LLM Messages (with RAG)
            self.llm_service,           # LLM Messages → Response Text
            self.tts_service,           # Response Text → Audio
            self.audio_collector        # Collect output audio
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
        self.runner = PipelineRunner(handle_sigint=True)
        
        logger.info("✅ Pipeline built successfully!\n")
    
    async def start_conversation(self):
        """Start the conversation loop (similar to simple-chatbot)"""
        if not self.pipeline or not self.task or not self.runner:
            raise RuntimeError("Pipeline not built. Call build_pipeline() first.")
        
        logger.info("🚀 Starting conversation...")
        logger.info("   Le bot écoute en continu grâce au VAD")
        logger.info("   Parlez pour démarrer la conversation\n")
        
        self.is_running = True
        
        # Queue initial frame to start pipeline
        await self.task.queue_frame(StartFrame())
        
        # Run the pipeline (this will block until stopped)
        try:
            await self.runner.run(self.task)
        except KeyboardInterrupt:
            logger.info("\n⏸️ Conversation interrupted by user")
        except Exception as e:
            logger.error(f"❌ Error during conversation: {e}", exc_info=True)
        finally:
            self.is_running = False
            logger.info("🛑 Conversation stopped")
    
    async def stop_conversation(self):
        """Stop the conversation"""
        if self.task:
            logger.info("🛑 Stopping conversation...")
            await self.task.cancel()
            self.is_running = False
    
    async def process_audio_chunk(self, audio_bytes: bytes, sample_rate: int = 16000):
        """
        Process a single audio chunk (useful for streaming)
        
        Args:
            audio_bytes: Raw PCM audio bytes
            sample_rate: Sample rate (default 16kHz for Whisper)
        """
        if not self.task:
            raise RuntimeError("Pipeline not built")
        
        # Create audio frame
        audio_frame = AudioRawFrame(
            audio=audio_bytes,
            sample_rate=sample_rate,
            num_channels=1
        )
        
        # Queue for processing
        await self.task.queue_frame(audio_frame)


async def create_realtime_voice_agent(
    whisper_model: str = "base",
    ollama_model: str = "qwen2:1.5b",
    piper_voice: str = "fr_FR-siwis-medium",
    device: str = "cuda",
    rag_data_path: str = "data"
) -> RealtimeVoiceAgent:
    """
    Factory function to create and initialize a realtime voice agent
    
    Returns:
        Initialized RealtimeVoiceAgent ready to start conversation
    """
    agent = RealtimeVoiceAgent(
        whisper_model=whisper_model,
        ollama_model=ollama_model,
        piper_voice=piper_voice,
        device=device,
        rag_data_path=rag_data_path
    )
    
    await agent.initialize()
    agent.build_pipeline()
    
    return agent


# Example usage
if __name__ == "__main__":
    async def main():
        """Example: Start a continuous conversation"""
        
        # Create agent
        agent = await create_realtime_voice_agent(
            whisper_model="base",
            ollama_model="qwen2:1.5b",
            device="cuda"
        )
        
        # Start conversation (runs until Ctrl+C)
        await agent.start_conversation()
    
    # Run
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n👋 Au revoir!")
