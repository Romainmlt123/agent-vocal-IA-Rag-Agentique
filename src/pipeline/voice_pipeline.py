"""
Pipeline orchestrator for voice AI agent with RAG.

This module provides the main pipeline that integrates:
- STT (Whisper) -> RAG (multi-subject routing) -> LLM (Ollama) -> TTS (Piper)

Based on Pipecat framework for real-time streaming audio.
"""

import asyncio
import os
from typing import Optional, Dict, Any, List
from loguru import logger

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    Frame,
    TextFrame,
    TranscriptionFrame,
    LLMMessagesAppendFrame,
    LLMRunFrame,
    EndFrame,
)

from src.services.local_stt import LocalSTTService
from src.services.local_llm import LocalLLMService
from src.services.local_tts import LocalTTSService
from src.services.rag_service import RAGService


class VoicePipelineOrchestrator:
    """
    Orchestrates the complete voice AI pipeline with RAG.
    
    Pipeline flow:
    1. Audio input → Whisper STT → Text transcription
    2. Text → RAG Service (routing + retrieval) → Context
    3. Text + Context → Ollama LLM → Response
    4. Response → Piper TTS → Audio output
    
    Features:
    - Real-time streaming with <2s latency
    - Multi-subject RAG routing (maths, physique, anglais)
    - Asynchronous frame processing
    - Context aggregation for conversation history
    """
    
    def __init__(
        self,
        stt_model_size: str = "base",
        llm_model: str = "qwen2:1.5b",
        tts_voice: str = "fr_FR-siwis-medium",
        rag_top_k: int = 4,
        enable_metrics: bool = True,
        ollama_url: str = "http://localhost:11434",
    ):
        """
        Initialize the voice pipeline orchestrator.
        
        Args:
            stt_model_size: Whisper model size (tiny/base/small/medium/large)
            llm_model: Ollama model name
            tts_voice: Piper TTS voice name
            rag_top_k: Number of documents to retrieve from RAG
            enable_metrics: Enable performance metrics
            ollama_url: Ollama server URL
        """
        self.stt_model_size = stt_model_size
        self.llm_model = llm_model
        self.tts_voice = tts_voice
        self.rag_top_k = rag_top_k
        self.enable_metrics = enable_metrics
        self.ollama_url = ollama_url
        
        # Services (initialized in setup)
        self.stt: Optional[LocalSTTService] = None
        self.llm: Optional[LocalLLMService] = None
        self.tts: Optional[LocalTTSService] = None
        self.rag: Optional[RAGService] = None
        
        # Pipeline components
        self.pipeline: Optional[Pipeline] = None
        self.task: Optional[PipelineTask] = None
        self.runner: Optional[PipelineRunner] = None
        self.context_aggregator: Optional[LLMContextAggregatorPair] = None
        
        # System prompt
        self.system_prompt = """Tu es un assistant pédagogique vocal intelligent.

**Ton rôle :**
- Aider les étudiants en mathématiques, physique et anglais
- Guider sans donner les réponses directement
- Poser des questions pour stimuler la réflexion
- Expliquer les concepts de manière claire et progressive

**Règles importantes :**
- Réponds en français (sauf pour l'anglais)
- Sois concis et précis
- Utilise les documents fournis par le système RAG
- Ne génère PAS de caractères spéciaux (ta sortie sera convertie en audio)
- Si tu ne connais pas la réponse, dis-le honnêtement

**Format de réponse :**
- Phrases courtes et claires
- Pas de markdown, LaTeX ou formules complexes en texte
- Utilise des mots simples pour les formules (ex: "x au carré plus 2x égale 0")
"""
        
        logger.info(f"VoicePipelineOrchestrator initialized")
        logger.info(f"  - STT: Whisper {stt_model_size}")
        logger.info(f"  - LLM: {llm_model}")
        logger.info(f"  - TTS: {tts_voice}")
        logger.info(f"  - RAG: Top-{rag_top_k} retrieval")
    
    async def setup(self):
        """Initialize all services and build the pipeline."""
        logger.info("Setting up voice pipeline...")
        
        # 1. Initialize services
        logger.info("Initializing STT service...")
        self.stt = LocalSTTService(
            model_size=self.stt_model_size,
            language="fr",
            device="cuda"
        )
        
        logger.info("Initializing LLM service...")
        self.llm = LocalLLMService(
            base_url=self.ollama_url,
            model=self.llm_model,
            temperature=0.7,
            max_tokens=512
        )
        
        logger.info("Initializing TTS service...")
        self.tts = LocalTTSService(
            voice=self.tts_voice,
            speed=1.0
        )
        
        logger.info("Initializing RAG service...")
        self.rag = RAGService(
            data_dir="data",
            top_k=self.rag_top_k
        )
        await self.rag.initialize()
        
        # 2. Setup LLM context
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        context = LLMContext(messages)
        self.context_aggregator = LLMContextAggregatorPair(context)
        
        # 3. Build pipeline
        # Pipeline flow: STT → RAG → LLM → TTS
        logger.info("Building pipeline...")
        self.pipeline = Pipeline([
            self.stt,                           # Audio → Text transcription
            self.context_aggregator.user(),     # Add user message to context
            self.rag,                            # Retrieve relevant documents
            self.llm,                            # Generate response
            self.tts,                            # Text → Audio
            self.context_aggregator.assistant(), # Add assistant response to context
        ])
        
        # 4. Create pipeline task
        self.task = PipelineTask(
            self.pipeline,
            params=PipelineParams(
                enable_metrics=self.enable_metrics,
                enable_usage_metrics=self.enable_metrics,
            )
        )
        
        # 5. Create runner
        self.runner = PipelineRunner()
        
        logger.info("✅ Voice pipeline setup complete!")
    
    async def process_audio(self, audio_data: bytes, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Process audio input through the complete pipeline.
        
        Args:
            audio_data: Raw audio bytes
            sample_rate: Audio sample rate in Hz
            
        Returns:
            Dictionary with transcription, response, and metadata
        """
        if not self.task:
            raise RuntimeError("Pipeline not initialized. Call setup() first.")
        
        logger.info("Processing audio through pipeline...")
        
        # Queue audio frame for processing
        from pipecat.frames.frames import AudioRawFrame
        audio_frame = AudioRawFrame(audio=audio_data, sample_rate=sample_rate, num_channels=1)
        
        await self.task.queue_frame(audio_frame)
        
        # Note: In a real-time streaming scenario, you would listen to output frames
        # For now, this is a simplified interface
        logger.info("Audio queued for processing")
        
        return {
            "status": "processing",
            "message": "Audio queued for pipeline processing"
        }
    
    async def process_text(self, text: str) -> Dict[str, Any]:
        """
        Process text input through the pipeline (bypassing STT).
        
        Args:
            text: User text input
            
        Returns:
            Dictionary with response and metadata
        """
        if not self.task:
            raise RuntimeError("Pipeline not initialized. Call setup() first.")
        
        logger.info(f"Processing text: {text}")
        
        # Create transcription frame
        transcription_frame = TranscriptionFrame(text=text, user_id="user", timestamp=None)
        
        # Queue for processing
        await self.task.queue_frame(transcription_frame)
        
        return {
            "status": "processing",
            "input": text,
            "message": "Text queued for pipeline processing"
        }
    
    async def process_question(self, question: str) -> Dict[str, Any]:
        """
        Process a question through RAG + LLM (without audio processing).
        
        This is a simplified interface for testing without full pipeline.
        
        Args:
            question: User question text
            
        Returns:
            Dictionary with answer, subject, sources, etc.
        """
        if not self.rag or not self.llm:
            raise RuntimeError("Services not initialized. Call setup() first.")
        
        logger.info(f"📝 Processing question: {question}")
        
        # 1. Route to subject
        subject = self.rag.route_query(question)
        logger.info(f"🎯 Routed to subject: {subject}")
        
        # 2. Retrieve documents
        docs = self.rag.retrieve(question, subject)
        logger.info(f"📚 Retrieved {len(docs)} documents")
        
        # 3. Format context
        context = self.rag.format_context(docs)
        
        # 4. Create prompt
        prompt = f"""Documents pertinents :
{context}

Question de l'étudiant : {question}

Réponds de manière pédagogique en t'appuyant sur les documents ci-dessus.
"""
        
        # 5. Generate response with LLM
        logger.info("🤖 Generating LLM response...")
        response = await self.llm.generate(prompt)
        
        logger.info(f"✅ Response generated: {response[:100]}...")
        
        return {
            "question": question,
            "subject": subject,
            "answer": response,
            "sources": [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs],
            "num_sources": len(docs),
        }
    
    async def run(self):
        """Run the pipeline (blocking call)."""
        if not self.runner or not self.task:
            raise RuntimeError("Pipeline not initialized. Call setup() first.")
        
        logger.info("🚀 Starting pipeline runner...")
        await self.runner.run(self.task)
    
    async def stop(self):
        """Stop the pipeline gracefully."""
        if self.task:
            logger.info("Stopping pipeline...")
            await self.task.queue_frame(EndFrame())
            await self.task.cancel()
        logger.info("✅ Pipeline stopped")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics."""
        if not self.task or not self.enable_metrics:
            return {}
        
        # TODO: Extract metrics from pipeline task
        return {
            "enabled": self.enable_metrics,
            "message": "Metrics collection enabled (implementation pending)"
        }


class RAGFrameProcessor(FrameProcessor):
    """
    Custom frame processor to inject RAG context into LLM messages.
    
    This processor:
    1. Intercepts TranscriptionFrame (user text)
    2. Routes query to appropriate subject
    3. Retrieves relevant documents
    4. Adds context to LLM messages
    """
    
    def __init__(self, rag_service: RAGService):
        super().__init__()
        self.rag = rag_service
    
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and inject RAG context when needed."""
        await super().process_frame(frame, direction)
        
        # Check if this is a user transcription
        if isinstance(frame, TranscriptionFrame):
            text = frame.text
            logger.info(f"🎤 User said: {text}")
            
            # 1. Route to subject
            subject = self.rag.route_query(text)
            logger.info(f"🎯 Routed to: {subject}")
            
            # 2. Retrieve documents
            docs = self.rag.retrieve(text, subject)
            logger.info(f"📚 Retrieved {len(docs)} documents")
            
            # 3. Format context
            context = self.rag.format_context(docs)
            
            # 4. Create enhanced prompt with context
            enhanced_text = f"""Documents pertinents sur le sujet "{subject}" :
{context}

Question de l'étudiant : {text}

Réponds en t'appuyant sur les documents ci-dessus.
"""
            
            # 5. Create new message frame with enhanced context
            message_frame = LLMMessagesAppendFrame(
                messages=[{"role": "user", "content": enhanced_text}]
            )
            
            # Push the enhanced message instead of the original transcription
            await self.push_frame(message_frame, direction)
            
            # Also push an LLM run frame to trigger generation
            await self.push_frame(LLMRunFrame(), direction)
            
            return  # Don't push the original transcription frame
        
        # Push all other frames through unchanged
        await self.push_frame(frame, direction)


# Factory function for easy pipeline creation
async def create_voice_pipeline(
    stt_model_size: str = "base",
    llm_model: str = "qwen2:1.5b",
    tts_voice: str = "fr_FR-siwis-medium",
    rag_top_k: int = 4,
    enable_metrics: bool = True,
) -> VoicePipelineOrchestrator:
    """
    Factory function to create and setup a voice pipeline.
    
    Args:
        stt_model_size: Whisper model size
        llm_model: Ollama model name
        tts_voice: Piper TTS voice
        rag_top_k: Number of RAG documents to retrieve
        enable_metrics: Enable performance metrics
        
    Returns:
        Configured VoicePipelineOrchestrator instance
    """
    pipeline = VoicePipelineOrchestrator(
        stt_model_size=stt_model_size,
        llm_model=llm_model,
        tts_voice=tts_voice,
        rag_top_k=rag_top_k,
        enable_metrics=enable_metrics,
    )
    
    await pipeline.setup()
    
    return pipeline


# Example usage
if __name__ == "__main__":
    async def main():
        # Create pipeline
        pipeline = await create_voice_pipeline(
            stt_model_size="base",
            llm_model="qwen2:1.5b",
            tts_voice="fr_FR-siwis-medium",
            rag_top_k=4,
        )
        
        # Test with a question
        result = await pipeline.process_question(
            "Comment résoudre l'équation x² + 2x - 8 = 0 ?"
        )
        
        print("\n" + "="*60)
        print(f"Question: {result['question']}")
        print(f"Subject: {result['subject']}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources: {result['num_sources']} documents")
        print("="*60)
        
        # Cleanup
        await pipeline.stop()
    
    asyncio.run(main())
