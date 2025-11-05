"""
Interface Gradio pour Agent Vocal Temps Réel
Permet une conversation continue via interface web
"""

import asyncio
import numpy as np
import gradio as gr
from loguru import logger
import time
from typing import Optional, Tuple

from src.realtime_voice_agent import RealtimeVoiceAgent, create_realtime_voice_agent


class GradioRealtimeInterface:
    """
    Interface Gradio pour conversation vocale en temps réel
    """
    
    def __init__(self, agent: RealtimeVoiceAgent):
        """
        Initialize interface
        
        Args:
            agent: Initialized RealtimeVoiceAgent
        """
        self.agent = agent
        self.is_recording = False
        self.conversation_active = False
        
    async def start_session(self):
        """Start a new conversation session"""
        if not self.conversation_active:
            logger.info("🎬 Starting new session")
            self.conversation_active = True
            # Reset conversation history
            if self.agent.conversation_manager:
                self.agent.conversation_manager.conversation_history.clear()
            return "✅ Session démarrée - Vous pouvez maintenant parler"
        return "⚠️ Session déjà active"
    
    async def stop_session(self):
        """Stop current conversation session"""
        if self.conversation_active:
            logger.info("⏹️ Stopping session")
            self.conversation_active = False
            await self.agent.stop_conversation()
            return "✅ Session terminée"
        return "⚠️ Aucune session active"
    
    async def process_audio_input(
        self,
        audio_input: Optional[Tuple[int, np.ndarray]]
    ) -> Tuple[str, Optional[Tuple[int, np.ndarray]]]:
        """
        Process audio input from microphone
        
        Args:
            audio_input: Tuple of (sample_rate, audio_array) from Gradio Audio
            
        Returns:
            Tuple of (status_message, audio_output)
        """
        if not self.conversation_active:
            return "⚠️ Session non active - Cliquez sur 'Démarrer Session'", None
        
        if audio_input is None:
            return "⚠️ Aucun audio détecté", None
        
        start_time = time.time()
        
        try:
            sample_rate, audio_data = audio_input
            
            logger.info(f"🎤 Audio reçu: {len(audio_data)} samples @ {sample_rate}Hz")
            
            # Convert to int16 PCM bytes
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32767).astype(np.int16)
            
            audio_bytes = audio_data.tobytes()
            
            # Process through pipeline
            await self.agent.process_audio_chunk(audio_bytes, sample_rate)
            
            # Wait a bit for processing
            await asyncio.sleep(1.0)
            
            # Get response audio
            response_audio = self.agent.audio_collector.get_audio()
            
            if response_audio and len(response_audio) > 0:
                # Convert bytes to numpy array
                audio_array = np.frombuffer(response_audio, dtype=np.int16)
                output_audio = (22050, audio_array)
                
                elapsed = time.time() - start_time
                status = f"✅ Réponse générée en {elapsed:.1f}s ({len(response_audio)} bytes)"
                
                # Clear audio collector for next turn
                self.agent.audio_collector.clear()
                
                return status, output_audio
            else:
                return "⚠️ Aucune réponse audio générée", None
                
        except Exception as e:
            logger.error(f"❌ Error processing audio: {e}", exc_info=True)
            return f"❌ Erreur: {str(e)}", None
    
    async def process_text_input(self, text_input: str) -> Tuple[str, str, Optional[Tuple[int, np.ndarray]]]:
        """
        Process text input (alternative to voice)
        
        Args:
            text_input: User question as text
            
        Returns:
            Tuple of (subject, response_text, audio_output)
        """
        if not self.conversation_active:
            return "❓ Non détecté", "⚠️ Session non active", None
        
        if not text_input or text_input.strip() == "":
            return "❓ Non détecté", "⚠️ Entrez une question", None
        
        start_time = time.time()
        
        try:
            logger.info(f"📝 Text input: {text_input}")
            
            # Get RAG context
            subject, context = self.agent.rag_service.retrieve(text_input)
            logger.info(f"📚 Subject: {subject}")
            
            # Build prompt
            system_prompt = f"""Tu es un tuteur IA spécialisé en {subject}.
Utilise le contexte suivant pour répondre de manière précise et pédagogique.

Contexte:
{context}

Réponds de manière claire et concise (2-3 phrases maximum).
N'utilise pas de caractères spéciaux car ta réponse sera convertie en audio."""
            
            # Get LLM response
            response_text = await self.agent.llm_service.generate_response(
                prompt=text_input,
                system_prompt=system_prompt
            )
            
            logger.info(f"🤖 Response: {response_text[:100]}...")
            
            # Generate audio
            audio_bytes = await self.agent.tts_service.synthesize(response_text)
            
            # Prepare output
            subject_emoji = {
                'maths': '🔢 Mathématiques',
                'physique': '⚛️ Physique',
                'anglais': '🇬🇧 Anglais',
                'unknown': '❓ Non détecté'
            }.get(subject, f'📚 {subject}')
            
            audio_output = None
            if audio_bytes and len(audio_bytes) > 0:
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_output = (22050, audio_array)
            
            elapsed = time.time() - start_time
            response_with_info = f"{response_text}\n\n⏱️ Temps: {elapsed:.1f}s"
            
            return subject_emoji, response_with_info, audio_output
            
        except Exception as e:
            logger.error(f"❌ Error: {e}", exc_info=True)
            return "❌ Erreur", f"Erreur: {str(e)}", None
    
    def create_interface(self) -> gr.Blocks:
        """
        Create Gradio interface
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(title="Agent Vocal IA Local - Temps Réel") as interface:
            gr.Markdown("""
            # 🎙️ Agent Vocal IA Local - Temps Réel avec RAG
            
            **Mode Temps Réel** : Conversation continue jusqu'à déconnexion
            
            ### 🚀 Comment utiliser:
            1. Cliquez sur **"Démarrer Session"**
            2. Parlez via le micro OU tapez votre question
            3. L'IA répond avec contexte RAG (maths/physique/anglais)
            4. Cliquez sur **"Arrêter Session"** pour terminer
            
            ---
            """)
            
            with gr.Row():
                with gr.Column():
                    session_status = gr.Textbox(
                        label="📊 Statut Session",
                        value="⚪ Session non démarrée",
                        interactive=False
                    )
                    
                    with gr.Row():
                        start_btn = gr.Button("▶️ Démarrer Session", variant="primary")
                        stop_btn = gr.Button("⏹️ Arrêter Session", variant="stop")
            
            gr.Markdown("### 🎤 Mode Audio (Micro)")
            
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="🎤 Parlez ici",
                        streaming=False
                    )
                    audio_process_btn = gr.Button("🎵 Traiter Audio")
                
                with gr.Column():
                    audio_output = gr.Audio(
                        label="🔊 Réponse Audio",
                        type="numpy"
                    )
                    audio_status = gr.Textbox(
                        label="📊 Statut",
                        interactive=False
                    )
            
            gr.Markdown("### 💬 Mode Texte (Alternative)")
            
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(
                        label="📝 Votre question",
                        placeholder="Ex: Comment résoudre une équation du second degré ?",
                        lines=3
                    )
                    text_submit_btn = gr.Button("📤 Envoyer")
                
                with gr.Column():
                    subject_output = gr.Textbox(
                        label="📚 Domaine détecté",
                        interactive=False
                    )
                    response_output = gr.Textbox(
                        label="💡 Réponse",
                        lines=8,
                        interactive=False
                    )
                    text_audio_output = gr.Audio(
                        label="🔊 Audio de la réponse",
                        type="numpy"
                    )
            
            # Event handlers
            start_btn.click(
                fn=self.start_session,
                outputs=session_status
            )
            
            stop_btn.click(
                fn=self.stop_session,
                outputs=session_status
            )
            
            audio_process_btn.click(
                fn=self.process_audio_input,
                inputs=audio_input,
                outputs=[audio_status, audio_output]
            )
            
            text_submit_btn.click(
                fn=self.process_text_input,
                inputs=text_input,
                outputs=[subject_output, response_output, text_audio_output]
            )
        
        return interface


async def launch_gradio_realtime(
    whisper_model: str = "base",
    ollama_model: str = "qwen2:1.5b",
    device: str = "cuda",
    share: bool = False
):
    """
    Launch Gradio interface for realtime voice agent
    
    Args:
        whisper_model: Whisper model size
        ollama_model: Ollama model name
        device: Device (cuda/cpu)
        share: Whether to create public link
    """
    logger.info("🎨 Launching Gradio Realtime Interface...")
    
    # Create and initialize agent
    agent = await create_realtime_voice_agent(
        whisper_model=whisper_model,
        ollama_model=ollama_model,
        device=device
    )
    
    # Create interface
    ui = GradioRealtimeInterface(agent)
    interface = ui.create_interface()
    
    # Launch
    logger.info("🚀 Gradio interface ready!")
    interface.launch(
        share=share,
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=False
    )


if __name__ == "__main__":
    import sys
    
    # Parse args
    device = "cuda" if "--cpu" not in sys.argv else "cpu"
    share = "--share" in sys.argv
    
    # Run
    asyncio.run(launch_gradio_realtime(
        whisper_model="base",
        ollama_model="qwen2:1.5b",
        device=device,
        share=share
    ))
