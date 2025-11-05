"""
Interface Gradio pour Agent Vocal IA
Optimisée pour Google Colab avec Pipeline Pipecat
"""

import gradio as gr
import asyncio
import logging
import numpy as np
from pathlib import Path
import soundfile as sf
import io

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradioVoiceInterface:
    """Interface Gradio pour le pipeline vocal"""
    
    def __init__(self, pipeline):
        """
        Initialize Gradio interface
        
        Args:
            pipeline: VoicePipeline instance
        """
        self.pipeline = pipeline
        self.interface = None
        logger.info("Gradio interface initialized")
    
    def process_audio_sync(self, audio_input):
        """
        Process audio input (synchronous wrapper)
        
        Args:
            audio_input: Tuple (sample_rate, audio_array) from Gradio
            
        Returns:
            Tuple (transcription, subject, response, audio_output)
        """
        if audio_input is None:
            return "❌ Aucun audio détecté", "", "", None
        
        try:
            sample_rate, audio_array = audio_input
            
            logger.info(f"Received audio: shape={audio_array.shape}, sr={sample_rate}")
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            # Normalize to int16
            if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
                audio_array = (audio_array * 32767).astype(np.int16)
            
            # Convert to bytes
            audio_bytes = audio_array.tobytes()
            
            # Run pipeline asynchronously using existing event loop if available
            try:
                # Try to get existing loop (Colab/Jupyter)
                loop = asyncio.get_running_loop()
                # If we have a running loop, use run_in_executor to avoid conflicts
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    result = executor.submit(
                        lambda: asyncio.run(self.pipeline.process_audio(audio_bytes, sample_rate))
                    ).result()
            except RuntimeError:
                # No running loop, create a new one
                result = asyncio.run(self.pipeline.process_audio(audio_bytes, sample_rate))
            
            # Extract results
            transcription = result.get('transcription', 'Aucune transcription')
            subject = result.get('subject', 'unknown')
            response = result.get('response', 'Aucune réponse')
            audio_output_bytes = result.get('audio_output', b'')
            output_sample_rate = result.get('sample_rate', 22050)
            
            logger.info(f"Result: subject={subject}, trans_len={len(transcription)}, resp_len={len(response)}, audio_len={len(audio_output_bytes)}")
            
            # Convert output audio bytes to numpy array for Gradio
            audio_output = None
            if audio_output_bytes:
                audio_array_out = np.frombuffer(audio_output_bytes, dtype=np.int16)
                audio_output = (output_sample_rate, audio_array_out)
            
            # Format subject emoji
            subject_emoji = {
                'maths': '🔢 Mathématiques',
                'physique': '⚛️ Physique',
                'anglais': '🇬🇧 Anglais',
                'unknown': '❓ Non détecté'
            }.get(subject, f'📚 {subject}')
            
            logger.info(f"✅ Processing complete: {subject}")
            
            return transcription, subject_emoji, response, audio_output
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            return f"❌ Erreur: {str(e)}", "", "", None
    
    async def process_text_async(self, text_input):
        """
        Process text input (async version for Colab)
        
        Args:
            text_input: Text question from user
            
        Returns:
            Tuple (subject, response, audio_output)
        """
        # IMMEDIATE TEST - return dummy data to verify interface works
        if text_input and "TEST" in text_input.upper():
            return "🧪 TEST", "Interface fonctionne ! Question reçue: " + text_input, None
        
        if not text_input or text_input.strip() == "":
            return "❓ Non détecté", "❌ Veuillez entrer une question", None
        
        try:
            logger.info(f"Processing text (async): '{text_input}'")
            
            # Call pipeline directly without creating new event loop
            result = await self.pipeline.process_text(text_input)
            
            logger.info(f"Pipeline returned: {result.keys() if result else 'None'}")
            
            # Extract results
            subject = result.get('subject', 'unknown')
            response = result.get('response', 'Aucune réponse')
            audio_output_bytes = result.get('audio_output', b'')
            output_sample_rate = result.get('sample_rate', 22050)
            
            logger.info(f"Result: subject={subject}, response_len={len(response)}, audio_len={len(audio_output_bytes)}")
            
            # DEBUG: Add debug info to response
            debug_info = f"\n\n--- DEBUG INFO ---\n"
            debug_info += f"Domaine brut: {subject}\n"
            debug_info += f"Longueur réponse: {len(response)} caractères\n"
            debug_info += f"Audio généré: {len(audio_output_bytes)} bytes\n"
            debug_info += f"Sample rate: {output_sample_rate} Hz\n"
            
            # If response is empty, add more debug
            if not response or response == 'Aucune réponse':
                debug_info += "\n⚠️ PROBLÈME: Réponse vide!\n"
                debug_info += f"Résultat complet du pipeline: {result}\n"
            
            # Add debug to response
            response_with_debug = response + debug_info
            
            # Convert output audio bytes to numpy array for Gradio
            audio_output = None
            if audio_output_bytes and len(audio_output_bytes) > 0:
                logger.info(f"Converting {len(audio_output_bytes)} bytes to audio array")
                audio_array_out = np.frombuffer(audio_output_bytes, dtype=np.int16)
                audio_output = (output_sample_rate, audio_array_out)
                logger.info(f"Audio output shape: {audio_array_out.shape}")
            else:
                logger.warning("No audio output bytes received!")
            
            # Format subject emoji
            subject_emoji = {
                'maths': '🔢 Mathématiques',
                'physique': '⚛️ Physique',
                'anglais': '🇬🇧 Anglais',
                'unknown': '❓ Non détecté'
            }.get(subject, f'📚 {subject}')
            
            logger.info(f"✅ Text processing complete: {subject} -> {subject_emoji}")
            
            return subject_emoji, response_with_debug, audio_output
            
        except Exception as e:
            logger.error(f"Error processing text: {e}", exc_info=True)
            return "", f"❌ Erreur: {str(e)}", None
    
    def process_text_sync(self, text_input):
        """
        Synchronous wrapper that runs async function in event loop
        """
        try:
            # Get the current event loop (should be Colab's loop)
            import nest_asyncio
            nest_asyncio.apply()  # Allow nested event loops
            
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.process_text_async(text_input))
        except Exception as e:
            logger.error(f"Sync wrapper error: {e}", exc_info=True)
            return "", f"❌ Erreur: {str(e)}", None
    
    def build_interface(self):
        """Build Gradio interface"""
        
        # Custom CSS
        custom_css = """
        .gradio-container {
            font-family: 'Arial', sans-serif;
        }
        .header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .output-box {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }
        """
        
        with gr.Blocks(css=custom_css, title="Agent Vocal IA - RAG Agentique") as interface:
            
            # Header
            gr.HTML("""
                <div class="header">
                    <h1>🎤 Agent Vocal IA avec RAG Agentique</h1>
                    <p>Assistant pédagogique intelligent - Maths, Physique, Anglais</p>
                </div>
            """)
            
            # Tabs for different input methods
            with gr.Tabs():
                
                # Tab 1: Audio Input
                with gr.Tab("🎙️ Entrée Vocale"):
                    gr.Markdown("""
                    ### Posez votre question à voix haute
                    Cliquez sur le microphone, posez votre question, puis cliquez sur le bouton "Traiter l'audio".
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            audio_input = gr.Audio(
                                sources=["microphone"],  # 'sources' (plural) in newer Gradio
                                type="numpy",
                                label="🎤 Enregistrement vocal"
                            )
                            audio_submit_btn = gr.Button("🚀 Traiter l'audio", variant="primary", size="lg")
                        
                        with gr.Column(scale=1):
                            audio_transcription = gr.Textbox(
                                label="📝 Transcription",
                                placeholder="Votre question apparaîtra ici...",
                                lines=2
                            )
                            audio_subject = gr.Textbox(
                                label="📚 Domaine détecté",
                                placeholder="Matière identifiée...",
                                lines=1
                            )
                            audio_response = gr.Textbox(
                                label="💡 Réponse de l'IA",
                                placeholder="La réponse apparaîtra ici...",
                                lines=8
                            )
                            audio_output = gr.Audio(
                                label="🔊 Réponse vocale",
                                type="numpy"
                            )
                    
                    # Audio processing
                    audio_submit_btn.click(
                        fn=self.process_audio_sync,
                        inputs=[audio_input],
                        outputs=[audio_transcription, audio_subject, audio_response, audio_output]
                    )
                
                # Tab 2: Text Input
                with gr.Tab("⌨️ Entrée Texte"):
                    gr.Markdown("""
                    ### Posez votre question par écrit
                    Tapez votre question ci-dessous et cliquez sur "Envoyer".
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            text_input = gr.Textbox(
                                label="📝 Votre question",
                                placeholder="Ex: Comment résoudre une équation du second degré ?",
                                lines=3
                            )
                            text_submit_btn = gr.Button("🚀 Envoyer", variant="primary", size="lg")
                        
                        with gr.Column(scale=1):
                            text_subject = gr.Textbox(
                                label="📚 Domaine détecté",
                                placeholder="Matière identifiée...",
                                lines=1
                            )
                            text_response = gr.Textbox(
                                label="💡 Réponse de l'IA",
                                placeholder="La réponse apparaîtra ici...",
                                lines=10
                            )
                            text_audio_output = gr.Audio(
                                label="🔊 Réponse vocale",
                                type="numpy"
                            )
                    
                    # Text processing
                    text_submit_btn.click(
                        fn=self.process_text_sync,
                        inputs=[text_input],
                        outputs=[text_subject, text_response, text_audio_output]
                    )
                    
                    # Examples
                    gr.Examples(
                        examples=[
                            ["Comment résoudre une équation du second degré ?"],
                            ["Qu'est-ce que la force de gravitation ?"],
                            ["Comment conjuguer le verbe 'to be' au présent ?"],
                            ["Explique-moi le théorème de Pythagore"],
                            ["Quelle est la troisième loi de Newton ?"],
                            ["Comment utiliser le present perfect en anglais ?"]
                        ],
                        inputs=[text_input],
                        label="💡 Questions exemples"
                    )
            
            # Footer
            gr.Markdown("""
            ---
            ### 📊 Informations
            - **STT**: Whisper (faster-whisper)
            - **LLM**: Ollama (Qwen2 1.5B)
            - **TTS**: Piper (fr_FR-siwis-medium)
            - **RAG**: ChromaDB + FAISS
            - **Framework**: Pipecat
            
            **Architecture 100% locale** - Aucune API externe utilisée
            """)
        
        self.interface = interface
        return interface
    
    def launch(self, share=True, server_port=7860):
        """
        Launch Gradio interface
        
        Args:
            share: Create public link (True for Colab)
            server_port: Port number
        """
        if not self.interface:
            self.build_interface()
        
        logger.info(f"Launching Gradio interface on port {server_port}")
        
        self.interface.launch(
            share=share,
            server_port=server_port,
            server_name="0.0.0.0",
            show_error=True,
            quiet=False
        )


def create_gradio_app(pipeline):
    """
    Create and launch Gradio app
    
    Args:
        pipeline: VoicePipeline instance
        
    Returns:
        GradioVoiceInterface instance
    """
    app = GradioVoiceInterface(pipeline)
    return app


# Example usage for Google Colab
if __name__ == "__main__":
    import sys
    sys.path.append('..')
    from pipeline.voice_pipeline import create_voice_pipeline
    
    async def main():
        # Create pipeline
        pipeline = await create_voice_pipeline(
            whisper_model="base",
            ollama_model="qwen2:1.5b",
            device="cuda"
        )
        
        # Create and launch Gradio app
        app = create_gradio_app(pipeline)
        app.build_interface()
        app.launch(share=True)
    
    # Run
    asyncio.run(main())
