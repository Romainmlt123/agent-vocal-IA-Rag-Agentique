"""
Gradio UI - Interactive Push-to-Talk Voice Tutoring Interface.
"""

import gradio as gr
import numpy as np
from typing import Optional, Tuple, List
import time

from .config import get_config
from .orchestrator import get_orchestrator, SessionState
from .utils import get_logger, setup_logging


logger = get_logger(__name__)


class VoiceTutoringUI:
    """Gradio UI for voice tutoring system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize UI.
        
        Args:
            config_path: Path to config file
        """
        self.config = get_config(config_path)
        self.orchestrator = get_orchestrator(config_path)
        self.current_session = None
        self.is_recording = False
        
        logger.info("Voice Tutoring UI initialized")
    
    def start_session(self) -> Tuple[str, str, str]:
        """
        Start a new tutoring session.
        
        Returns:
            Tuple of (status, session_info, button_text)
        """
        self.current_session = self.orchestrator.create_session()
        self.is_recording = True
        
        status = "🎙️ Session started - Speak your question..."
        session_info = f"Session ID: {self.current_session.session_id}"
        button_text = "⏸️ Stop Recording"
        
        logger.info(f"Started session: {self.current_session.session_id}")
        return status, session_info, button_text
    
    def stop_session(self) -> Tuple[str, str]:
        """
        Stop the current session.
        
        Returns:
            Tuple of (status, button_text)
        """
        self.is_recording = False
        
        if self.current_session:
            status = f"⏹️ Session stopped: {self.current_session.session_id}"
        else:
            status = "⏹️ No active session"
        
        button_text = "🎙️ Start Recording"
        
        logger.info("Stopped session")
        return status, button_text
    
    def toggle_recording(self, current_button_text: str) -> Tuple[str, str, str]:
        """
        Toggle recording on/off.
        
        Args:
            current_button_text: Current button text
        
        Returns:
            Tuple of (status, session_info, new_button_text)
        """
        if "Start" in current_button_text:
            return self.start_session()
        else:
            status, button_text = self.stop_session()
            return status, "", button_text
    
    def process_audio_input(self, audio: Optional[Tuple[int, np.ndarray]]) -> Tuple[str, str, str, str, str, str, str]:
        """
        Process audio input from microphone.
        
        Args:
            audio: Tuple of (sample_rate, audio_data)
        
        Returns:
            Tuple of (status, transcript, subject, hints_l1, hints_l2, hints_l3, rag_sources)
        """
        if audio is None:
            return "⚠️ No audio recorded", "", "", "", "", "", ""
        
        try:
            sample_rate, audio_data = audio
            
            # Convert to float32 and normalize
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32) / 32768.0
            
            # Convert stereo to mono if needed
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Create session if needed
            if self.current_session is None:
                self.current_session = self.orchestrator.create_session()
            
            # Process through pipeline
            transcript = ""
            subject = ""
            hints_l1 = ""
            hints_l2 = ""
            hints_l3 = ""
            rag_sources = ""
            
            for event in self.orchestrator.process_audio(self.current_session, audio_data, sample_rate):
                if event.type == "transcript":
                    transcript = event.data
                elif event.type == "subject_detected":
                    subject = event.data["subject"].title()
                elif event.type == "rag_results":
                    # Format RAG sources
                    sources = event.data
                    if sources:
                        rag_sources = "📚 **Sources:**\n\n"
                        for i, src in enumerate(sources[:3], 1):
                            page_info = f" (page {src['page']})" if src['page'] else ""
                            rag_sources += f"{i}. **{src['source']}{page_info}** (score: {src['score']:.2f})\n"
                            rag_sources += f"   *{src['text'][:150]}...*\n\n"
                elif event.type == "hints":
                    hints_data = event.data
                    hints_l1 = hints_data.get("level1", "")
                    hints_l2 = hints_data.get("level2", "")
                    hints_l3 = hints_data.get("level3", "")
            
            status = f"✅ Audio processed for {subject}" if subject else "✅ Audio processed"
            return status, transcript, subject, hints_l1, hints_l2, hints_l3, rag_sources
        
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            return f"❌ Error: {str(e)}", "", "", "", "", "", ""
    
    def process_text_input(self, text_input: str) -> Tuple[str, str, str, str, str, str, str]:
        """
        Process text input (for testing without audio).
        
        Args:
            text_input: Text query
        
        Returns:
            Tuple of (status, transcript, subject, hints_l1, hints_l2, hints_l3, rag_sources)
        """
        if not text_input.strip():
            return "⚠️ Please enter a question", "", "", "", "", "", ""
        
        try:
            # Create session if needed
            if self.current_session is None:
                self.current_session = self.orchestrator.create_session()
            
            transcript = text_input
            subject = ""
            hints_l1 = ""
            hints_l2 = ""
            hints_l3 = ""
            rag_sources = ""
            
            for event in self.orchestrator.process_text_query(self.current_session, text_input):
                if event.type == "subject_detected":
                    subject = event.data["subject"].title()
                elif event.type == "rag_results":
                    # Format RAG sources
                    sources = event.data
                    if sources:
                        rag_sources = "📚 **Sources:**\n\n"
                        for i, src in enumerate(sources[:3], 1):
                            page_info = f" (page {src['page']})" if src['page'] else ""
                            rag_sources += f"{i}. **{src['source']}{page_info}** (score: {src['score']:.2f})\n"
                            rag_sources += f"   *{src['text'][:150]}...*\n\n"
                elif event.type == "hints":
                    hints_data = event.data
                    hints_l1 = hints_data.get("level1", "")
                    hints_l2 = hints_data.get("level2", "")
                    hints_l3 = hints_data.get("level3", "")
            
            status = f"✅ Response generated for {subject}"
            return status, transcript, subject, hints_l1, hints_l2, hints_l3, rag_sources
        
        except Exception as e:
            logger.error(f"Error processing text: {e}", exc_info=True)
            return f"❌ Error: {str(e)}", "", "", "", "", "", ""
    
    def build_interface(self) -> gr.Blocks:
        """
        Build the Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(title="Agent Vocal Prof", theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # 🎓 Agent Vocal Prof
            ### Local Voice Tutoring with RAG & Multi-Subject Support
            
            Ask questions in **Math**, **Physics**, or **English**, and receive guided hints (never direct answers!).
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎤 Input")
                    
                    # Text input (for testing)
                    text_input = gr.Textbox(
                        label="Type your question (for testing)",
                        placeholder="e.g., Comment résoudre une équation du second degré?",
                        lines=3
                    )
                    
                    submit_btn = gr.Button("🚀 Submit Question", variant="primary")
                    
                    gr.Markdown("---")
                    gr.Markdown("### 🎙️ Voice Input (Push-to-Talk)")
                    
                    audio_input = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="Record your question"
                    )
                    
                    # Push-to-talk button
                    # recording_btn = gr.Button("🎙️ Start Recording", variant="secondary")
                    
                    # status_text = gr.Textbox(label="Status", interactive=False)
                    # session_info = gr.Textbox(label="Session Info", interactive=False, visible=False)
                
                with gr.Column(scale=2):
                    gr.Markdown("### 📝 Response")
                    
                    status_output = gr.Textbox(label="Status", interactive=False)
                    
                    with gr.Row():
                        transcript_output = gr.Textbox(label="Your Question", interactive=False)
                        subject_output = gr.Textbox(label="Detected Subject", interactive=False)
                    
                    gr.Markdown("### 💡 3-Level Hint Ladder")
                    
                    with gr.Accordion("🔵 Level 1: Conceptual Hint", open=True):
                        hint1_output = gr.Textbox(
                            label="",
                            lines=3,
                            interactive=False,
                            placeholder="High-level guidance about the concept..."
                        )
                    
                    with gr.Accordion("🟡 Level 2: Strategic Hint", open=True):
                        hint2_output = gr.Textbox(
                            label="",
                            lines=3,
                            interactive=False,
                            placeholder="Approach or method to use..."
                        )
                    
                    with gr.Accordion("🟢 Level 3: Detailed Hint", open=True):
                        hint3_output = gr.Textbox(
                            label="",
                            lines=4,
                            interactive=False,
                            placeholder="Step-by-step guidance..."
                        )
                    
                    with gr.Accordion("📚 RAG Sources", open=False):
                        sources_output = gr.Markdown("")
            
            # Event handlers
            submit_btn.click(
                fn=self.process_text_input,
                inputs=[text_input],
                outputs=[
                    status_output,
                    transcript_output,
                    subject_output,
                    hint1_output,
                    hint2_output,
                    hint3_output,
                    sources_output
                ]
            )
            
            # Push-to-talk toggle
            # recording_btn.click(
            #     fn=self.toggle_recording,
            #     inputs=[recording_btn],
            #     outputs=[status_text, session_info, recording_btn]
            # )
            
            # Audio processing (when recording stops)
            audio_input.change(
                fn=self.process_audio_input,
                inputs=[audio_input],
                outputs=[
                    status_output,
                    transcript_output,
                    subject_output,
                    hint1_output,
                    hint2_output,
                    hint3_output,
                    sources_output
                ]
            )
            
            gr.Markdown("""
            ---
            ### ℹ️ About
            
            This system uses:
            - **ASR**: Faster-Whisper + Silero VAD
            - **RAG**: Sentence-Transformers + FAISS
            - **LLM**: Local models via llama-cpp-python
            - **TTS**: Piper (FR/EN)
            
            **No external APIs required** - Everything runs locally! 🔒
            """)
        
        return interface
    
    def launch(self, **kwargs):
        """
        Launch the Gradio interface.
        
        Args:
            **kwargs: Additional arguments for gr.Interface.launch()
        """
        interface = self.build_interface()
        
        # Use config settings
        launch_kwargs = {
            "server_name": self.config.ui.server_name,
            "server_port": self.config.ui.server_port,
            "share": self.config.ui.share,
            "quiet": False,
        }
        
        # Override with provided kwargs
        launch_kwargs.update(kwargs)
        
        logger.info(f"Launching UI on {launch_kwargs['server_name']}:{launch_kwargs['server_port']}")
        interface.launch(**launch_kwargs)


def main():
    """Main entry point for running the UI."""
    # Setup logging
    config = get_config()
    setup_logging(
        log_level=config.orchestrator.log_level,
        log_file=str(config.logs_dir / "ui_gradio.log")
    )
    
    # Create and launch UI
    ui = VoiceTutoringUI()
    ui.launch()


if __name__ == "__main__":
    main()
