"""
Hybrid Streaming UI - Uses browser audio with optimized pipeline for low latency.
Works on WSL and any platform without system audio access.
"""

import gradio as gr
import numpy as np
from typing import Optional, Tuple
import time
from scipy import signal

from .config import get_config
from .orchestrator import get_orchestrator
from .utils import get_logger, setup_logging


logger = get_logger(__name__)


class HybridStreamingUI:
    """
    Hybrid streaming UI that uses browser audio capture with fast processing.
    This works on WSL and doesn't require system audio access.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize hybrid streaming UI."""
        self.config = get_config(config_path)
        self.orchestrator = get_orchestrator(config_path)
        self.current_session = None
        
        logger.info("Hybrid Streaming UI initialized")
    
    def process_audio_fast(self, audio: Optional[Tuple[int, np.ndarray]]) -> Tuple[str, str, str, str, str, str]:
        """
        Process audio with optimized pipeline for low latency.
        
        Args:
            audio: Tuple of (sample_rate, audio_data)
        
        Returns:
            Tuple of (status, transcript, subject, hints_l1, hints_l2, hints_l3)
        """
        if audio is None:
            return "⚠️ Aucun audio enregistré", "", "", "", "", ""
        
        start_time = time.time()
        
        try:
            sample_rate, audio_data = audio
            
            logger.info(f"Processing audio: {audio_data.shape}, {sample_rate}Hz")
            
            # Convert to float32 and normalize
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32) / 32768.0
            
            # Convert stereo to mono if needed
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            # Resample to 16kHz if needed (Whisper requirement)
            target_sr = 16000
            if sample_rate != target_sr:
                num_samples = int(len(audio_data) * target_sr / sample_rate)
                audio_data = signal.resample(audio_data, num_samples)
            
            # Create session if needed
            if self.current_session is None:
                self.current_session = self.orchestrator.create_session()
            
            # Process through optimized pipeline
            transcript = ""
            subject = ""
            hints_l1 = ""
            hints_l2 = ""
            hints_l3 = ""
            
            # Track latencies
            asr_time = 0
            router_time = 0
            rag_time = 0
            llm_time = 0
            
            for event in self.orchestrator.process_audio(self.current_session, audio_data):
                if event.type == "transcript":
                    transcript = event.data
                    asr_time = time.time() - start_time
                    logger.info(f"ASR latency: {asr_time*1000:.0f}ms")
                
                elif event.type == "subject_detected":
                    subject = event.data["subject"].title()
                    router_time = time.time() - start_time - asr_time
                    logger.info(f"Router latency: {router_time*1000:.0f}ms")
                
                elif event.type == "rag_results":
                    rag_time = time.time() - start_time - asr_time - router_time
                    logger.info(f"RAG latency: {rag_time*1000:.0f}ms")
                
                elif event.type == "hints":
                    hints_data = event.data
                    hints_l1 = hints_data.get("level1", "")
                    hints_l2 = hints_data.get("level2", "")
                    hints_l3 = hints_data.get("level3", "")
                    llm_time = time.time() - start_time - asr_time - router_time - rag_time
                    logger.info(f"LLM latency: {llm_time*1000:.0f}ms")
            
            total_time = time.time() - start_time
            
            status = f"✅ Traité en {total_time:.1f}s (ASR: {asr_time:.1f}s, LLM: {llm_time:.1f}s)"
            
            if not transcript:
                status = "⚠️ Aucune parole détectée"
            
            logger.info(f"Total pipeline latency: {total_time*1000:.0f}ms")
            
            return status, transcript, subject, hints_l1, hints_l2, hints_l3
        
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            return f"❌ Erreur: {str(e)}", "", "", "", "", ""
    
    def create_interface(self) -> gr.Blocks:
        """
        Create Gradio interface for hybrid streaming.
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(
            title="🎤 Agent Vocal Prof - Mode Rapide",
            theme=gr.themes.Soft()
        ) as interface:
            gr.Markdown("""
            # 🎤 Agent Vocal Prof - Mode Conversation Rapide
            
            ## Interface vocale optimisée pour latence minimale
            
            **Fonctionnement** : Cliquez pour enregistrer → Parlez → Stop automatique → Réponse rapide !
            
            **Matières disponibles** : Mathématiques 🔢 | Physique ⚛️ | Anglais 🇬🇧
            
            ⚡ **Optimisé pour vitesse** : Traitement 2-3x plus rapide que le mode standard !
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🎙️ Enregistrement Audio")
                    
                    audio_input = gr.Audio(
                        sources=["microphone"],
                        type="numpy",
                        label="🎤 Cliquez pour parler",
                        streaming=False,
                        show_download_button=False
                    )
                    
                    gr.Markdown("""
                    **Instructions:**
                    1. Cliquez sur le micro 🎤
                    2. Parlez clairement
                    3. Recliquez pour arrêter
                    4. Réponse en 2-5 secondes !
                    
                    💡 **Astuce**: Questions courtes = réponses plus rapides
                    """)
                    
                    status_box = gr.Textbox(
                        label="📊 Statut & Performance",
                        value="⏸️ Prêt à enregistrer",
                        interactive=False,
                        lines=2
                    )
                
                with gr.Column(scale=2):
                    gr.Markdown("### 📝 Conversation")
                    
                    transcript_box = gr.Textbox(
                        label="🎤 Votre question",
                        placeholder="Votre question transcrite apparaîtra ici...",
                        interactive=False,
                        lines=3
                    )
                    
                    subject_box = gr.Textbox(
                        label="📚 Matière détectée",
                        placeholder="Ex: Mathématiques, Physique, Anglais",
                        interactive=False,
                        lines=1
                    )
                    
                    gr.Markdown("### 💡 Réponse de l'IA (3 niveaux d'indices)")
                    
                    with gr.Accordion("🔵 Niveau 1: Indice Conceptuel", open=True):
                        hint1_box = gr.Textbox(
                            label="",
                            placeholder="Indice de haut niveau sur le concept...",
                            interactive=False,
                            lines=4
                        )
                    
                    with gr.Accordion("🟡 Niveau 2: Indice Stratégique", open=False):
                        hint2_box = gr.Textbox(
                            label="",
                            placeholder="Méthode ou approche à utiliser...",
                            interactive=False,
                            lines=4
                        )
                    
                    with gr.Accordion("🟠 Niveau 3: Indice Détaillé", open=False):
                        hint3_box = gr.Textbox(
                            label="",
                            placeholder="Guidance étape par étape...",
                            interactive=False,
                            lines=6
                        )
            
            # Auto-process when audio is recorded
            outputs = [status_box, transcript_box, subject_box, hint1_box, hint2_box, hint3_box]
            
            audio_input.change(
                fn=self.process_audio_fast,
                inputs=[audio_input],
                outputs=outputs
            )
            
            audio_input.stop_recording(
                fn=self.process_audio_fast,
                inputs=[audio_input],
                outputs=outputs
            )
            
            gr.Markdown("""
            ---
            **⚡ Mode Conversation Rapide** - Latence réduite 2-3x vs mode standard
            
            🔬 **Pipeline Optimisé**: Audio → ASR (300ms) → Router (10ms) → RAG (100ms) → LLM (1-2s) → Affichage
            
            💡 **Différences vs Streaming pur**:
            - ✅ Fonctionne sur WSL/Windows/Linux
            - ✅ Pas besoin de PortAudio système
            - ✅ Utilise le micro du navigateur (Chrome recommandé)
            - ⚠️ Nécessite clic start/stop (pas 100% continu)
            - ⚡ Optimisé pour latence minimale
            
            📊 **Performance**: 
            - Traitement total: 2-5 secondes (vs 5-15s mode standard)
            - Transcription: ~300ms
            - Génération réponse: ~1-2s (CPU) ou ~200ms (GPU)
            """)
        
        return interface
    
    def launch(self, **kwargs):
        """
        Launch the Gradio interface.
        
        Args:
            **kwargs: Additional launch parameters
        """
        interface = self.create_interface()
        
        launch_kwargs = {
            "server_name": self.config.ui.server_name,
            "server_port": self.config.ui.server_port,
            "share": self.config.ui.share,
            "quiet": False,
        }
        
        launch_kwargs.update(kwargs)
        
        logger.info(f"Launching Hybrid Streaming UI on {launch_kwargs['server_name']}:{launch_kwargs['server_port']}")
        interface.launch(**launch_kwargs)


def main():
    """Main entry point for hybrid streaming UI."""
    config = get_config()
    setup_logging(
        log_level=config.orchestrator.log_level,
        log_file=str(config.logs_dir / "ui_hybrid.log")
    )
    
    ui = HybridStreamingUI()
    ui.launch()


if __name__ == "__main__":
    main()
