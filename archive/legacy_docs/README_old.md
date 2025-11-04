# Agent Vocal Prof 🎓🎤

> A fully local, streaming voice tutoring agent with RAG-powered multi-subject support and agentic routing to specialized small language models.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎯 Project Overview

**Agent Vocal Prof** is a professional voice tutoring system that runs 100% locally, with no external APIs or API keys required. It combines:

- **🎙️ Streaming Audio I/O**: Real-time speech recognition and text-to-speech
- **📚 Multi-Subject RAG**: Retrieval-augmented generation for Math, Physics, and English
- **🤖 Agentic Routing**: Intelligent selection of specialized small language models based on subject matter
- **🎓 Pedagogical Design**: 3-level hint ladder that guides students without giving away answers
- **🖱️ Push-to-Talk UI**: Simple Gradio interface for interactive tutoring sessions

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Gradio UI (Push-to-Talk)                  │
│  [Start/Stop] | Live Transcript | Hint Ladder | RAG Sources      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Orchestrator                                │
│  Session State Management | Event Pipeline | Error Handling      │
└──┬────────┬────────┬────────┬────────┬──────────────────────────┘
   │        │        │        │        │
   ▼        ▼        ▼        ▼        ▼
┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
│ ASR │ │ RAG │ │Route│ │ LLM │ │ TTS │
│     │ │     │ │     │ │     │ │     │
│Whisp│ │FAISS│ │TF-  │ │llama│ │Piper│
│er+  │ │sent.│ │IDF  │ │.cpp │ │     │
│VAD  │ │trans│ │key- │ │GGUF │ │FR/EN│
│     │ │form │ │words│ │     │ │     │
└─────┘ └─────┘ └─────┘ └─────┘ └─────┘
```

### Pipeline Flow

1. **Audio Input** → VAD detects speech → Faster-Whisper transcribes
2. **Transcript** → Router detects subject (math/physics/english)
3. **RAG Retrieval** → Fetch relevant context from subject-specific FAISS index
4. **LLM Generation** → Specialized model generates 3-level hint ladder
5. **Audio Output** → Piper-TTS synthesizes response in FR/EN
6. **UI Update** → Display transcript, hints, sources, and stream audio

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- (Optional) CUDA-capable GPU for faster inference

### Installation - Local (WSL/VSCode)

```bash
# Clone the repository
git clone https://github.com/your-org/intelligence_lab_agent_vocal.git
cd intelligence_lab_agent_vocal

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download models (see models/README.md)
# - LLM: Download Phi-3 or Qwen GGUF models
# - TTS: Download Piper voices for FR/EN

# Build RAG indexes
bash scripts/build_indexes.sh

# Run Gradio UI
bash scripts/run_gradio.sh
```

### Installation - Google Colab

Open and run: [`notebooks/00_setup_colab.ipynb`](notebooks/00_setup_colab.ipynb)

This notebook will:
- Install all dependencies
- Check GPU availability
- Download necessary models
- Run smoke tests
- Launch the Gradio interface

## 📁 Project Structure

```
agent-vocal-prof/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies (pinned versions)
├── CHANGELOG.md                 # Version history
├── CONTRIBUTING.md              # Contribution guidelines
├── .gitignore                   # Git ignore rules
│
├── notebooks/
│   ├── 00_setup_colab.ipynb    # Colab setup + GPU checks + smoke tests
│   └── 10_demo_pipeline.ipynb  # End-to-end demo: ASR→RAG→LLM→TTS
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Load config.yaml, environment variables
│   ├── asr.py                  # VAD + ASR streaming (Silero + Whisper)
│   ├── rag_build.py            # Document ingestion → embeddings → FAISS
│   ├── rag.py                  # Retrieve relevant passages per subject
│   ├── router.py               # Subject detection + model routing
│   ├── llm.py                  # llama-cpp wrapper with streaming
│   ├── tts.py                  # Piper-TTS for FR/EN speech synthesis
│   ├── orchestrator.py         # Full pipeline orchestration + state
│   ├── ui_gradio.py            # Gradio push-to-talk interface
│   └── utils.py                # Logging, file I/O, helpers
│
├── config/
│   └── config.yaml             # Models, paths, chunk sizes, n_ctx, etc.
│
├── data/
│   ├── maths/                  # Math PDF/TXT documents
│   ├── physique/               # Physics PDF/TXT documents
│   └── anglais/                # English PDF/TXT documents
│
├── models/
│   ├── llm/                    # GGUF model files (not committed)
│   ├── voices/                 # Piper voice models (not committed)
│   └── README.md               # Download instructions
│
├── tests/
│   ├── test_rag.py
│   ├── test_llm.py
│   ├── test_asr.py
│   ├── test_tts.py
│   └── test_orch.py
│
└── scripts/
    ├── build_indexes.sh        # Build FAISS indexes for all subjects
    └── run_gradio.sh           # Launch Gradio UI
```

## 🎓 Pedagogical Features

### 3-Level Hint Ladder

The agent never provides direct solutions. Instead, it offers progressively detailed hints:

1. **Level 1 - Conceptual Hint**: High-level guidance pointing to the relevant concept
2. **Level 2 - Strategic Hint**: Specific approach or method to use
3. **Level 3 - Detailed Hint**: Step-by-step breakdown (but still requires student to execute)

### RAG Source Traceability

All responses display:
- Source document title
- Page number
- Relevance score
- Excerpt snippet

This ensures transparency and allows students to verify information.

## 🔧 Configuration

Edit `config/config.yaml` to customize:

```yaml
# ASR settings
asr:
  model: "base"  # tiny, base, small, medium, large
  language: "fr"
  vad_threshold: 0.5

# RAG settings
rag:
  chunk_size: 512
  chunk_overlap: 50
  top_k: 4
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"

# LLM settings
llm:
  models:
    maths: "models/llm/phi-3-mini-4k-instruct-q4.gguf"
    physique: "models/llm/qwen2-1.5b-instruct-q4.gguf"
    anglais: "models/llm/phi-3-mini-4k-instruct-q4.gguf"
  n_ctx: 4096
  temperature: 0.7
  max_tokens: 512

# TTS settings
tts:
  voice_fr: "models/voices/fr_FR-siwis-medium.onnx"
  voice_en: "models/voices/en_US-lessac-medium.onnx"
  speed: 1.0
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test module
pytest tests/test_rag.py -v

# Lint code
flake8 src/ tests/
black src/ tests/ --check
```

## 📊 Performance Tips

### For Google Colab

- Use GPU runtime for faster inference
- Enable high-RAM if processing large documents
- Cache models in Google Drive to avoid re-downloading

### For Local Development

- Use quantized GGUF models (Q4_K_M recommended)
- Limit `n_ctx` to 4096 for faster generation
- Use `faster-whisper` base model for balanced speed/accuracy
- Enable CPU optimizations: `OMP_NUM_THREADS=4`

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Key areas for contribution:
- Additional subject domains
- Improved routing algorithms
- Better prompt engineering for hint generation
- UI/UX enhancements
- Documentation and examples

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with:
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Fast ASR
- [sentence-transformers](https://www.sbert.net/) - Embeddings
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - Local LLM inference
- [piper-tts](https://github.com/rhasspy/piper) - Neural TTS
- [Gradio](https://www.gradio.app/) - UI framework

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Note**: This is a fully local system with no external API dependencies. All processing happens on your machine or Colab instance.
