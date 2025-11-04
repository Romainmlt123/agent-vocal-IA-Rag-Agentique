# Quick Start Guide - Agent Vocal Prof

This guide will get you up and running in 15 minutes.

## Prerequisites

- Python 3.10 or higher
- 8GB RAM minimum (16GB recommended)
- (Optional) NVIDIA GPU for faster inference

## Installation Steps

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/your-org/intelligence_lab_agent_vocal.git
cd intelligence_lab_agent_vocal

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Models

#### Language Models

```bash
# Install HuggingFace CLI
pip install huggingface-hub

# Download Qwen2 1.5B (recommended for beginners - smaller, faster)
huggingface-cli download Qwen/Qwen2-1.5B-Instruct-GGUF \
    qwen2-1_5b-instruct-q4_0.gguf \
    --local-dir models/llm

# OR download Phi-3 Mini (better quality, larger)
huggingface-cli download microsoft/Phi-3-mini-4k-instruct-gguf \
    Phi-3-mini-4k-instruct-q4.gguf \
    --local-dir models/llm
```

#### TTS Voices (Optional)

```bash
# French voice
cd models/voices
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-fr-fr-siwis-medium.tar.gz
tar -xzf voice-fr-fr-siwis-medium.tar.gz

# English voice
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-en-us-lessac-medium.tar.gz
tar -xzf voice-en-us-lessac-medium.tar.gz

cd ../..
```

### 3. Update Configuration

Edit `config/config.yaml` to point to your downloaded models:

```yaml
llm:
  models:
    maths: "models/llm/qwen2-1_5b-instruct-q4_0.gguf"
    physique: "models/llm/qwen2-1_5b-instruct-q4_0.gguf"
    anglais: "models/llm/qwen2-1_5b-instruct-q4_0.gguf"
```

### 4. Build RAG Indexes

```bash
# Build FAISS indexes from sample documents
bash scripts/build_indexes.sh
```

Expected output:
```
================================
Building RAG Indexes
================================

Running RAG index builder...
Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
Embedding model loaded successfully
Building index for subject: maths
...

================================
Index Building Complete
================================
```

### 5. Run the Application

```bash
# Launch Gradio UI
bash scripts/run_gradio.sh
```

The interface will open at: http://localhost:7860

## First Test

1. Open http://localhost:7860 in your browser
2. Type a question in the text box, for example:
   - **Math**: "Comment résoudre une équation du second degré?"
   - **Physics**: "Quelle est la formule de l'énergie cinétique?"
   - **English**: "How do I use the present perfect tense?"
3. Click "Submit Question"
4. View the 3-level hint ladder response
5. Check the "RAG Sources" accordion to see where information came from

## Google Colab Setup

For a ready-to-run Colab experience:

1. Open `notebooks/00_setup_colab.ipynb` in Google Colab
2. Run all cells in sequence
3. The notebook will:
   - Install dependencies
   - Download models
   - Build indexes
   - Launch the UI with a public link

## Troubleshooting

### "Model file not found"

Make sure you've downloaded the models and updated `config/config.yaml` with the correct paths.

```bash
ls -lh models/llm/  # Should show *.gguf files
```

### "No RAG indexes found"

Build the indexes first:

```bash
bash scripts/build_indexes.sh
```

### "Out of memory"

Try these settings in `config/config.yaml`:

```yaml
llm:
  n_ctx: 2048  # Reduce from 4096
  max_tokens: 256  # Reduce from 512
```

Use smaller models like Qwen2 1.5B instead of Phi-3.

### Slow generation

- **Use GPU**: Install CUDA-enabled PyTorch
- **Reduce context**: Lower `n_ctx` in config
- **Smaller model**: Use TinyLlama or Qwen2 1.5B

### Import errors

Make sure you're in the virtual environment:

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Next Steps

### Add Your Own Documents

1. Add PDF or TXT files to:
   - `data/maths/` - Math documents
   - `data/physique/` - Physics documents
   - `data/anglais/` - English documents

2. Rebuild indexes:
   ```bash
   bash scripts/build_indexes.sh
   ```

### Customize Prompts

Edit the system prompt in `config/config.yaml`:

```yaml
llm:
  system_prompt: |
    Your custom instructions here...
```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=src tests/
```

## Architecture Overview

```
User Question (text/audio)
    ↓
[ASR] Transcribe speech → text
    ↓
[Router] Detect subject → route to model
    ↓
[RAG] Retrieve relevant passages
    ↓
[LLM] Generate 3-level hints
    ↓
[TTS] Synthesize audio response
    ↓
Display: Transcript + Hints + Sources + Audio
```

## Key Features

✅ **100% Local** - No API keys, no external services  
✅ **Multi-Subject** - Math, Physics, English (extensible)  
✅ **RAG-Powered** - Grounded responses from your documents  
✅ **Pedagogical** - 3-level hint ladder, never direct answers  
✅ **Streaming** - Real-time audio and text generation  
✅ **Push-to-Talk** - Simple voice interaction  
✅ **Bilingual** - French and English support

## Performance Expectations

| Component | Time (CPU) | Time (GPU) |
|-----------|------------|------------|
| ASR (1s audio) | 0.2s | 0.1s |
| Router | <0.001s | <0.001s |
| RAG retrieval | 0.05s | 0.02s |
| LLM generation | 5-15s | 1-3s |
| TTS (1 sentence) | 0.5s | 0.2s |

**Total pipeline:** ~6-16s on CPU, ~1-4s on GPU

## Documentation

- **README.md** - Full project documentation
- **CONTRIBUTING.md** - How to contribute
- **notebooks/00_setup_colab.ipynb** - Colab setup guide
- **notebooks/10_demo_pipeline.ipynb** - Component demos
- **models/README.md** - Model download instructions

## Support

- **Issues**: https://github.com/your-org/intelligence_lab_agent_vocal/issues
- **Discussions**: https://github.com/your-org/intelligence_lab_agent_vocal/discussions

## License

MIT License - see LICENSE file for details.

---

**Happy tutoring! 🎓**
