# Models Directory

This directory contains the language models and voice models used by the agent.

## Directory Structure

```
models/
├── llm/          # Language models (GGUF format)
└── voices/       # Piper TTS voice models
```

## Language Models (LLM)

Download quantized GGUF models from HuggingFace:

### Recommended Models

**Phi-3 Mini (4K context)** - Good for Math and English:
- Model: `microsoft/Phi-3-mini-4k-instruct-gguf`
- Download: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
- File: `Phi-3-mini-4k-instruct-q4.gguf` (Q4_K_M quantization)
- Size: ~2.4 GB

**Qwen2 1.5B** - Lightweight option for Physics:
- Model: `Qwen/Qwen2-1.5B-Instruct-GGUF`
- Download: https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF
- File: `qwen2-1_5b-instruct-q4_0.gguf`
- Size: ~1 GB

### Installation

```bash
# Create directory
mkdir -p models/llm

# Download using huggingface-cli (recommended)
pip install huggingface-hub

# Phi-3
huggingface-cli download microsoft/Phi-3-mini-4k-instruct-gguf \
    Phi-3-mini-4k-instruct-q4.gguf \
    --local-dir models/llm

# Qwen2
huggingface-cli download Qwen/Qwen2-1.5B-Instruct-GGUF \
    qwen2-1_5b-instruct-q4_0.gguf \
    --local-dir models/llm
```

Or download manually from the HuggingFace model pages.

## Voice Models (TTS)

Download Piper voice models from the official releases:

### Recommended Voices

**French**: `fr_FR-siwis-medium`
- Quality: High quality female voice
- Download: https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-fr-fr-siwis-medium.tar.gz
- Files: `fr_FR-siwis-medium.onnx` + `.onnx.json`

**English**: `en_US-lessac-medium`
- Quality: High quality female voice  
- Download: https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-en-us-lessac-medium.tar.gz
- Files: `en_US-lessac-medium.onnx` + `.onnx.json`

### Installation

```bash
# Create directory
mkdir -p models/voices

# Download and extract (French)
cd models/voices
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-fr-fr-siwis-medium.tar.gz
tar -xzf voice-fr-fr-siwis-medium.tar.gz
rm voice-fr-fr-siwis-medium.tar.gz

# Download and extract (English)
wget https://github.com/rhasspy/piper/releases/download/v1.2.0/voice-en-us-lessac-medium.tar.gz
tar -xzf voice-en-us-lessac-medium.tar.gz
rm voice-en-us-lessac-medium.tar.gz

cd ../..
```

## Verify Installation

After downloading models, verify the structure:

```bash
ls -lh models/llm/
# Should show: *.gguf files

ls -lh models/voices/
# Should show: *.onnx and *.onnx.json files
```

## Update Configuration

Update `config/config.yaml` with the actual model filenames:

```yaml
llm:
  models:
    maths: "models/llm/Phi-3-mini-4k-instruct-q4.gguf"
    physique: "models/llm/qwen2-1_5b-instruct-q4_0.gguf"
    anglais: "models/llm/Phi-3-mini-4k-instruct-q4.gguf"

tts:
  voices:
    fr: "models/voices/fr_FR-siwis-medium.onnx"
    en: "models/voices/en_US-lessac-medium.onnx"
```

## Model Size Considerations

Total size with recommended models: ~6-7 GB

For minimal setup (single model + voices): ~3-4 GB

For Colab (limited storage):
- Use the smallest models (Qwen2 1.5B)
- Download at runtime in notebook
- Consider using Google Drive for persistence

## Alternative Models

If you need smaller models for resource-constrained environments:

- **TinyLlama 1.1B**: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0-GGUF
- **StableLM 2 1.6B**: https://huggingface.co/stabilityai/stablelm-2-1_6b-GGUF

## Note

**These models are not included in the repository** to comply with size constraints and licensing. Users must download them separately.
