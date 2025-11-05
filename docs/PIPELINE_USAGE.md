# 🚀 Guide d'Utilisation - Pipeline Pipecat

## 📋 Vue d'ensemble

Ce guide explique comment utiliser le **pipeline Pipecat** pour l'agent vocal IA, optimisé pour Google Colab.

---

## 🎯 Démarrage Rapide (Google Colab)

### Option 1 : Utiliser le Notebook Complet

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Romainmlt123/agent-vocal-ia-RAG-Agentique/blob/pipecat-local-colab/notebooks/demo_pipecat_colab.ipynb)

1. **Ouvrir le notebook** : Cliquez sur le badge ci-dessus
2. **Activer le GPU** : `Runtime → Change runtime type → GPU (T4)`
3. **Exécuter toutes les cellules** : `Runtime → Run all`
4. **Attendre** : ~10-12 minutes pour l'installation complète
5. **Utiliser l'interface** : Un lien public Gradio apparaîtra

### Option 2 : Installation Manuelle

```bash
# 1. Cloner le repository
git clone -b pipecat-local-colab https://github.com/Romainmlt123/agent-vocal-ia-RAG-Agentique.git
cd agent-vocal-ia-RAG-Agentique

# 2. Installer les dépendances
pip install -r requirements-colab.txt

# 3. Installer et démarrer Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull qwen2:1.5b

# 4. Construire les index RAG
python -m src.legacy.rag_build

# 5. Lancer l'interface
python -m src.ui.ui_gradio_pipecat
```

---

## 🏗️ Architecture du Pipeline

### Composants Pipecat

```python
from src.pipeline.voice_pipeline import create_voice_pipeline

# Créer le pipeline
pipeline = await create_voice_pipeline(
    whisper_model="base",      # Taille du modèle Whisper
    ollama_model="qwen2:1.5b", # Modèle LLM
    device="cuda",             # GPU acceleration
    rag_data_path="data"       # Chemin données RAG
)
```

### Flux de Traitement

```
┌─────────────────────────────────────────────────────────────┐
│                      PIPELINE PIPECAT                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  AudioRawFrame                                               │
│      ↓                                                       │
│  LocalSTTService (Whisper)                                   │
│      ↓                                                       │
│  TranscriptionFrame                                          │
│      ↓                                                       │
│  RAGService (Router + Retrieval)                             │
│      ↓                                                       │
│  TextFrame (with context)                                    │
│      ↓                                                       │
│  LocalLLMService (Ollama)                                    │
│      ↓                                                       │
│  TextFrame (response)                                        │
│      ↓                                                       │
│  LocalTTSService (Piper)                                     │
│      ↓                                                       │
│  TTSAudioRawFrame                                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📚 Utilisation du Pipeline

### 1. Traitement Audio

```python
import numpy as np

# Audio input (PCM 16-bit, 16kHz, mono)
audio_bytes = audio_array.tobytes()
sample_rate = 16000

# Process
result = await pipeline.process_audio(audio_bytes, sample_rate)

print(f"Transcription: {result['transcription']}")
print(f"Subject: {result['subject']}")
print(f"Response: {result['response']}")
print(f"Audio output: {len(result['audio_output'])} bytes")
```

### 2. Traitement Texte

```python
# Text input
question = "Comment résoudre une équation du second degré ?"

# Process
result = await pipeline.process_text(question)

print(f"Subject: {result['subject']}")
print(f"Response: {result['response']}")
```

### 3. Avec Interface Gradio

```python
from src.ui.ui_gradio_pipecat import create_gradio_app

# Create app
app = create_gradio_app(pipeline)

# Build and launch
app.build_interface()
app.launch(share=True, server_port=7860)
```

---

## ⚙️ Configuration

### Modèles Whisper Disponibles

| Modèle | Taille | VRAM | Latence | Qualité |
|--------|--------|------|---------|---------|
| `tiny` | 39M | 1GB | 100ms | Acceptable |
| `base` | 74M | 1GB | 200ms | **Recommandé** |
| `small` | 244M | 2GB | 400ms | Bonne |
| `medium` | 769M | 5GB | 800ms | Très bonne |
| `large` | 1550M | 10GB | 1.5s | Excellente |

**Pour Colab T4 : Utiliser `base` ou `small`**

### Modèles Ollama Recommandés

| Modèle | Taille | VRAM | Latence | Description |
|--------|--------|------|---------|-------------|
| `qwen2:1.5b` | 900MB | 2GB | 800ms | **Recommandé pour Colab** |
| `llama3.2:1b` | 700MB | 2GB | 600ms | Très rapide |
| `llama3.2:3b` | 2GB | 4GB | 1.2s | Plus précis |
| `mistral:7b` | 4GB | 8GB | 3s | Meilleure qualité (nécessite A100) |

### Variables d'Environnement

```python
# Configuration du pipeline
WHISPER_MODEL = "base"          # tiny/base/small/medium/large
OLLAMA_MODEL = "qwen2:1.5b"     # Modèle Ollama
DEVICE = "cuda"                 # cuda/cpu
RAG_DATA_PATH = "data"          # Chemin données RAG
GRADIO_PORT = 7860              # Port Gradio
```

---

## 🎨 Interface Gradio

### Fonctionnalités

#### Onglet "Entrée Vocale" 🎙️
1. Cliquer sur le microphone
2. Parler clairement
3. Cliquer sur "Traiter l'audio"
4. Attendre les résultats (1-2s)

#### Onglet "Entrée Texte" ⌨️
1. Taper une question
2. Cliquer sur "Envoyer"
3. Consulter la réponse
4. Écouter l'audio généré

### Questions Exemples

**Mathématiques** 🔢
```
- Comment résoudre une équation du second degré ?
- Explique-moi le théorème de Pythagore
- C'est quoi une fonction affine ?
```

**Physique** ⚛️
```
- Qu'est-ce que la force de gravitation ?
- Quelle est la troisième loi de Newton ?
- Comment calculer l'énergie cinétique ?
```

**Anglais** 🇬🇧
```
- Comment conjuguer le verbe 'to be' au présent ?
- Comment utiliser le present perfect ?
- Quelle est la différence entre 'make' et 'do' ?
```

---

## 📊 Performance

### Latence Mesurée (Colab T4)

| Composant | Temps | % Total |
|-----------|-------|---------|
| STT (Whisper base) | 200ms | 13% |
| RAG (retrieval + routing) | 100ms | 7% |
| LLM (Qwen2 1.5B) | 800ms | 53% |
| TTS (Piper) | 300ms | 20% |
| Overhead Pipeline | 100ms | 7% |
| **TOTAL** | **1.5s** | **100%** |

### Optimisations Appliquées

✅ **Faster-Whisper** : 2x plus rapide que Whisper standard  
✅ **Modèle LLM compact** : 1.5B paramètres au lieu de 7B+  
✅ **GPU acceleration** : Tous les modèles sur CUDA  
✅ **Streaming** : Traitement asynchrone des frames  
✅ **Piper TTS** : 3x plus rapide que Coqui/Bark  

---

## 🐛 Dépannage

### Problème : Pas de GPU détecté

```bash
# Vérifier le GPU
!nvidia-smi

# Si vide :
# Runtime → Change runtime type → GPU (T4)
```

### Problème : Ollama ne démarre pas

```bash
# Redémarrer Ollama
!pkill ollama
!ollama serve &
sleep 5
!ollama list
```

### Problème : Out of Memory (OOM)

```python
# Utiliser des modèles plus petits
pipeline = await create_voice_pipeline(
    whisper_model="tiny",     # Au lieu de "base"
    ollama_model="qwen2:1.5b" # Le plus compact
)
```

### Problème : Latence trop élevée

1. **Vérifier le GPU** : `!nvidia-smi`
2. **Réduire les modèles** : Utiliser `tiny` pour Whisper
3. **Limiter les tokens** : Réduire `max_tokens` dans LocalLLMService
4. **Désactiver le RAG** : Pour tester uniquement le LLM

---

## 🔧 Développement

### Structure des Fichiers

```
src/
├── pipeline/
│   ├── __init__.py
│   └── voice_pipeline.py          # Pipeline Pipecat principal
│
├── services/
│   ├── local_stt.py               # Whisper STT service
│   ├── local_llm.py               # Ollama LLM service
│   ├── local_tts.py               # Piper TTS service
│   └── rag_service.py             # RAG + routing service
│
└── ui/
    ├── ui_gradio_pipecat.py       # Interface Gradio pour pipeline
    ├── ui_gradio.py               # Interface Gradio legacy
    └── ui_hybrid.py               # Interface hybride
```

### Ajouter un Nouveau Service

```python
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import Frame, FrameDirection

class MyCustomService(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        
        # Process the frame
        # ...
        
        await self.push_frame(frame, direction)
```

### Modifier le Pipeline

```python
# Dans voice_pipeline.py
self.pipeline = Pipeline([
    self.stt_service,
    self.transcription_collector,
    self.rag_service,
    self.my_custom_service,    # Ajouter ici
    self.llm_service,
    self.response_collector,
    self.tts_service,
    self.audio_buffer
])
```

---

## 📖 Références

- **Pipecat Framework** : https://pipecat.ai/
- **Pipecat GitHub** : https://github.com/pipecat-ai/pipecat
- **Whisper** : https://github.com/openai/whisper
- **Faster-Whisper** : https://github.com/SYSTRAN/faster-whisper
- **Ollama** : https://ollama.com/
- **Piper TTS** : https://github.com/rhasspy/piper
- **Gradio** : https://gradio.app/

---

## 📧 Support

Pour toute question ou problème :

- **Issues GitHub** : [Créer une issue](https://github.com/Romainmlt123/agent-vocal-ia-RAG-Agentique/issues)
- **Documentation** : `docs/ARCHITECTURE.md`
- **Auteur** : Romain Mallet

---

**🎓 Projet Académique** - Intelligence Lab Agent Vocal  
**📅 Novembre 2024**
