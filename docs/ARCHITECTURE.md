# Architecture du Projet

## Vue d'ensemble

Ce projet implémente un agent vocal intelligent utilisant le framework **Pipecat** pour assurer un streaming audio en temps réel avec une latence minimale (<2s).

## Architecture Technique

### Stack Technologique

```
┌─────────────────────────────────────────────────────────────┐
│                     UTILISATEUR                              │
│              (Microphone → Interface Gradio)                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  PIPELINE PIPECAT                            │
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐ │
│  │ Whisper  │──▶│  Router  │──▶│   RAG    │──▶│ Ollama  │ │
│  │   STT    │   │ (detect  │   │ (docs +  │   │   LLM   │ │
│  │          │   │ subject) │   │ vectors) │   │         │ │
│  └──────────┘   └──────────┘   └──────────┘   └─────────┘ │
│                                                      │       │
│                                              ┌───────▼─────┐│
│                                              │  Piper TTS  ││
│                                              │  (French)   ││
│                                              └─────────────┘│
└─────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  AUDIO OUTPUT                                │
│                    (Speaker)                                 │
└─────────────────────────────────────────────────────────────┘
```

### Composants Principaux

#### 1. **LocalSTTService** (`src/services/local_stt.py`)
- **Rôle** : Reconnaissance vocale (Speech-to-Text)
- **Technologie** : Whisper via faster-whisper
- **Modèles supportés** : tiny, base, small, medium, large
- **Optimisations** :
  - GPU acceleration (CUDA)
  - VAD (Voice Activity Detection)
  - Batch processing
- **Latence moyenne** : ~200ms (base model, GPU)

#### 2. **RAGService** (`src/services/rag_service.py`)
- **Rôle** : Retrieval-Augmented Generation avec routing
- **Fonctionnalités** :
  - Détection automatique du domaine (maths, physique, anglais)
  - 3 index vectoriels séparés (FAISS)
  - Top-K retrieval (k=4)
  - Scores de pertinence
- **Latence moyenne** : ~100ms
- **Embeddings** : sentence-transformers/all-MiniLM-L6-v2

#### 3. **LocalLLMService** (`src/services/local_llm.py`)
- **Rôle** : Génération de réponses pédagogiques
- **Technologie** : Ollama (Qwen2 1.5B, Llama 3.2)
- **Streaming** : Token-by-token pour réponse progressive
- **Approche** : Socratique (guide sans donner la réponse)
- **Latence moyenne** : ~800ms (1.5B model, GPU)

#### 4. **LocalTTSService** (`src/services/local_tts.py`)
- **Rôle** : Synthèse vocale (Text-to-Speech)
- **Technologie** : Piper TTS
- **Voix** : fr_FR-siwis-medium
- **Qualité** : 22050 Hz, mono
- **Latence moyenne** : ~300ms

## Flux de Données (Pipecat Pipeline)

### Frame Processing

Pipecat utilise un système de **Frames** pour représenter différents types de données :

```python
AudioRawFrame → Whisper → TranscriptionFrame → Router
                                                   ↓
                                              TextFrame (with context)
                                                   ↓
                                              RAG Documents
                                                   ↓
                                              LLMContextFrame
                                                   ↓
                                              Ollama LLM
                                                   ↓
                                              TextFrame (response)
                                                   ↓
                                              Piper TTS
                                                   ↓
                                              AudioRawFrame → Output
```

### Exemple de Pipeline

```python
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner

# Création du pipeline
pipeline = Pipeline([
    transport.input(),        # Audio input
    stt_service,             # Whisper STT
    context_aggregator.user(), # Contexte utilisateur
    rag_service,             # RAG + Routing
    llm_service,             # Ollama LLM
    tts_service,             # Piper TTS
    transport.output(),      # Audio output
    context_aggregator.assistant()
])

# Exécution
task = PipelineTask(pipeline)
runner = PipelineRunner()
await runner.run(task)
```

## Performance

### Latence Totale (Google Colab T4)

| Phase | Latence | Pourcentage |
|-------|---------|-------------|
| STT (Whisper base) | 200ms | 13% |
| RAG (retrieval + routing) | 100ms | 7% |
| LLM (Qwen2 1.5B) | 800ms | 53% |
| TTS (Piper) | 300ms | 20% |
| Overhead Pipeline | 100ms | 7% |
| **TOTAL** | **1.5s** | **100%** |

### Optimisations Appliquées

1. **Modèles compacts** : Qwen2 1.5B au lieu de 7B+
2. **Faster-Whisper** : 2x plus rapide que Whisper standard
3. **GPU acceleration** : Tous les modèles sur CUDA
4. **Streaming** : Token-by-token pour le LLM
5. **Embeddings en cache** : Pas de recalcul
6. **VAD** : Détection silences pour éviter les traitements inutiles

## Déploiement

### Google Colab (Recommandé)

**Avantages** :
- GPU T4 gratuit
- Environnement pré-configuré
- Accès web facile
- Pas d'installation locale

**Configuration** :
```python
# Runtime → Change runtime type → GPU (T4)
# Puis exécuter le notebook demo_complete.ipynb
```

### Local (Linux/WSL)

**Prérequis** :
- GPU NVIDIA (recommandé)
- CUDA 11.8+
- 8GB RAM minimum
- 10GB espace disque

**Installation** :
```bash
pip install -r requirements-colab.txt
ollama serve &
ollama pull qwen2:1.5b
python -m src.legacy.rag_build
```

## Architecture de Code

### Structure des Dossiers

```
src/
├── services/           # Services Pipecat (NEW)
│   ├── local_stt.py   # Whisper STT
│   ├── local_llm.py   # Ollama LLM
│   ├── local_tts.py   # Piper TTS
│   └── rag_service.py # RAG + Routing
│
├── pipeline/          # Orchestration
│   └── voice_pipeline.py  # Pipeline complet
│
├── ui/               # Interfaces utilisateur
│   ├── ui_gradio.py  # Interface Gradio
│   └── ui_hybrid.py  # Interface hybride
│
└── legacy/           # Ancienne architecture (référence)
    └── [9 fichiers]  # Code non-Pipecat
```

### Design Patterns

#### 1. **FrameProcessor Pattern**
Tous les services héritent de `FrameProcessor` ou `AIService` :

```python
class LocalSTTService(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, AudioRawFrame):
            # Process audio
            text = await self.transcribe(frame.audio)
            await self.push_frame(TranscriptionFrame(text))
        else:
            await self.push_frame(frame, direction)
```

#### 2. **Async Pipeline**
Tout le traitement est asynchrone pour le streaming :

```python
async def process_audio(audio_data):
    # Non-blocking processing
    result = await async_model.transcribe(audio_data)
    return result
```

#### 3. **Context Aggregation**
Le contexte conversationnel est agrégé automatiquement :

```python
context_aggregator = LLMContextAggregatorPair(context)
# Gère automatiquement user/assistant messages
```

## Évolutions Futures

### Court Terme
- [ ] Améliorer la détection de domaine (Router)
- [ ] Ajouter plus de documents dans chaque matière
- [ ] Optimiser le prompt système pour meilleure pédagogie
- [ ] Implémenter des métriques de performance

### Moyen Terme
- [ ] Support multi-utilisateurs
- [ ] Historique de conversation persistent
- [ ] API REST pour intégration externe
- [ ] Support d'autres langues

### Long Terme
- [ ] Quantization des modèles (INT8/INT4)
- [ ] Modèles multimodaux (texte + images)
- [ ] Fine-tuning sur des données pédagogiques
- [ ] Déploiement cloud (AWS/GCP)

## Références

- **Pipecat Framework** : https://pipecat.ai/
- **Whisper** : https://github.com/openai/whisper
- **Ollama** : https://ollama.com/
- **Piper TTS** : https://github.com/rhasspy/piper
- **Faster-Whisper** : https://github.com/SYSTRAN/faster-whisper

---

**Auteur** : Romain Mallet  
**Dernière mise à jour** : Novembre 2024
