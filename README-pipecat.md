# 🎙️ Agent Vocal IA - Architecture Pipecat

> **Branche**: `pipecat-local-colab`  
> **Objectif**: Agent vocal IA en temps réel avec RAG agentique, 100% local, optimisé pour Google Colab

---

## 🎯 Vue d'ensemble

Cette version du projet utilise le framework **Pipecat** pour créer un agent vocal IA conversationnel en temps réel, capable de :

- 🎤 **Reconnaissance vocale** locale (Whisper)
- 🤖 **Génération de réponses** avec LLM local (Ollama + Llama 3.2)
- 🔊 **Synthèse vocale** locale (Piper TTS)
- 📚 **RAG Agentique** multi-matières (Maths, Physique, Anglais)
- ⚡ **Latence <2s** (objectif streaming temps réel)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Utilisateur (Microphone)                      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Pipecat Pipeline                              │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  │
│  │  STT   │→ │  RAG   │→ │  LLM   │→ │  TTS   │→ │ Audio  │  │
│  │Whisper │  │ChromaDB│  │ Ollama │  │ Piper  │  │ Out    │  │
│  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Composants Clés

#### 1. **Services Locaux** (`src/services/`)

- `local_stt.py` : Whisper (faster-whisper) pour transcription
- `local_llm.py` : Ollama client pour LLM local
- `local_tts.py` : Piper TTS pour synthèse vocale française
- `rag_service.py` : RAG avec ChromaDB + routing multi-matières

#### 2. **Pipeline Pipecat**

Utilise les `FrameProcessor` de Pipecat pour traiter les flux audio/texte :

```python
pipeline = Pipeline([
    LocalSTTService(model="base"),      # Audio → Text
    RAGService(subjects=["maths"]),     # Enrichissement contextuel
    LocalLLMService(model="llama3.2"),  # Génération réponse
    LocalTTSService(voice="fr_FR"),     # Text → Audio
])
```

#### 3. **RAG Agentique**

- **Routing intelligent** : Détecte automatiquement la matière (maths/physique/anglais)
- **Vectorstores séparés** : Un index ChromaDB par matière
- **Embeddings locaux** : sentence-transformers (all-MiniLM-L6-v2)

---

## 📂 Structure du Projet

```
agent-vocal-ia-RAG-Agentique/
├── notebooks/
│   ├── 01_setup_colab_pipecat.ipynb     # Setup Colab complet
│   ├── 02_test_components.ipynb         # Tests unitaires
│   ├── 03_full_agent_demo.ipynb         # Demo complète
│   └── 04_advanced_rag.ipynb            # RAG agentique avancé
├── src/
│   ├── services/
│   │   ├── local_stt.py                 # Whisper STT
│   │   ├── local_llm.py                 # Ollama LLM
│   │   ├── local_tts.py                 # Piper TTS
│   │   └── rag_service.py               # RAG + Routing
│   └── agents/
│       └── voice_agent.py                # Agent principal
├── data/
│   ├── maths/                            # Documents maths
│   ├── physique/                         # Documents physique
│   └── anglais/                          # Documents anglais
├── requirements-colab.txt                # Dépendances Colab
└── README-pipecat.md                     # Ce fichier
```

---

## 🚀 Utilisation sur Google Colab

### Étape 1 : Ouvrir le Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Romainmlt123/agent-vocal-ia-RAG-Agentique/blob/pipecat-local-colab/notebooks/01_setup_colab_pipecat.ipynb)

### Étape 2 : Activer le GPU

1. Menu : **Runtime** > **Change runtime type**
2. Hardware accelerator : **GPU** (T4 ou A100)
3. Cliquez sur **Save**

### Étape 3 : Exécuter le Setup

Exécutez toutes les cellules du notebook `01_setup_colab_pipecat.ipynb` :

- Installation des dépendances système
- Installation de Pipecat et packages Python
- Installation d'Ollama + téléchargement du modèle
- Téléchargement de Whisper
- Configuration de Piper TTS
- Construction des index RAG

**⏱️ Temps estimé** : 10-15 minutes

### Étape 4 : Tester l'Agent

Une fois le setup terminé, passez au notebook `03_full_agent_demo.ipynb` pour interagir avec l'agent vocal.

---

## 💻 Utilisation Locale (WSL/Linux)

Si vous voulez développer localement avec un GPU :

```bash
# Cloner le projet
git clone -b pipecat-local-colab https://github.com/Romainmlt123/agent-vocal-ia-RAG-Agentique.git
cd agent-vocal-ia-RAG-Agentique

# Créer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installer les dépendances
pip install -r requirements-colab.txt

# Installer Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:1b

# Lancer Ollama
ollama serve &

# Construire les index RAG
python scripts/build_rag_indexes.py

# Tester l'agent
python src/agents/voice_agent.py
```

---

## 📊 Performance

### Latences Mesurées (Google Colab T4)

| Composant | Latence | Modèle |
|-----------|---------|--------|
| **STT** (Whisper) | ~200ms | base (74M params) |
| **RAG** (Retrieval) | ~100ms | all-MiniLM-L6-v2 |
| **LLM** (Ollama) | ~800ms | llama3.2:1b |
| **TTS** (Piper) | ~300ms | fr_FR-siwis-medium |
| **Total** | **~1.4s** | Pipeline complet |

### Optimisations

- ✅ Utilisation de `faster-whisper` (vs whisper OpenAI)
- ✅ Modèle LLM léger (1B params au lieu de 7B+)
- ✅ Piper TTS (plus rapide que Coqui/Bark)
- ✅ Embeddings cachés en RAM
- ✅ Streaming token-by-token pour LLM
- ✅ GPU acceleration pour tous les modèles

---

## 🔧 Configuration

### Modèles Disponibles

#### STT (Whisper)
- `tiny` : 39M params, ~100ms, 60% précision
- `base` : 74M params, ~200ms, 75% précision ✅ **Recommandé**
- `small` : 244M params, ~500ms, 85% précision
- `medium` : 769M params, ~1.5s, 90% précision

#### LLM (Ollama)
- `llama3.2:1b` : 1B params, ~800ms ✅ **Recommandé Colab**
- `llama3.2:3b` : 3B params, ~2s
- `mistral:7b` : 7B params, ~5s (nécessite A100)

#### TTS (Piper)
- `fr_FR-siwis-medium` : Français naturel ✅
- `en_US-lessac-medium` : Anglais américain

### Personnalisation

Modifier les paramètres dans le notebook ou via le code :

```python
from src.services.local_stt import LocalSTTService
from src.services.local_llm import LocalLLMService
from src.services.local_tts import LocalTTSService

# Configuration personnalisée
stt = LocalSTTService(
    model_size="small",  # Changer le modèle
    language="fr",
    device="cuda"
)

llm = LocalLLMService(
    model="mistral:7b",  # Changer le modèle
    temperature=0.8,
    max_tokens=1024
)

tts = LocalTTSService(
    voice_model="en_US-lessac-medium",  # Voix anglaise
    speed=1.2
)
```

---

## 🐛 Dépannage

### Problème : Pas de GPU détecté

**Solution** :
1. Runtime > Change runtime type > GPU
2. Redémarrer le runtime
3. Vérifier avec `!nvidia-smi`

### Problème : Ollama ne répond pas

**Solution** :
```python
# Redémarrer Ollama
!pkill ollama
!ollama serve &
import time; time.sleep(5)
```

### Problème : Out of Memory (OOM)

**Solution** :
- Utiliser `llama3.2:1b` au lieu de modèles plus gros
- Réduire `max_tokens` du LLM
- Utiliser Whisper `tiny` ou `base`

### Problème : Latence trop élevée

**Optimisations** :
1. Réduire la taille des modèles
2. Activer le streaming LLM
3. Désactiver le RAG pour les tests
4. Utiliser le batch processing

---

## 📚 Documentation Complète

- [Pipecat Documentation](https://docs.pipecat.ai/)
- [Ollama Models](https://ollama.com/library)
- [Whisper GitHub](https://github.com/openai/whisper)
- [Piper TTS](https://github.com/rhasspy/piper)
- [LangChain RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)

---

## 🤝 Contribution

Cette branche est expérimentale. Les PR sont bienvenues pour :

- Améliorer la latence
- Ajouter de nouveaux modèles
- Optimiser le RAG
- Améliorer la documentation

---

## 📝 Licence

MIT License - Voir [LICENSE](../LICENSE)

---

## 🎓 Auteur

Développé dans le cadre du projet Agent Vocal IA - RAG Agentique

**Contact** : [GitHub](https://github.com/Romainmlt123)

---

**🚀 Prêt à commencer ? Ouvrez le notebook sur Colab !**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Romainmlt123/agent-vocal-ia-RAG-Agentique/blob/pipecat-local-colab/notebooks/01_setup_colab_pipecat.ipynb)
