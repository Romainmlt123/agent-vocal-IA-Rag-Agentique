# 🎤 Agent Vocal IA avec RAG Agentique

> Agent vocal intelligent en temps réel, 100% local, avec Retrieval-Augmented Generation multi-matières

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Pipecat](https://img.shields.io/badge/Framework-Pipecat-blue)](https://pipecat.ai/)

---

## 📋 Vue d'ensemble

**Agent Vocal IA** est un système de tutorat vocal intelligent qui fonctionne **100% localement**, sans aucune API externe. Il combine :

- 🎙️ **Reconnaissance vocale en temps réel** (Whisper)
- 🤖 **LLM local streaming** (Ollama)
- 📚 **RAG Agentique multi-matières** (Maths, Physique, Anglais)
- 🔊 **Synthèse vocale naturelle** (Piper TTS)
- ⚡ **Architecture Pipecat** pour streaming < 2s
- 🎓 **Pédagogie** : Guide l'étudiant sans donner les réponses

---

## 🏗️ Architecture

### **Pipeline de Streaming Temps Réel**

```
Microphone → Whisper STT → Router → RAG → Ollama LLM → Piper TTS → Speaker
     ↓           ↓           ↓        ↓        ↓          ↓          ↓
AudioFrame → TextFrame → Context → TextFrame → AudioFrame → Audio Output
```

### **Stack Technologique**

| Composant | Technologie | Modèle | Latence |
|-----------|------------|--------|---------|
| **STT** | Whisper (faster-whisper) | base (74M) | ~200ms |
| **Embeddings** | sentence-transformers | all-MiniLM-L6-v2 | ~50ms |
| **Vectorstore** | FAISS/ChromaDB | 3 index (par matière) | ~100ms |
| **LLM** | Ollama | Qwen2 1.5B / Llama 3.2 | ~800ms |
| **TTS** | Piper | fr_FR-siwis-medium | ~300ms |
| **Framework** | Pipecat | Pipeline asynchrone | **Total: ~1.5s** |

---

## 🚀 Démarrage Rapide

### **Option 1 : Google Colab (Recommandé pour démo)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Romainmlt123/agent-vocal-IA-Rag-Agentique/blob/pipecat-local-colab/notebooks/demo_complete.ipynb)

1. Ouvrir le notebook ci-dessus
2. Runtime → Change runtime type → **GPU (T4)**
3. Exécuter toutes les cellules (⏱️ ~10 minutes)
4. Utiliser l'interface Gradio pour tester l'agent

**Ce qui est installé automatiquement :**
- Toutes les dépendances Python
- Ollama + modèle LLM
- Whisper + Piper TTS
- Construction des index RAG
- Interface Gradio interactive

---

### **Option 2 : Installation Locale (Linux/WSL)**

```bash
# 1. Cloner le projet
git clone -b pipecat-local-colab https://github.com/Romainmlt123/agent-vocal-IA-Rag-Agentique.git
cd agent-vocal-IA-Rag-Agentique

# 2. Créer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements-colab.txt

# 4. Installer Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull qwen2:1.5b

# 5. Construire les index RAG
python -m src.legacy.rag_build

# 6. Lancer l'interface
python -m src.ui.ui_gradio
```

Ouvrir http://localhost:7860

---

## 📂 Structure du Projet

```
agent-vocal-IA-Rag-Agentique/
├── README.md                           # Ce fichier
├── requirements-colab.txt              # Dépendances pour Colab
├── requirements.txt                    # Dépendances locales
│
├── notebooks/
│   ├── demo_complete.ipynb            # 🌟 Demo complète (Colab)
│   ├── 00_setup_colab.ipynb           # Setup initial
│   └── 01_setup_colab_pipecat.ipynb   # Setup Pipecat
│
├── src/
│   ├── services/                       # Services Pipecat
│   │   ├── local_stt.py               # Whisper STT
│   │   ├── local_llm.py               # Ollama LLM
│   │   ├── local_tts.py               # Piper TTS
│   │   └── rag_service.py             # RAG + Routing
│   │
│   ├── ui/                            # Interfaces utilisateur
│   │   ├── ui_gradio.py               # Interface Gradio classique
│   │   └── ui_hybrid.py               # Interface hybride optimisée
│   │
│   └── legacy/                        # Ancienne architecture (référence)
│       ├── asr.py, llm.py, rag.py...
│       └── orchestrator.py
│
├── data/                              # Documents pour RAG
│   ├── maths/
│   │   ├── equations_second_degre.txt
│   │   └── index.faiss
│   ├── physique/
│   │   ├── mecanique_newton.txt
│   │   └── index.faiss
│   └── anglais/
│       ├── grammar_tenses.txt
│       └── index.faiss
│
├── models/                            # Modèles téléchargés (gitignore)
│   ├── llm/                           # Modèles LLM
│   └── voices/                        # Voix TTS
│
├── scripts/
│   ├── build_indexes.sh               # Construction index RAG
│   └── run_gradio.sh                  # Lancement UI
│
├── docs/                              # Documentation technique
│   ├── STREAMING_MODE_USAGE.md
│   └── STREAMING_VOICE_DESIGN.md
│
└── archive/                           # Fichiers obsolètes archivés
    ├── legacy_docs/
    └── legacy_scripts/
```

---

## 🎯 Fonctionnalités

### **1. RAG Agentique Multi-Matières**

- **Routing intelligent** : Détection automatique du domaine (maths/physique/anglais)
- **Vectorstores séparés** : Un index FAISS par matière pour une recherche optimale
- **Top-K retrieval** : Récupération des 4 documents les plus pertinents
- **Score de pertinence** : Transparence sur les sources utilisées

### **2. Streaming Audio Temps Réel**

- **Latence totale < 2s** (Colab T4 GPU)
- **Architecture asynchrone** : Traitement concurrent des frames
- **VAD (Voice Activity Detection)** : Détection automatique de la parole
- **Streaming token-by-token** : Réponse LLM progressive

### **3. 100% Local**

- **Aucune API externe** : Pas de clés OpenAI, Google, etc.
- **Données privées** : Tout reste sur votre machine/Colab
- **Offline-capable** : Fonctionne sans internet (après installation)

### **4. Interface Intuitive**

- **Gradio Web UI** : Interface moderne et réactive
- **Microphone intégré** : Enregistrement direct depuis le navigateur
- **Visualisation** : Transcription, domaine détecté, sources RAG
- **Audio playback** : Écoute de la réponse synthétisée

---

## 🔧 Configuration et Personnalisation

### **Changer le modèle LLM**

```python
# Dans src/services/local_llm.py ou notebook
llm = LocalLLMService(
    model="llama3.2:3b",  # Au lieu de qwen2:1.5b
    temperature=0.8,
    max_tokens=1024
)
```

**Modèles disponibles** :
- `qwen2:1.5b` (900MB) - Rapide, Colab T4 ✅
- `llama3.2:1b` (700MB) - Très rapide
- `llama3.2:3b` (2GB) - Plus précis
- `mistral:7b` (4GB) - Meilleure qualité (nécessite A100)

### **Changer le modèle Whisper**

```python
# Dans src/services/local_stt.py
stt = LocalSTTService(
    model_size="small",  # tiny, base, small, medium, large
    language="fr",
    device="cuda"
)
```

### **Ajouter un nouveau domaine**

1. Créer le dossier : `data/nouveau_domaine/`
2. Ajouter des documents `.txt`
3. Construire l'index : `python -m src.legacy.rag_build`
4. Mettre à jour le router dans `src/services/rag_service.py`

---

## 📊 Performance

### **Benchmarks (Google Colab T4)**

| Scénario | Latence Totale | Détails |
|----------|----------------|---------|
| **Question courte** (5 mots) | 1.2s | STT: 150ms, LLM: 600ms, TTS: 250ms |
| **Question moyenne** (15 mots) | 1.5s | STT: 200ms, LLM: 800ms, TTS: 300ms |
| **Question longue** (30 mots) | 2.3s | STT: 400ms, LLM: 1200ms, TTS: 500ms |

### **Optimisations Appliquées**

✅ Faster-Whisper au lieu de Whisper OpenAI (2x plus rapide)  
✅ Modèle LLM 1.5B au lieu de 7B+ (5x plus rapide)  
✅ Piper TTS au lieu de Coqui/Bark (3x plus rapide)  
✅ Embeddings pré-calculés et cachés  
✅ GPU acceleration partout où possible  
✅ Streaming token-by-token pour le LLM  

---

## 🐛 Dépannage

### **Colab : Pas de GPU détecté**

```python
# Vérifier le GPU
!nvidia-smi

# Si vide, changer le runtime :
# Runtime → Change runtime type → GPU (T4)
```

### **Ollama ne répond pas**

```bash
# Vérifier le statut
!pgrep ollama

# Redémarrer si nécessaire
!pkill ollama
!ollama serve &
```

### **Out of Memory (OOM)**

Réduire la taille des modèles :
```python
# Utiliser des modèles plus petits
llm = LocalLLMService(model="qwen2:1.5b")  # Au lieu de 3b/7b
stt = LocalSTTService(model_size="tiny")   # Au lieu de base/small
```

### **Latence trop élevée**

1. Vérifier que le GPU est bien utilisé
2. Réduire `max_tokens` du LLM
3. Désactiver temporairement le RAG pour tester
4. Utiliser Whisper `tiny` pour les tests

---

## 🤝 Contribution

Les contributions sont les bienvenues ! Domaines d'amélioration :

- 🚀 Optimisation de la latence
- 📚 Ajout de nouveaux domaines/matières
- 🎨 Amélioration de l'interface utilisateur
- 🧪 Tests et benchmarks
- 📖 Documentation et tutoriels

---

## 📝 Licence

MIT License - Voir [LICENSE](LICENSE)

---

## 🙏 Remerciements

Construit avec :
- [Pipecat](https://pipecat.ai/) - Framework de streaming audio
- [Ollama](https://ollama.com/) - Exécution LLM locale
- [Whisper](https://github.com/openai/whisper) - Reconnaissance vocale
- [Piper](https://github.com/rhasspy/piper) - Synthèse vocale
- [LangChain](https://python.langchain.com/) - RAG et agents
- [Gradio](https://gradio.app/) - Interface utilisateur

---

## 📧 Contact

**Projet** : Agent Vocal IA - RAG Agentique  
**Auteur** : Romain Mallet  
**GitHub** : [@Romainmlt123](https://github.com/Romainmlt123)

---

## 🎓 Utilisation Académique

Ce projet a été développé dans un cadre académique pour démontrer :
- L'intégration de LLM locaux dans des applications réelles
- L'architecture RAG agentique avec routing multi-domaines
- Le streaming audio en temps réel avec Pipecat
- Les optimisations nécessaires pour déployer sur des ressources limitées (Colab)

---

**🚀 Prêt à tester ? Lancez le notebook sur Colab !**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Romainmlt123/agent-vocal-IA-Rag-Agentique/blob/pipecat-local-colab/notebooks/demo_complete.ipynb)
