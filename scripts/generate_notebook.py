"""
Script pour générer le notebook de setup Colab complet.
Ce script crée un notebook Jupyter avec toutes les cellules nécessaires.
"""

import json
from pathlib import Path

# Définir les cellules du notebook
cells = [
    # Cell 1: Titre
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🚀 Setup Google Colab - Agent Vocal IA avec Pipecat\n",
            "\n",
            "Ce notebook configure un environnement Google Colab complet pour développer un agent vocal IA en temps réel avec :\n",
            "- **Pipecat** : Framework pour agents vocaux\n",
            "- **Whisper** : STT (Speech-to-Text) local\n",
            "- **Ollama** : LLM local\n",
            "- **Piper TTS** : Synthèse vocale locale\n",
            "- **RAG** : Recherche documentaire avec ChromaDB\n",
            "\n",
            "## ⚠️ Prérequis\n",
            "- Compte Google Colab\n",
            "- **GPU T4/A100** activé (Runtime > Change runtime type > GPU)\n",
            "\n",
            "---"
        ]
    },
    
    # Cell 2: Vérification GPU
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📊 Étape 1 : Vérification du GPU\n",
            "\n",
            "Vérifions que vous avez bien accès à un GPU pour accélérer l'inférence des modèles."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Vérifier la disponibilité du GPU\n",
            "!nvidia-smi\n",
            "\n",
            "import torch\n",
            "print(f\"\\n✅ PyTorch version: {torch.__version__}\")\n",
            "print(f\"✅ CUDA available: {torch.cuda.is_available()}\")\n",
            "if torch.cuda.is_available():\n",
            "    print(f\"✅ GPU: {torch.cuda.get_device_name(0)}\")\n",
            "    print(f\"✅ CUDA version: {torch.version.cuda}\")\n",
            "else:\n",
            "    print(\"⚠️  Pas de GPU détecté ! Allez dans Runtime > Change runtime type > GPU\")"
        ]
    },
    
    # Cell 3: Cloner le projet
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📦 Étape 2 : Cloner le Projet\n",
            "\n",
            "Clonons le repository GitHub contenant notre agent vocal."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Cloner le repository\n",
            "!git clone -b pipecat-local-colab https://github.com/Romainmlt123/agent-vocal-ia-RAG-Agentique.git\n",
            "%cd agent-vocal-ia-RAG-Agentique\n",
            "\n",
            "# Afficher la structure\n",
            "!ls -la"
        ]
    },
    
    # Cell 4: Installation dépendances système
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🔧 Étape 3 : Installation des Dépendances Système\n",
            "\n",
            "Installons les bibliothèques système nécessaires pour l'audio."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Mise à jour du système et installation des dépendances audio\n",
            "!apt-get update -qq\n",
            "!apt-get install -y -qq portaudio19-dev python3-pyaudio ffmpeg espeak-ng libsndfile1\n",
            "\n",
            "print(\"✅ Dépendances système installées\")"
        ]
    },
    
    # Cell 5: Installation Python packages
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📚 Étape 4 : Installation des Packages Python\n",
            "\n",
            "Installons tous les packages Python nécessaires (cela peut prendre 5-10 minutes)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Installation des packages Python depuis requirements-colab.txt\n",
            "!pip install -q -r requirements-colab.txt\n",
            "\n",
            "# Vérifier les installations critiques\n",
            "import sys\n",
            "packages_to_check = [\n",
            "    'pipecat',\n",
            "    'torch',\n",
            "    'whisper',\n",
            "    'faster_whisper',\n",
            "    'langchain',\n",
            "    'chromadb'\n",
            "]\n",
            "\n",
            "print(\"\\n📦 Vérification des packages installés:\\n\")\n",
            "for pkg in packages_to_check:\n",
            "    try:\n",
            "        __import__(pkg)\n",
            "        print(f\"✅ {pkg}\")\n",
            "    except ImportError:\n",
            "        print(f\"❌ {pkg} - ÉCHEC\")\n",
            "\n",
            "print(\"\\n✅ Installation des packages Python terminée\")"
        ]
    },
    
    # Cell 6: Installation Ollama
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🤖 Étape 5 : Installation d'Ollama\n",
            "\n",
            "Ollama permet d'exécuter des LLMs localement. Nous allons l'installer et télécharger un modèle léger."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Installer Ollama\n",
            "!curl -fsSL https://ollama.com/install.sh | sh\n",
            "\n",
            "print(\"✅ Ollama installé\")"
        ]
    },
    
    # Cell 7: Démarrer Ollama
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Démarrer le serveur Ollama en arrière-plan\n",
            "import subprocess\n",
            "import time\n",
            "\n",
            "print(\"🚀 Démarrage du serveur Ollama...\")\n",
            "ollama_process = subprocess.Popen(\n",
            "    ['ollama', 'serve'],\n",
            "    stdout=subprocess.PIPE,\n",
            "    stderr=subprocess.PIPE\n",
            ")\n",
            "\n",
            "# Attendre le démarrage\n",
            "time.sleep(5)\n",
            "print(\"✅ Serveur Ollama démarré\")"
        ]
    },
    
    # Cell 8: Télécharger modèle Ollama
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Télécharger un modèle léger (llama3.2:1b adapté à Colab T4)\n",
            "print(\"📥 Téléchargement du modèle Llama 3.2 (1B)...\")\n",
            "print(\"⏱️  Cela peut prendre 2-5 minutes selon votre connexion\\n\")\n",
            "\n",
            "!ollama pull llama3.2:1b\n",
            "\n",
            "print(\"\\n✅ Modèle téléchargé et prêt à l'emploi\")"
        ]
    },
    
    # Cell 9: Télécharger Whisper
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🎤 Étape 6 : Préparation de Whisper (STT)\n",
            "\n",
            "Téléchargeons le modèle Whisper pour la reconnaissance vocale."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Télécharger le modèle Whisper\n",
            "from faster_whisper import WhisperModel\n",
            "\n",
            "print(\"📥 Téléchargement du modèle Whisper (base)...\")\n",
            "model = WhisperModel(\"base\", device=\"cuda\", compute_type=\"float16\")\n",
            "print(\"✅ Modèle Whisper téléchargé\")\n",
            "\n",
            "# Test rapide\n",
            "print(\"\\n🧪 Test de Whisper...\")\n",
            "# Le test complet sera fait dans le prochain notebook"
        ]
    },
    
    # Cell 10: Configuration Piper TTS
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🔊 Étape 7 : Configuration de Piper TTS\n",
            "\n",
            "Piper TTS pour la synthèse vocale locale en français."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Créer le dossier pour les voix Piper\n",
            "!mkdir -p /root/.local/share/piper/voices\n",
            "\n",
            "# Télécharger une voix française\n",
            "print(\"📥 Téléchargement de la voix française Piper...\")\n",
            "!wget -q -O /root/.local/share/piper/voices/fr_FR-siwis-medium.onnx \\\n",
            "    https://github.com/rhasspy/piper/releases/download/v1.2.0/fr_FR-siwis-medium.onnx\n",
            "\n",
            "print(\"✅ Voix Piper TTS téléchargée\")"
        ]
    },
    
    # Cell 11: Build RAG indexes
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📚 Étape 8 : Construction des Index RAG\n",
            "\n",
            "Construisons les index vectoriels pour la recherche documentaire."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Importer le service RAG\n",
            "import sys\n",
            "sys.path.append('/content/agent-vocal-ia-RAG-Agentique/src')\n",
            "\n",
            "from services.rag_service import AgenticRAGService\n",
            "import asyncio\n",
            "import nest_asyncio\n",
            "nest_asyncio.apply()\n",
            "\n",
            "print(\"🔧 Initialisation du service RAG...\")\n",
            "\n",
            "# Créer le service RAG\n",
            "rag_service = AgenticRAGService(\n",
            "    base_path=\"./data\",\n",
            "    subjects=[\"maths\", \"physique\", \"anglais\"]\n",
            ")\n",
            "\n",
            "# Charger les documents\n",
            "async def load_documents():\n",
            "    await rag_service._initialize()\n",
            "    \n",
            "    # Charger chaque matière\n",
            "    for subject in [\"maths\", \"physique\", \"anglais\"]:\n",
            "        subject_path = f\"./data/{subject}\"\n",
            "        print(f\"\\n📖 Chargement des documents {subject}...\")\n",
            "        await rag_service.load_documents_from_directory(\n",
            "            subject_path,\n",
            "            subject=subject\n",
            "        )\n",
            "    \n",
            "    print(\"\\n✅ Tous les documents sont indexés !\")\n",
            "\n",
            "# Exécuter\n",
            "await load_documents()"
        ]
    },
    
    # Cell 12: Résumé
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## ✅ Configuration Terminée !\n",
            "\n",
            "### 📋 Récapitulatif\n",
            "\n",
            "Votre environnement Colab est maintenant configuré avec :\n",
            "\n",
            "- ✅ GPU détecté et PyTorch configuré\n",
            "- ✅ Pipecat framework installé\n",
            "- ✅ Whisper (base) pour la reconnaissance vocale\n",
            "- ✅ Ollama + Llama 3.2 (1B) pour le LLM\n",
            "- ✅ Piper TTS pour la synthèse vocale française\n",
            "- ✅ RAG avec ChromaDB et documents indexés\n",
            "\n",
            "### 🚀 Prochaines Étapes\n",
            "\n",
            "1. **Notebook 02** : Test des composants individuels\n",
            "2. **Notebook 03** : Demo complète de l'agent vocal\n",
            "3. **Notebook 04** : RAG agentique avancé\n",
            "\n",
            "---\n",
            "\n",
            "💡 **Astuce** : Sauvegardez votre session Colab pour éviter de recommencer l'installation !"
        ]
    },
    
    # Cell 13: Test rapide
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🧪 Test Rapide (Optionnel)\n",
            "\n",
            "Testons rapidement le LLM Ollama."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Test rapide d'Ollama\n",
            "!ollama run llama3.2:1b \"Explique-moi en une phrase ce qu'est un agent vocal IA\""
        ]
    }
]

# Créer le notebook
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        },
        "colab": {
            "provenance": [],
            "gpuType": "T4"
        },
        "accelerator": "GPU"
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

# Sauvegarder
output_path = Path("notebooks/01_setup_colab_pipecat.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"✅ Notebook créé : {output_path}")
print(f"📊 Nombre de cellules : {len(cells)}")
