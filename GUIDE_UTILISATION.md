# 📖 Guide d'Utilisation Complet
## Agent Vocal IA avec RAG Agentique + Pipecat

**Version** : 2.0 (Pipecat)  
**Date** : Novembre 2024  
**Auteur** : Romain Mallet

---

## 🎯 Vue d'ensemble

Ce guide vous explique **étape par étape** comment utiliser l'agent vocal IA dans Google Colab, depuis l'installation jusqu'à l'interaction vocale.

### Ce que fait l'agent

L'agent vocal est un **tuteur IA intelligent** qui :
- 🎤 **Écoute** vos questions vocales (français)
- 🧠 **Comprend** le domaine (maths, physique, anglais)
- 📚 **Recherche** dans sa base de connaissances (RAG)
- 💬 **Répond** de manière pédagogique (approche socratique)
- 🔊 **Parle** la réponse en français naturel

### Avantages de cette version Pipecat

✅ **Streaming temps réel** : Latence <2 secondes  
✅ **100% local** : Aucune API externe  
✅ **Optimisé Colab** : Fonctionne sur GPU T4 gratuit  
✅ **Architecture modulaire** : Facile à étendre  

---

## 📋 Table des Matières

1. [Prérequis](#prérequis)
2. [Installation Complète](#installation-complète)
3. [Utilisation de l'Interface Graphique](#utilisation-de-linterface-graphique)
4. [Exemples de Questions](#exemples-de-questions)
5. [Dépannage](#dépannage)
6. [Architecture Technique](#architecture-technique)
7. [FAQ](#faq)

---

## 🚀 Prérequis

### Compte Google
- Compte Google (gratuit)
- Accès à Google Colab : https://colab.research.google.com

### Aucune installation locale requise !
Tout se passe dans le navigateur grâce à Google Colab.

---

## 📦 Installation Complète

### Étape 1️⃣ : Ouvrir le Notebook

1. Allez sur GitHub : https://github.com/Romainmlt123/agent-vocal-ia-RAG-Agentique
2. Naviguez vers : `notebooks/demo_pipecat_colab.ipynb`
3. Cliquez sur le bouton **"Open in Colab"** (badge en haut du notebook)

**OU** cliquez directement sur ce lien :
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Romainmlt123/agent-vocal-ia-RAG-Agentique/blob/pipecat-local-colab/notebooks/demo_pipecat_colab.ipynb)

### Étape 2️⃣ : Activer le GPU (IMPORTANT !)

**⚠️ CETTE ÉTAPE EST CRUCIALE**

1. Dans Colab, allez dans le menu : **Runtime** → **Change runtime type**
2. Dans **Hardware accelerator**, sélectionnez **GPU**
3. Laissez **GPU type** sur **T4** (par défaut)
4. Cliquez sur **Save**

✅ Le notebook va redémarrer. C'est normal !

**Pourquoi le GPU ?**
- Whisper (STT) : 10x plus rapide sur GPU
- Ollama (LLM) : 5x plus rapide sur GPU
- Total : Latence passe de ~15s à ~1.5s

### Étape 3️⃣ : Exécuter les Cellules d'Installation

Exécutez **séquentiellement** les cellules suivantes (cliquez sur ▶️ ou `Ctrl+Enter`) :

#### Cellule 1 : Vérification GPU ⏱️ ~10 secondes
```python
# Vérifie que le GPU est bien activé
!nvidia-smi
```

**✅ Résultat attendu** : Vous devriez voir `Tesla T4` et la mémoire GPU (~15GB)

**❌ Si erreur** : Retournez à l'Étape 2️⃣ pour activer le GPU

---

#### Cellule 2 : Installation des Dépendances Python ⏱️ ~5 minutes
```python
# Installe pipecat, whisper, ollama, etc.
!pip install -q pipecat-ai[silero] faster-whisper ...
```

**✅ Résultat attendu** : Liste des packages installés avec versions

**Que se passe-t-il ?**
- Installation de Pipecat (framework de streaming)
- Faster-whisper (STT optimisé)
- ChromaDB/FAISS (bases vectorielles)
- Piper TTS (synthèse vocale)
- Gradio (interface graphique)

---

#### Cellule 3 : Installation d'Ollama ⏱️ ~3 minutes
```python
# Installe Ollama et télécharge Qwen2 1.5B
!curl -fsSL https://ollama.com/install.sh | sh
!ollama pull qwen2:1.5b
```

**✅ Résultat attendu** : 
```
✅ Ollama installé !
✅ Serveur Ollama démarré !
✅ Modèle Qwen2 1.5B prêt !
```

**Que se passe-t-il ?**
- Installation d'Ollama (serveur LLM local)
- Démarrage du serveur en arrière-plan
- Téléchargement du modèle Qwen2 1.5B (~900 MB)

**Pourquoi Qwen2 1.5B ?**
- Optimisé pour Colab T4 (8GB RAM GPU)
- Rapide (~800ms de latence)
- Performant pour le tutorat

---

#### Cellule 4 : Clonage du Repository ⏱️ ~30 secondes
```python
# Clone le code source depuis GitHub
!git clone -b pipecat-local-colab https://github.com/...
```

**✅ Résultat attendu** : `✅ Repository cloné (branche: pipecat-local-colab)`

---

#### Cellule 5 : Téléchargement des Modèles ⏱️ ~2 minutes
```python
# Télécharge Whisper et Piper
urllib.request.urlretrieve(voice_url, voice_path)
```

**✅ Résultat attendu** :
```
✅ Whisper : sera téléchargé automatiquement au premier usage
✅ Modèle vocal téléchargé
✅ Config téléchargée
```

**Modèles téléchargés** :
- Whisper base (~140 MB) - au premier usage
- Piper fr_FR-siwis-medium (~60 MB)

---

#### Cellule 6 : Construction des Index RAG ⏱️ ~1 minute
```python
# Construit les index vectoriels pour chaque matière
!python -m src.legacy.rag_build
```

**✅ Résultat attendu** :
```
✅ maths      : 45.2 KB
✅ physique   : 38.7 KB
✅ anglais    : 52.1 KB
```

**Que se passe-t-il ?**
- Lecture des documents texte (équations, lois physiques, grammaire)
- Création d'embeddings (vecteurs)
- Construction d'index FAISS pour recherche rapide

---

#### Cellule 7 : Initialisation du Pipeline Pipecat ⏱️ ~1 minute
```python
# Crée le pipeline vocal complet
pipeline = await create_voice_pipeline(...)
```

**✅ Résultat attendu** :
```
✅ Pipeline Pipecat initialisé avec succès !
📊 Configuration :
  • STT      : Whisper base (faster-whisper + CUDA)
  • LLM      : Ollama Qwen2 1.5B
  • TTS      : Piper fr_FR-siwis-medium
  • RAG      : 3 domaines (maths, physique, anglais)
  • Framework: Pipecat
  • Device   : CUDA (GPU)
🚀 Prêt pour le traitement !
```

**Que se passe-t-il ?**
- Chargement de Whisper sur GPU
- Connexion au serveur Ollama
- Chargement de Piper TTS
- Chargement des index RAG
- Construction du pipeline Pipecat avec tous les processeurs

**⏱️ Temps total d'installation : ~10-12 minutes**

---

#### Cellule 8 : Test Rapide (Optionnel) ⏱️ ~5 secondes
```python
# Test avec une question textuelle
result = await pipeline.process_text("Comment résoudre une équation du second degré ?")
```

**✅ Résultat attendu** :
```
📝 Transcription : Comment résoudre une équation du second degré ?
📚 Domaine détecté : maths
💡 Réponse : Pour résoudre une équation du second degré...
🔊 Audio généré : 245632 bytes à 22050 Hz
```

**Objectif** : Valider que tout fonctionne avant de lancer l'interface

---

#### Cellule 9 : Lancement de l'Interface Gradio ⏱️ ~10 secondes
```python
# Lance l'interface graphique
from src.ui.ui_gradio_pipecat import create_gradio_app
app = create_gradio_app(pipeline)
app.launch(share=True)
```

**✅ Résultat attendu** :
```
🚀 Lancement de l'interface...
✅ Interface Gradio en cours de démarrage...
Running on public URL: https://xxxxx.gradio.live
```

**🎉 Cliquez sur le lien public pour accéder à l'interface !**

---

## 🎨 Utilisation de l'Interface Graphique

### Vue d'ensemble de l'interface

L'interface Gradio se compose de **3 sections principales** :

```
┌─────────────────────────────────────────────────────────────┐
│  🎤 Agent Vocal IA - RAG Agentique (Pipecat)                │
├─────────────────────────────────────────────────────────────┤
│  📥 ENTRÉE (2 onglets)                                       │
│  ├─ 🎙️ Entrée Vocale : [🔴 Enregistrer]                     │
│  └─ 💬 Entrée Texte   : [Zone de saisie] [Exemples]         │
├─────────────────────────────────────────────────────────────┤
│  📤 SORTIE                                                   │
│  ├─ 📝 Transcription : "Comment résoudre..."                 │
│  ├─ 📚 Domaine       : "maths"                               │
│  ├─ 💡 Réponse       : "Pour résoudre une équation..."       │
│  └─ 🔊 Audio         : [▶️ Player audio]                     │
├─────────────────────────────────────────────────────────────┤
│  ⚙️ PARAMÈTRES (Avancés)                                     │
│  └─ [Sliders pour température, max_tokens, etc.]            │
└─────────────────────────────────────────────────────────────┘
```

### Mode 1 : Entrée Vocale 🎙️

**Pour poser une question vocalement** :

1. **Cliquez sur l'onglet "🎙️ Entrée Vocale"**

2. **Autorisez le microphone** :
   - Votre navigateur demandera l'autorisation
   - Cliquez sur **"Autoriser"** ou **"Allow"**
   - ⚠️ Si refusé, rechargez la page et réessayez

3. **Enregistrez votre question** :
   - Cliquez sur le bouton **"🔴 Enregistrer"**
   - Le bouton devient rouge : 🔴 **Enregistrement en cours**
   - Parlez clairement en français : *"Comment résoudre une équation du second degré ?"*
   - Cliquez à nouveau pour arrêter

4. **Attendez le traitement** :
   - Un spinner apparaît : ⏳ *"Traitement en cours..."*
   - Durée : **1-2 secondes** (grâce au GPU + Pipecat)

5. **Consultez les résultats** :
   - **📝 Transcription** : Votre question transcrite par Whisper
   - **📚 Domaine** : Matière détectée (maths/physique/anglais)
   - **💡 Réponse** : Réponse pédagogique du tuteur IA
   - **🔊 Audio** : Cliquez sur ▶️ pour écouter la réponse

**💡 Conseils pour un bon enregistrement** :
- ✅ Environnement calme (peu de bruit de fond)
- ✅ Parlez clairement et à vitesse normale
- ✅ Questions de 5-15 secondes (optimal)
- ❌ Évitez les questions trop longues (>30s)

---

### Mode 2 : Entrée Texte 💬

**Pour poser une question par écrit** :

1. **Cliquez sur l'onglet "💬 Entrée Texte"**

2. **Tapez votre question** dans la zone de texte :
   ```
   Comment résoudre une équation du second degré ?
   ```

3. **OU utilisez les exemples** :
   - Cliquez sur un bouton d'exemple (ex : *"Équations du 2nd degré"*)
   - La question s'insère automatiquement

4. **Cliquez sur "Envoyer"** (ou appuyez sur `Entrée`)

5. **Consultez les résultats** (comme pour l'entrée vocale)

**Avantage du mode texte** :
- Pas de problème de transcription
- Plus rapide (~100ms en moins)
- Utile pour tester ou déboguer

---

### Exemples de Questions par Domaine

#### 🔢 Mathématiques

**Questions supportées** :
- "Comment résoudre une équation du second degré ?"
- "Explique-moi le discriminant"
- "Quelles sont les formules pour les racines d'une équation ?"
- "Comment calculer x² + 5x + 6 = 0 ?"

**Documents RAG disponibles** :
- Équations du second degré (formules, discriminant, méthodes)
- Théorème de Pythagore
- Identités remarquables

**Approche pédagogique** :
L'agent guide sans donner la solution finale :
```
❓ "Comment résoudre x² + 5x + 6 = 0 ?"

💡 "Excellente question ! Pour résoudre cette équation du second degré,
    nous allons utiliser la formule du discriminant.
    
    Premièrement, identifie les coefficients : a, b et c.
    Peux-tu me dire quels sont les coefficients dans cette équation ?"
```

---

#### ⚛️ Physique

**Questions supportées** :
- "Qu'est-ce que la force de gravitation ?"
- "Explique-moi les lois de Newton"
- "Comment calculer une force ?"
- "Quelle est la troisième loi de Newton ?"

**Documents RAG disponibles** :
- Lois de Newton (3 lois)
- Mécanique classique
- Forces et mouvements

**Exemple d'interaction** :
```
❓ "Quelle est la troisième loi de Newton ?"

💡 "La troisième loi de Newton, aussi appelée principe d'action-réaction,
    stipule que : 'À toute action correspond une réaction égale et opposée.'
    
    Peux-tu me donner un exemple de cette loi dans la vie quotidienne ?"
```

---

#### 🇬🇧 Anglais

**Questions supportées** :
- "Comment conjuguer le verbe 'to be' au présent ?"
- "Explique-moi le present perfect"
- "Quelle est la différence entre present simple et present continuous ?"
- "Comment utiliser le past simple ?"

**Documents RAG disponibles** :
- Temps verbaux (present simple, present continuous, present perfect, past simple)
- Conjugaisons
- Règles de grammaire

**Exemple d'interaction** :
```
❓ "Comment utiliser le present perfect ?"

💡 "Le present perfect se forme avec 'have/has + participe passé'.
    
    Il s'utilise pour :
    1. Actions passées avec résultat présent
    2. Expériences de vie
    
    Essaie de former une phrase au present perfect avec le verbe 'visit'."
```

---

### Paramètres Avancés ⚙️

**Pour les utilisateurs avancés**, vous pouvez ajuster :

#### Temperature (0.0 - 1.0)
- **Défaut** : 0.7
- **Basse (0.3)** : Réponses plus déterministes et sûres
- **Haute (0.9)** : Réponses plus créatives et variées
- **Usage** : Gardez 0.7 pour un équilibre optimal

#### Max Tokens (50 - 500)
- **Défaut** : 150
- **Plus bas** : Réponses plus concises
- **Plus haut** : Réponses plus détaillées
- **Usage** : 150 tokens ≈ 100-120 mots en français

#### Top K (Retrieval)
- **Défaut** : 4
- **Signification** : Nombre de documents récupérés du RAG
- **Usage** : Laissez à 4 pour un bon équilibre pertinence/contexte

---

## 📊 Comprendre les Résultats

### Champs de Sortie

#### 1. 📝 Transcription
**Ce que c'est** : Votre question transcrite par Whisper

**Exemple** :
```
Comment résoudre une équation du second degré ?
```

**Utilité** :
- Vérifier que Whisper a bien compris
- Détecter les erreurs de transcription
- Ajuster votre prononciation si nécessaire

**Précision attendue** : >95% en français dans un environnement calme

---

#### 2. 📚 Domaine Détecté
**Ce que c'est** : La matière détectée automatiquement par le router

**Valeurs possibles** :
- `maths` 🔢
- `physique` ⚛️
- `anglais` 🇬🇧
- `general` (si aucun domaine ne correspond)

**Comment ça marche ?**
Le router analyse les mots-clés de la question :
- "équation", "résoudre", "discriminant" → **maths**
- "force", "Newton", "mouvement" → **physique**
- "verbe", "conjuguer", "temps" → **anglais**

**Précision attendue** : >90% sur questions claires

---

#### 3. 💡 Réponse
**Ce que c'est** : La réponse pédagogique générée par Ollama + RAG

**Caractéristiques** :
- ✅ Utilise le contexte RAG (documents pertinents)
- ✅ Approche socratique (guide plutôt que donne la réponse)
- ✅ Adaptée au niveau (explications claires)
- ✅ En français naturel

**Exemple** :
```
Pour résoudre une équation du second degré de la forme ax² + bx + c = 0,
nous devons d'abord calculer le discriminant Δ = b² - 4ac.

Ensuite, selon la valeur du discriminant :
- Si Δ > 0 : deux solutions réelles distinctes
- Si Δ = 0 : une solution réelle double
- Si Δ < 0 : pas de solution réelle

Peux-tu me dire quel est le discriminant de l'équation x² + 5x + 6 = 0 ?
```

**Longueur** : 50-200 mots selon la question

---

#### 4. 🔊 Audio de Sortie
**Ce que c'est** : La réponse synthétisée en audio par Piper TTS

**Caractéristiques** :
- 🔊 Voix féminine française naturelle (fr_FR-siwis-medium)
- 🎵 Qualité : 22050 Hz, mono
- ⏱️ Durée : ~3-10 secondes selon longueur de réponse
- 📦 Format : WAV

**Comment écouter** :
1. Cliquez sur le bouton ▶️ du player audio
2. Utilisez les contrôles (pause, volume)
3. Téléchargez si besoin (icône ⬇️)

**Qualité attendue** : Voix claire et naturelle, légèrement robotique

---

## 🔍 Dépannage

### Problème 1 : "No GPU detected"

**Symptôme** :
```
RuntimeError: CUDA not available
```

**Solution** :
1. ✅ Vérifiez que le GPU est activé : `Runtime → Change runtime type → GPU`
2. ✅ Redémarrez le runtime : `Runtime → Restart runtime`
3. ✅ Réexécutez la cellule de vérification GPU

**Vérification** :
```python
import torch
print(torch.cuda.is_available())  # Doit afficher True
```

---

### Problème 2 : "Ollama connection refused"

**Symptôme** :
```
ConnectionRefusedError: [Errno 111] Connection refused
```

**Cause** : Le serveur Ollama n'est pas démarré

**Solution** :
```python
# Redémarrez Ollama
import subprocess
ollama_process = subprocess.Popen(['ollama', 'serve'], 
                                   stdout=subprocess.DEVNULL, 
                                   stderr=subprocess.DEVNULL)
import time
time.sleep(5)
```

**Vérification** :
```bash
!ollama list  # Doit afficher qwen2:1.5b
```

---

### Problème 3 : Microphone ne fonctionne pas

**Symptôme** : Le bouton d'enregistrement ne s'active pas

**Solutions** :
1. ✅ **Autorisations navigateur** :
   - Chrome/Edge : Cliquez sur l'icône 🔒 dans la barre d'adresse
   - Activez "Microphone"
   - Rechargez la page

2. ✅ **Utilisez HTTPS** :
   - Le microphone ne fonctionne qu'en HTTPS
   - Le lien Gradio public est en HTTPS par défaut

3. ✅ **Alternative** : Utilisez le mode texte si problème persiste

---

### Problème 4 : Transcription incorrecte

**Symptôme** : Whisper transcrit mal votre question

**Causes possibles** :
- 🔇 Bruit de fond trop élevé
- 🗣️ Prononciation peu claire
- 🎤 Microphone de mauvaise qualité
- ⏱️ Question trop rapide ou trop lente

**Solutions** :
1. ✅ Enregistrez dans un environnement calme
2. ✅ Parlez clairement et à vitesse normale
3. ✅ Approchez-vous du microphone
4. ✅ Utilisez le mode texte si problème persiste

**Test** :
```python
# Testez juste la transcription
audio_file = "test.wav"
result = pipeline.stt.transcribe(audio_file)
print(result)
```

---

### Problème 5 : Latence trop élevée (>5s)

**Symptôme** : Le traitement prend plus de 5 secondes

**Causes possibles** :
- ❌ GPU non activé
- 🐌 Modèle trop gros (medium/large au lieu de base)
- 💾 RAM saturée

**Solutions** :
1. ✅ Vérifiez le GPU : `!nvidia-smi`
2. ✅ Utilisez `whisper_model="base"` (pas medium ou large)
3. ✅ Redémarrez le runtime pour libérer la RAM
4. ✅ Vérifiez que CUDA est utilisé :
   ```python
   print(pipeline.stt.device)  # Doit afficher "cuda"
   ```

**Latence normale** :
- STT : 200ms
- RAG : 100ms
- LLM : 800ms
- TTS : 300ms
- **Total : ~1.4s**

---

### Problème 6 : "Out of Memory"

**Symptôme** :
```
RuntimeError: CUDA out of memory
```

**Cause** : Le GPU n'a plus de mémoire disponible

**Solutions** :
1. ✅ **Redémarrez le runtime** : `Runtime → Restart runtime`
2. ✅ **Utilisez des modèles plus petits** :
   ```python
   whisper_model="tiny"  # Au lieu de "base"
   ollama_model="qwen2:1.5b"  # Plus petit que 3b/7b
   ```
3. ✅ **Libérez la mémoire GPU** :
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

**Vérification mémoire GPU** :
```bash
!nvidia-smi
# Regardez "Memory-Usage" (doit être < 14GB sur 15GB)
```

---

## 🏗️ Architecture Technique

### Pipeline Pipecat

```
┌─────────────────────────────────────────────────────────────────┐
│                       ENTRÉE UTILISATEUR                        │
│                    (Audio WAV ou Texte)                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │   LocalSTTService (Whisper)            │
        │   • Modèle : base (74M params)         │
        │   • Device : CUDA                       │
        │   • Latence : ~200ms                    │
        └────────────┬───────────────────────────┘
                     │ TranscriptionFrame
                     ▼
        ┌────────────────────────────────────────┐
        │   TranscriptionCollector               │
        │   • Collecte la transcription           │
        └────────────┬───────────────────────────┘
                     │ TextFrame
                     ▼
        ┌────────────────────────────────────────┐
        │   RAGService (Router + Retrieval)      │
        │   • Détection domaine (maths/phys/eng) │
        │   • Retrieval top-4 docs FAISS          │
        │   • Latence : ~100ms                    │
        └────────────┬───────────────────────────┘
                     │ TextFrame + Context
                     ▼
        ┌────────────────────────────────────────┐
        │   LocalLLMService (Ollama)             │
        │   • Modèle : Qwen2 1.5B                │
        │   • Streaming : token-by-token          │
        │   • Latence : ~800ms                    │
        └────────────┬───────────────────────────┘
                     │ TextFrame (response)
                     ▼
        ┌────────────────────────────────────────┐
        │   ResponseCollector                     │
        │   • Collecte la réponse complète        │
        └────────────┬───────────────────────────┘
                     │ TextFrame
                     ▼
        ┌────────────────────────────────────────┐
        │   LocalTTSService (Piper)              │
        │   • Voix : fr_FR-siwis-medium          │
        │   • Sample rate : 22050 Hz              │
        │   • Latence : ~300ms                    │
        └────────────┬───────────────────────────┘
                     │ AudioRawFrame
                     ▼
        ┌────────────────────────────────────────┐
        │   AudioBufferProcessor                  │
        │   • Collecte l'audio complet            │
        └────────────┬───────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                       SORTIE UTILISATEUR                         │
│         (Transcription + Domaine + Réponse + Audio)             │
└─────────────────────────────────────────────────────────────────┘
```

### Types de Frames Pipecat

| Frame Type | Description | Données |
|------------|-------------|---------|
| `AudioRawFrame` | Audio brut | bytes, sample_rate, channels |
| `TranscriptionFrame` | Transcription STT | text, timestamp |
| `TextFrame` | Texte générique | text |
| `LLMMessagesFrame` | Messages LLM | messages (list) |
| `TTSAudioRawFrame` | Audio TTS | bytes, sample_rate |

### Flux Asynchrone

Pipecat traite les frames de manière **asynchrone** :

```python
async def process_frame(self, frame: Frame, direction: FrameDirection):
    # Chaque processeur traite la frame
    if isinstance(frame, AudioRawFrame):
        # Traitement audio
        result = await self.transcribe(frame.audio)
        await self.push_frame(TranscriptionFrame(result))
    else:
        # Passe la frame au suivant
        await self.push_frame(frame, direction)
```

**Avantages** :
- ✅ Pas de blocage
- ✅ Traitement concurrent possible
- ✅ Latence minimale

---

## ❓ FAQ

### Q1 : Combien de temps prend une requête complète ?

**R** : En moyenne **1.4-2 secondes** sur Colab T4 :
- STT (Whisper) : 200ms
- RAG : 100ms
- LLM (Qwen2) : 800ms
- TTS (Piper) : 300ms
- Overhead : 100ms

**Facteurs d'influence** :
- Longueur de la question (plus long = plus lent)
- Longueur de la réponse (plus long = plus lent)
- Charge du serveur Colab

---

### Q2 : Puis-je utiliser un autre modèle LLM ?

**R** : Oui ! Modifiez la cellule d'initialisation :

```python
pipeline = await create_voice_pipeline(
    ollama_model="llama3.2:1b",  # Alternatives: llama3.2:3b, mistral:7b
    ...
)
```

**Modèles recommandés pour Colab T4** :
- `qwen2:1.5b` ⭐ (recommandé, 900MB)
- `llama3.2:1b` (plus rapide, 700MB)
- `llama3.2:3b` (plus précis, 2GB)
- ❌ `llama3.2:7b` (trop gros pour T4, nécessite A100)

---

### Q3 : Puis-je ajouter de nouveaux domaines ?

**R** : Oui ! Créez un nouveau dossier dans `data/` :

1. Créez `data/nouveau_domaine/`
2. Ajoutez vos documents `.txt`
3. Reconstruisez les index :
   ```bash
   !python -m src.legacy.rag_build
   ```
4. Le router détectera automatiquement le nouveau domaine

---

### Q4 : L'agent fonctionne-t-il hors ligne ?

**R** : **Après installation, presque** :
- ✅ STT (Whisper) : 100% local
- ✅ LLM (Ollama) : 100% local
- ✅ TTS (Piper) : 100% local
- ✅ RAG : 100% local
- ❌ **MAIS** : Colab nécessite internet pour démarrer

**Pour usage 100% offline** : Installez localement sur votre PC

---

### Q5 : Puis-je utiliser ce projet en dehors de Colab ?

**R** : Oui ! Installez localement :

```bash
git clone -b pipecat-local-colab https://github.com/Romainmlt123/agent-vocal-ia-RAG-Agentique.git
cd agent-vocal-ia-RAG-Agentique
pip install -r requirements-colab.txt
ollama serve &
ollama pull qwen2:1.5b
python -m src.ui.ui_gradio_pipecat
```

**Prérequis** :
- GPU NVIDIA (CUDA 11.8+)
- 8GB RAM GPU minimum
- 16GB RAM système
- Linux/WSL2

---

### Q6 : Combien coûte l'utilisation sur Colab ?

**R** : **GRATUIT** avec Colab (GPU T4 gratuit) !

**Limites Colab gratuit** :
- ⏱️ Session max : 12 heures
- 💾 Stockage temporaire (perdu après session)
- 🚫 Pas de garantie GPU disponible (pic d'affluence)

**Colab Pro** (10€/mois) :
- ⏱️ Sessions plus longues (24h)
- 🚀 GPU plus puissants (V100, A100)
- ✅ Priorité sur GPU disponibles

---

### Q7 : Les données sont-elles privées ?

**R** : **OUI** à 100% :
- ✅ Tout local (aucune API externe)
- ✅ Pas d'envoi de données
- ✅ Aucun tracking
- ✅ Code open-source

**⚠️ Exception** : Dans Colab, Google peut voir les données dans la VM

---

### Q8 : Puis-je contribuer au projet ?

**R** : **Absolument !** Le projet est open-source :

1. Forkez le repo GitHub
2. Créez une branche : `git checkout -b feature/ma-fonctionnalite`
3. Faites vos modifications
4. Pushez : `git push origin feature/ma-fonctionnalite`
5. Ouvrez une Pull Request

**Contributions bienvenues** :
- 📚 Ajout de documents RAG
- 🌍 Support d'autres langues
- 🎨 Amélioration de l'interface
- 🐛 Corrections de bugs
- 📖 Documentation

---

## 📞 Support

### Problèmes ou Questions ?

1. **GitHub Issues** : https://github.com/Romainmlt123/agent-vocal-ia-RAG-Agentique/issues
2. **Documentation** : `docs/ARCHITECTURE.md`
3. **Email** : [Votre email si applicable]

---

## 📜 Licence

Ce projet est sous licence **MIT** - voir le fichier [LICENSE](../LICENSE)

---

## 🙏 Remerciements

- **Pipecat** : Framework de streaming audio
- **OpenAI** : Whisper (modèle STT)
- **Ollama** : Exécution LLM locale
- **Rhasspy** : Piper TTS
- **Hugging Face** : Modèles embeddings

---

## 🎓 Citation

Si vous utilisez ce projet dans un contexte académique :

```bibtex
@misc{agent-vocal-rag-pipecat,
  author = {Romain Mallet},
  title = {Agent Vocal IA avec RAG Agentique + Pipecat},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Romainmlt123/agent-vocal-ia-RAG-Agentique}
}
```

---

**🎉 Bon apprentissage avec votre agent vocal IA ! 🎉**

**📅 Dernière mise à jour** : Novembre 2024  
**✍️ Auteur** : Romain Mallet  
**🔗 GitHub** : [@Romainmlt123](https://github.com/Romainmlt123)
