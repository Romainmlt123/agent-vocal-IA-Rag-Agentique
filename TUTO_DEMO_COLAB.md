# 🚀 TUTORIEL RAPIDE - Lancer l'Agent Vocal sur Colab

## 📋 Prérequis (2 minutes)

1. **Compte Google** (gratuit)
2. **Connexion internet stable** (~1 GB de téléchargement)

---

## ⚡ Lancement Express (10 minutes)

### 1️⃣ Ouvrir le Notebook

**Lien direct** :
```
https://colab.research.google.com/github/Romainmlt123/agent-vocal-IA-Rag-Agentique/blob/pipecat-local-colab/notebooks/demo_simple_colab.ipynb
```

### 2️⃣ Activer le GPU **OBLIGATOIRE**

- Menu : `Exécution → Modifier le type d'exécution`
- Sélectionner : **T4 GPU**
- Cliquer : **Enregistrer**

### 3️⃣ Exécuter Toutes les Cellules

- Menu : `Exécution → Tout exécuter`
- OU raccourci : `Ctrl + F9`

**Durée** : ~10 minutes
- Installation : 3 min
- Ollama + Qwen2 : 3 min
- Whisper : 2 min
- RAG + Interface : 2 min

### 4️⃣ Cliquer sur le Lien Public

À la fin, vous verrez :
```
Running on public URL: https://xxxxx.gradio.live
```

**➡️ Cliquez dessus !**

---

## 🎮 Utiliser l'Agent

### Mode Texte (Recommandé pour tester)

1. **Onglet "💬 Mode Texte"**
2. **Tapez une question** :
   - Maths : *"Comment résoudre x² + 5x + 6 = 0 ?"*
   - Physique : *"Qu'est-ce que la deuxième loi de Newton ?"*
   - Anglais : *"Comment conjuguer to be au présent ?"*
3. **Cliquez** "🚀 Traiter Texte"
4. **Attendez** 2-3 secondes
5. **Résultats** :
   - 📝 Question comprise
   - 🎯 Domaine détecté
   - 💡 Réponse pédagogique
   - 🔊 Audio de la réponse
   - 📚 Sources utilisées

### Mode Audio (Nécessite micro)

1. **Onglet "🎙️ Mode Audio"**
2. **Autorisez le micro** (popup du navigateur)
3. **Cliquez sur le micro** pour enregistrer
4. **Posez votre question** en français
5. **Cliquez à nouveau** pour arrêter
6. **Cliquez** "🚀 Traiter Audio"
7. **Même résultats** que mode texte

---

## 📝 Questions d'Exemple

### 🧮 Maths
```
- Résous l'équation x² - 4 = 0
- Comment calculer le discriminant ?
- Quelles sont les solutions si delta est négatif ?
```

### ⚛️ Physique
```
- Énonce la première loi de Newton
- Qu'est-ce que la force en physique ?
- Comment calculer une accélération ?
```

### 🇬🇧 Anglais
```
- Conjugue to be au présent
- Comment utiliser le present continuous ?
- Quelle est la différence entre présent simple et continu ?
```

---

## ⏱️ Temps de Réponse

| Étape | Durée |
|-------|-------|
| Transcription audio (Whisper) | ~0.5s |
| Classification (Router) | ~0.1s |
| Recherche RAG (FAISS) | ~0.2s |
| Génération réponse (Ollama) | ~1-2s |
| Synthèse vocale (TTS) | ~0.3s |
| **TOTAL** | **~2-3s** |

---

## 🐛 Problèmes Courants

### ❌ "Runtime disconnected"
**Solution** : Relancer `Exécution → Tout exécuter`

### ❌ "CUDA out of memory"
**Solution** : 
```python
# Dans la cellule 4, changer le modèle :
!ollama pull qwen2:0.5b  # Plus petit (500MB au lieu de 900MB)
```

### ❌ "ModuleNotFoundError"
**Solution** : Vérifier que toutes les cellules 1-6 ont bien exécuté

### ❌ Réponse en anglais
**Solution** : L'agent peut répondre en anglais si le contexte RAG est limité. Ajouter plus de contenu dans `data/`.

### ❌ Audio ne se lit pas
**Solution** : Colab audio peut bugger - téléchargez le fichier audio pour l'écouter localement

---

## 📊 Architecture Simplifiée

```
Question Audio/Texte
    ↓
[Whisper ASR] → Transcription
    ↓
[Router] → Classification (maths/physique/anglais)
    ↓
[RAG FAISS] → Recherche contexte pertinent (top 3)
    ↓
[Ollama Qwen2] → Génération réponse pédagogique
    ↓
[TTS pyttsx3] → Synthèse vocale
    ↓
Réponse Audio + Texte
```

---

## 🎯 Fonctionnalités Clés

✅ **100% Local** - Aucun appel API externe  
✅ **Multi-domaines** - Maths, Physique, Anglais  
✅ **RAG Agentique** - Recherche sémantique dans documents  
✅ **Temps réel** - Réponse en 2-3 secondes  
✅ **Gratuit** - GPU Colab gratuit (T4)  

---

## 📚 Ressources

- **GitHub** : [agent-vocal-IA-Rag-Agentique](https://github.com/Romainmlt123/agent-vocal-IA-Rag-Agentique)
- **Branch** : `pipecat-local-colab`
- **Guide complet** : `GUIDE_UTILISATION.md`
- **Quickstart** : `QUICKSTART.md`

---

## 🎓 Pour le Jury

**Points forts à mentionner** :
1. **Architecture modulaire** - Components découplés (Router, RAG, LLM, ASR, TTS)
2. **RAG optimisé** - FAISS pour recherche vectorielle rapide
3. **Multi-domaines** - Classification automatique
4. **Open-source** - Whisper + Ollama (pas d'API payante)
5. **Temps réel** - Pipeline optimisé <3s
6. **Extensible** - Facile d'ajouter de nouveaux domaines

**Démo impressionnante** :
- Montrer transcription instantanée (Whisper)
- Montrer classification correcte (Router)
- Montrer sources RAG pertinentes
- Montrer réponse pédagogique qualitative
- Montrer audio synthétisé naturel

**Questions attendues du jury** :
- *"Pourquoi RAG ?"* → Pour contextualiser les réponses avec connaissances spécifiques
- *"Pourquoi local ?"* → Confidentialité, coût zéro, indépendance
- *"Évolutivité ?"* → Ajouter domaine = ajouter fichier .txt + rebuild index
- *"Performances ?"* → 2-3s sur GPU gratuit, <1s sur GPU dédié

---

## ✅ Checklist Avant Présentation

- [ ] GPU T4 activé
- [ ] Toutes cellules exécutées sans erreur
- [ ] Interface Gradio accessible
- [ ] Testé 1 question par domaine
- [ ] Audio fonctionne
- [ ] Sources RAG affichées

**Prêt à impressionner le jury ! 🚀**
