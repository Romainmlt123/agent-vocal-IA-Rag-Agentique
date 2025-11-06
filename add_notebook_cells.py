#!/usr/bin/env python3
"""
Script pour ajouter les nouvelles cellules au notebook
"""
import json

# Load notebook
with open('notebooks/demo_pipecat_colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"📖 Notebook chargé: {len(nb['cells'])} cellules")

# New title cell
new_title_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# 🎙️ Agent Vocal IA Local - Mode Temps Réel avec RAG\n",
        "\n",
        "**Nouvelle Architecture Inspirée de Pipecat simple-chatbot**\n",
        "\n",
        "## ✨ Fonctionnalités\n",
        "\n",
        "- 🎤 **Conversation Continue** : Discussion en temps réel jusqu'à déconnexion\n",
        "- 🔊 **VAD Silero** : Détection automatique de la voix\n",
        "- 🧠 **100% Local** : Whisper + Ollama + Piper + FAISS\n",
        "- 📚 **RAG Agentique** : 3 domaines (maths, physique, anglais)\n",
        "- 🔄 **Streaming** : Réponses en temps réel\n",
        "\n",
        "## 📋 Architecture\n",
        "\n",
        "```\n",
        "Audio Input (Micro)\n",
        "    ↓\n",
        "VAD (Voice Activity Detection)\n",
        "    ↓\n",
        "STT (Whisper) → Transcription\n",
        "    ↓\n",
        "RAG (FAISS) → Contexte pertinent\n",
        "    ↓\n",
        "LLM (Ollama) → Génération réponse\n",
        "    ↓\n",
        "TTS (Piper) → Audio Output\n",
        "    ↓\n",
        "Loop → Retour à l'écoute VAD\n",
        "```\n",
        "\n",
        "## 🚀 Utilisation\n",
        "\n",
        "1. **Exécuter toutes les cellules** (Runtime > Run all)\n",
        "2. **Test simple (Cellule 16)** : Test agent temps réel mode texte\n",
        "3. **Interface complète (Cellule 17)** : Interface Gradio (ancienne version qui bug)\n",
        "\n",
        "---"
    ]
}

# New test cell markdown
new_test_md_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 🆕 Étape 9 : Test Agent Temps Réel - Mode Texte\n",
        "\n",
        "Test de la nouvelle architecture temps réel (sans audio pour débuter)"
    ]
}

# New test code cell
new_test_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "print(\"🎯 Test Agent Temps Réel - Mode Texte\")\n",
        "print(\"=\" * 70)\n",
        "\n",
        "from src.realtime_voice_agent import create_realtime_voice_agent\n",
        "\n",
        "# Create realtime agent\n",
        "print(\"\\n⏳ Création de l'agent temps réel...\\n\")\n",
        "\n",
        "agent = await create_realtime_voice_agent(\n",
        "    whisper_model=\"base\",\n",
        "    ollama_model=\"qwen2:1.5b\",\n",
        "    device=\"cuda\",\n",
        "    rag_data_path=\"data\"\n",
        ")\n",
        "\n",
        "print(\"\\n✅ Agent temps réel créé!\")\n",
        "print(\"\\n📝 Test avec une question...\")\n",
        "\n",
        "# Test question\n",
        "test_question = \"Qu'est-ce que la force de gravitation ?\"\n",
        "print(f\"\\n❓ Question: {test_question}\\n\")\n",
        "\n",
        "# Get RAG context\n",
        "subject, context = agent.rag_service.retrieve(test_question)\n",
        "print(f\"📚 Domaine détecté: {subject}\")\n",
        "print(f\"📄 Contexte RAG: {context[:200]}...\\n\")\n",
        "\n",
        "# Build prompt\n",
        "system_prompt = f\"\"\"Tu es un tuteur IA spécialisé en {subject}.\n",
        "Utilise le contexte suivant pour répondre de manière précise et pédagogique.\n",
        "\n",
        "Contexte:\n",
        "{context}\n",
        "\n",
        "Réponds de manière claire et concise (2-3 phrases maximum).\n",
        "N'utilise pas de caractères spéciaux car ta réponse sera convertie en audio.\"\"\"\n",
        "\n",
        "# Get LLM response\n",
        "print(\"⏳ Génération de la réponse...\")\n",
        "response = await agent.llm_service.generate_response(\n",
        "    prompt=test_question,\n",
        "    system_prompt=system_prompt\n",
        ")\n",
        "\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"📊 RÉSULTAT\")\n",
        "print(\"=\"*70)\n",
        "print(f\"\\n📚 Domaine: {subject}\")\n",
        "print(f\"\\n💡 Réponse:\\n{response}\")\n",
        "\n",
        "# Generate audio\n",
        "print(f\"\\n⏳ Génération audio...\")\n",
        "audio_bytes = await agent.tts_service.synthesize(response)\n",
        "\n",
        "print(f\"\\n🔊 Audio généré: {len(audio_bytes)} bytes à 22050 Hz\")\n",
        "print(\"\\n✅ Test réussi! L'architecture temps réel fonctionne!\")"
    ]
}

# Insert new title at top
nb['cells'].insert(0, new_title_cell)

# Add new test cells after cell 15 (index 16 now with new title)
nb['cells'].insert(16, new_test_md_cell)
nb['cells'].insert(17, new_test_code_cell)

print(f"✅ Cellules ajoutées: {len(nb['cells'])} cellules au total")

# Save
with open('notebooks/demo_pipecat_colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("💾 Notebook sauvegardé!")
