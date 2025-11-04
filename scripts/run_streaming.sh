#!/bin/bash
#
# Script pour lancer l'interface streaming
#

cd "$(dirname "$0")/.." || exit 1

echo "🎤 Lancement de l'Agent Vocal Prof - Mode Streaming"
echo "=================================================="
echo ""

# Vérifier le venv
if [ ! -d "venv" ]; then
    echo "❌ Environnement virtuel non trouvé!"
    echo "   Exécutez: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activer venv
source venv/bin/activate

# Vérifier sounddevice
if ! python -c "import sounddevice" 2>/dev/null; then
    echo "⚠️  sounddevice non installé, installation..."
    pip install sounddevice
    echo ""
fi

# Vérifier qu'on est sur la bonne branche
BRANCH=$(git branch --show-current)
if [ "$BRANCH" != "feature/streaming-voice" ]; then
    echo "⚠️  Vous êtes sur la branche '$BRANCH'"
    echo "   Le mode streaming est sur 'feature/streaming-voice'"
    echo ""
    read -p "Voulez-vous basculer sur feature/streaming-voice? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git checkout feature/streaming-voice
    else
        echo "❌ Annulé"
        exit 1
    fi
fi

echo "✅ Environnement prêt"
echo ""
echo "📝 Instructions:"
echo "   1. Ouvrez http://localhost:7860 dans votre navigateur"
echo "   2. Cliquez sur 'Démarrer la conversation'"
echo "   3. Parlez dans votre microphone"
echo "   4. L'IA détecte automatiquement la fin de votre question"
echo "   5. La réponse est générée en temps réel"
echo ""
echo "⏹️  Pour arrêter: Ctrl+C"
echo ""
echo "🚀 Lancement en cours..."
echo ""

# Lancer l'interface
python -m src.ui_streaming
