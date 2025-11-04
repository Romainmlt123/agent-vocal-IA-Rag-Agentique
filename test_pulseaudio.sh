#!/bin/bash
# Test de connectivité PulseAudio pour streaming mode

echo "🔍 Test de configuration PulseAudio pour Streaming Mode"
echo ""

# Vérifier si la variable est définie
if [ -z "$PULSE_SERVER" ]; then
    export PULSE_SERVER=tcp:$(grep nameserver /etc/resolv.conf | awk '{print $2}')
    echo "✅ Variable PULSE_SERVER définie: $PULSE_SERVER"
else
    echo "✅ Variable PULSE_SERVER déjà définie: $PULSE_SERVER"
fi

echo ""
echo "🔌 Test de connexion au serveur PulseAudio..."
echo ""

# Tester la connexion
if pactl info &> /dev/null; then
    echo "✅ SUCCÈS - PulseAudio est accessible!"
    echo ""
    echo "📊 Informations serveur:"
    pactl info | grep -E "(Server String|Server Name|User Name)"
    echo ""
    echo "🎙️ Sources audio disponibles (microphones):"
    pactl list sources short
    echo ""
    echo "✅ Vous pouvez lancer le streaming mode:"
    echo "   bash scripts/run_streaming.sh"
else
    echo "❌ ÉCHEC - PulseAudio non accessible"
    echo ""
    echo "📝 Actions requises sur Windows:"
    echo ""
    echo "1️⃣  Installer PulseAudio:"
    echo "    PowerShell (admin): choco install pulseaudio"
    echo "    OU exécuter: setup_pulseaudio_windows.ps1"
    echo ""
    echo "2️⃣  Démarrer PulseAudio sur Windows:"
    echo "    C:\\PulseAudio\\bin\\pulseaudio.exe --start"
    echo ""
    echo "3️⃣  Re-tester cette commande"
    echo ""
    echo "⚠️  En attendant, utilisez le mode hybride:"
    echo "    python -m src.ui_hybrid (latence 2-5s)"
fi

echo ""
