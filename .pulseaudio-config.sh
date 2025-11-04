#!/bin/bash
# Configuration PulseAudio pour WSL - Streaming Audio natif

# Obtenir l'IP de l'hôte Windows
WINDOWS_IP=$(grep nameserver /etc/resolv.conf | awk '{print $2}')

# Configurer la variable d'environnement PulseAudio
export PULSE_SERVER=tcp:$WINDOWS_IP

echo "✅ Configuration PulseAudio WSL"
echo "   Serveur Windows: $WINDOWS_IP"
echo ""
echo "⚠️  IMPORTANT: Vous devez installer PulseAudio sur Windows:"
echo ""
echo "Option 1 - Chocolatey (recommandé):"
echo "  1. Ouvrir PowerShell en administrateur sur Windows"
echo "  2. Exécuter: choco install pulseaudio"
echo ""
echo "Option 2 - Manuel:"
echo "  1. Télécharger: https://www.freedesktop.org/wiki/Software/PulseAudio/Ports/Windows/Support/"
echo "  2. Extraire dans C:\\PulseAudio"
echo "  3. Créer C:\\PulseAudio\\etc\\pulse\\default.pa avec:"
echo "     load-module module-native-protocol-tcp auth-ip-acl=127.0.0.1;172.16.0.0/12"
echo "     load-module module-waveout sink_name=output source_name=input record=1"
echo "  4. Lancer: C:\\PulseAudio\\bin\\pulseaudio.exe"
echo ""
echo "Puis tester avec: pactl info"
