# Configuration du Streaming Audio sur WSL

## Problème identifié

WSL par défaut n'a **pas** de serveur audio configuré, ce qui bloque `sounddevice` :
```
❌ Aucun module kernel audio (lsmod | grep snd → vide)
❌ PulseAudio non installé
❌ sounddevice ne peut pas ouvrir de device audio
```

## ✅ Solution 1 : PulseAudio + Windows Audio Bridge (STREAMING NATIF)

### Étape 1 : Installer PulseAudio sur Windows

1. **Télécharger PulseAudio pour Windows** :
   - https://www.freedesktop.org/wiki/Software/PulseAudio/Ports/Windows/Support/
   - Ou via Chocolatey : `choco install pulseaudio`

2. **Configurer PulseAudio Windows en mode serveur** :
   ```powershell
   # Dans default.pa (généralement C:\PulseAudio\etc\pulse\default.pa)
   load-module module-native-protocol-tcp auth-ip-acl=127.0.0.1;172.16.0.0/12
   load-module module-waveout sink_name=output source_name=input record=1
   ```

3. **Lancer PulseAudio sur Windows** :
   ```powershell
   pulseaudio.exe --start
   ```

### Étape 2 : Configurer WSL pour se connecter

1. **Installer PulseAudio client sur WSL** :
   ```bash
   sudo apt-get update
   sudo apt-get install -y pulseaudio pulseaudio-utils
   ```

2. **Configurer la variable d'environnement** :
   ```bash
   # Ajouter dans ~/.bashrc
   export PULSE_SERVER=tcp:$(grep nameserver /etc/resolv.conf | awk '{print $2}')
   source ~/.bashrc
   ```

3. **Tester la connexion** :
   ```bash
   pactl info  # Devrait afficher le serveur PulseAudio Windows
   pactl list sources short  # Devrait lister le microphone Windows
   ```

4. **Lancer le streaming** :
   ```bash
   cd /root/intelligence_lab_agent_vocal
   source venv/bin/activate
   python -m src.ui_streaming
   ```

### Résultat attendu
✅ **Streaming natif complet** avec latence < 1s (comme ChatGPT Voice)
✅ `sounddevice` accède au microphone Windows via PulseAudio
✅ Conversation continue sans cliquer

---

## ✅ Solution 2 : Mode Hybride (ACTUEL - Plus simple)

**Avantages** :
- ✅ Fonctionne immédiatement sans configuration supplémentaire
- ✅ Pas besoin d'installer PulseAudio sur Windows
- ✅ Utilise l'API Web Audio du navigateur

**Inconvénients** :
- ⚠️ Nécessite de cliquer pour enregistrer (pas de conversation continue)
- ⚠️ Latence 2-5s (au lieu de <1s)

**Actuellement actif** : `ui_hybrid.py` sur port 7860

---

## ✅ Solution 3 : WSLg (Windows 11 uniquement)

Si vous êtes sur **Windows 11 avec WSLg** (WSL GUI support) :

```bash
# Vérifier si WSLg est disponible
echo $WAYLAND_DISPLAY  # Si non vide → WSLg actif

# Installer PipeWire (remplaçant moderne de PulseAudio)
sudo apt-get install -y pipewire pipewire-pulse

# Relancer le streaming
python -m src.ui_streaming
```

---

## Comparaison des solutions

| Solution | Latence | Setup | Streaming continu |
|----------|---------|-------|-------------------|
| **PulseAudio Bridge** | <1s | Moyen (Windows + WSL) | ✅ Oui |
| **Mode Hybride** | 2-5s | ✅ Aucun | ❌ Click-to-record |
| **WSLg (Win11)** | <1s | Facile | ✅ Oui |

---

## Recommandation

1. **Si vous voulez tester rapidement** → Restez avec le mode hybride (actuellement lancé)
2. **Si vous voulez le vrai streaming** → Configurez PulseAudio Bridge (1h de setup)
3. **Si Windows 11** → Activez WSLg (30 min)

Le streaming **EST possible sur WSL**, il faut juste configurer l'accès audio ! 🎯
