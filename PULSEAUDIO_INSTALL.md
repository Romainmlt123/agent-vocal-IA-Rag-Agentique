# Installation PulseAudio sur Windows (Sans Chocolatey)

## 🎯 Objectif
Permettre au mode streaming WSL d'accéder au microphone Windows pour une latence <1s.

## 📥 Étape 1 : Télécharger PulseAudio

**⚠️ IMPORTANT : Téléchargez les BINAIRES, pas le code source !**

1. **GitHub Releases (Recommandé)** : 
   https://github.com/pgaskin/pulseaudio-win32/releases/latest
   
   → Cliquez sur **"Assets"**
   → Téléchargez **`pulseaudio-X.X-msys2-x86_64.zip`** (le plus gros fichier, ~20-30 MB)
   → **PAS** le fichier "Source code (zip)" !

2. **Alternatif (plus ancien)** :
   https://www.freedesktop.org/wiki/Software/PulseAudio/Ports/Windows/Support/

## 📂 Étape 2 : Extraire

1. Extraire le ZIP téléchargé
2. Renommer le dossier en `PulseAudio`
3. Déplacer dans `C:\PulseAudio`

**Structure attendue :**
```
C:\PulseAudio\
├── bin\
│   └── pulseaudio.exe
├── etc\
│   └── pulse\
└── lib\
```

## ⚙️ Étape 3 : Configurer

1. Créer le fichier : `C:\PulseAudio\etc\pulse\default.pa`

2. Coller ce contenu :
```
load-module module-native-protocol-tcp auth-ip-acl=127.0.0.1;172.16.0.0/12
load-module module-waveout sink_name=output source_name=input record=1
```

3. Sauvegarder

## 🚀 Étape 4 : Démarrer PulseAudio

**Option A - Temporaire (pour tester)** :
```powershell
# Dans PowerShell (pas besoin d'admin)
C:\PulseAudio\bin\pulseaudio.exe
```

**Option B - Automatique au démarrage** :
1. Appuyer sur `Win + R`
2. Taper : `shell:startup`
3. Créer un raccourci vers `C:\PulseAudio\bin\pulseaudio.exe`

## ✅ Étape 5 : Tester

Dans WSL :
```bash
cd /root/intelligence_lab_agent_vocal
./test_pulseaudio.sh
```

**Si ✅ SUCCÈS** :
```bash
bash scripts/run_streaming.sh
```

**Si ❌ ÉCHEC** :
- Vérifier que pulseaudio.exe est lancé (Gestionnaire des tâches Windows)
- Vérifier le fichier default.pa
- Redémarrer pulseaudio.exe

## 🔥 Alternative : Mode Hybride (Sans PulseAudio)

Si PulseAudio ne fonctionne pas, utilisez le mode hybride déjà configuré :

```bash
python -m src.ui_hybrid
```

- ✅ Pas de configuration Windows nécessaire
- ✅ Latence 2-5s (vs 10-15s push-to-talk)
- ⚠️ Nécessite de cliquer le micro (pas de conversation continue)

## 📊 Comparaison

| Mode | Latence | Configuration | Conversation continue |
|------|---------|---------------|----------------------|
| Streaming natif | <1s | PulseAudio Windows | ✅ Oui |
| Hybride | 2-5s | Aucune | ❌ Click-to-record |
| Push-to-talk | 10-15s | Aucune | ❌ Click start/stop |
