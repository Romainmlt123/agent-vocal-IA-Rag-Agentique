# Script d'installation PulseAudio pour Windows
# À exécuter dans PowerShell en administrateur depuis le dossier du projet

Write-Host "🎙️ Installation PulseAudio pour Streaming Mode WSL" -ForegroundColor Cyan
Write-Host ""

# Vérifier si exécuté en admin
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "❌ Ce script doit être exécuté en administrateur!" -ForegroundColor Red
    Write-Host "   Clic droit sur PowerShell → Exécuter en tant qu'administrateur" -ForegroundColor Yellow
    pause
    exit
}

# Vérifier si Chocolatey est installé
$chocoInstalled = Get-Command choco -ErrorAction SilentlyContinue

if ($chocoInstalled) {
    Write-Host "✅ Chocolatey détecté - Installation automatique" -ForegroundColor Green
    choco install pulseaudio -y
} else {
    Write-Host "⚠️  Chocolatey non installé" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "📥 Installation manuelle de PulseAudio..." -ForegroundColor Cyan
    Write-Host ""
    
    # Télécharger PulseAudio
    $pulseUrl = "https://www.freedesktop.org/software/pulseaudio/releases/pulseaudio-1.1.zip"
    $downloadPath = "$env:TEMP\pulseaudio.zip"
    $extractPath = "C:\PulseAudio"
    
    Write-Host "1️⃣  Téléchargement de PulseAudio..." -ForegroundColor Cyan
    try {
        # Note: Le lien officiel peut ne pas fonctionner, on guide l'utilisateur
        Write-Host ""
        Write-Host "⚠️  Le téléchargement automatique n'est pas disponible." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "📝 Instructions manuelles:" -ForegroundColor Cyan
        Write-Host "   1. Téléchargez PulseAudio depuis:"
        Write-Host "      https://github.com/pgaskin/pulseaudio-win32/releases" -ForegroundColor Green
        Write-Host "      OU"
        Write-Host "      https://www.freedesktop.org/wiki/Software/PulseAudio/Ports/Windows/Support/" -ForegroundColor Green
        Write-Host ""
        Write-Host "   2. Extrayez le ZIP dans: C:\PulseAudio" -ForegroundColor Green
        Write-Host ""
        Write-Host "   3. Relancez ce script" -ForegroundColor Green
        Write-Host ""
        
        $continue = Read-Host "Avez-vous extrait PulseAudio dans C:\PulseAudio ? (o/n)"
        if ($continue -ne "o") {
            Write-Host ""
            Write-Host "❌ Installation annulée" -ForegroundColor Red
            Write-Host "💡 Alternative : Utilisez le mode hybride (2-5s latence, sans PulseAudio)" -ForegroundColor Yellow
            pause
            exit
        }
    } catch {
        Write-Host "❌ Erreur: $_" -ForegroundColor Red
        pause
        exit
    }
}

# Configurer PulseAudio
$pulseConfigDir = "C:\PulseAudio\etc\pulse"
$pulseConfigFile = "$pulseConfigDir\default.pa"

Write-Host ""
Write-Host "🔧 Configuration de PulseAudio..." -ForegroundColor Cyan

# Créer le répertoire de configuration si nécessaire
if (-not (Test-Path $pulseConfigDir)) {
    New-Item -ItemType Directory -Path $pulseConfigDir -Force | Out-Null
}

# Créer le fichier de configuration
$configContent = @"
# Configuration PulseAudio pour WSL
load-module module-native-protocol-tcp auth-ip-acl=127.0.0.1;172.16.0.0/12
load-module module-waveout sink_name=output source_name=input record=1
"@

Set-Content -Path $pulseConfigFile -Value $configContent

Write-Host "✅ Configuration créée: $pulseConfigFile" -ForegroundColor Green

# Démarrer PulseAudio
Write-Host ""
Write-Host "🚀 Démarrage de PulseAudio..." -ForegroundColor Cyan

$pulsePath = "C:\PulseAudio\bin\pulseaudio.exe"
if (Test-Path $pulsePath) {
    Start-Process -FilePath $pulsePath -ArgumentList "--start" -WindowStyle Hidden
    Write-Host "✅ PulseAudio démarré" -ForegroundColor Green
} else {
    $pulsePath = (Get-Command pulseaudio -ErrorAction SilentlyContinue).Source
    if ($pulsePath) {
        Start-Process -FilePath "pulseaudio" -ArgumentList "--start" -WindowStyle Hidden
        Write-Host "✅ PulseAudio démarré" -ForegroundColor Green
    } else {
        Write-Host "❌ pulseaudio.exe introuvable" -ForegroundColor Red
        Write-Host "   Vérifiez l'installation dans C:\PulseAudio\bin\" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "✅ Installation terminée!" -ForegroundColor Green
Write-Host ""
Write-Host "📝 Prochaines étapes dans WSL:" -ForegroundColor Cyan
Write-Host "   1. Dans votre terminal WSL, exécuter: pactl info"
Write-Host "   2. Si la connexion fonctionne, lancer: bash scripts/run_streaming.sh"
Write-Host "   3. Profitez du streaming en temps réel (<1s de latence)!"
Write-Host ""
