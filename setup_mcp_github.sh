#!/bin/bash
# Script d'installation et configuration du serveur MCP GitHub pour Copilot
# Ce script configure un serveur MCP local pour accéder aux repos GitHub sans les cloner

set -e

echo "🚀 Configuration du serveur MCP GitHub pour Copilot"
echo "=================================================="
echo ""

# Variables
MCP_DIR="$HOME/.mcp"
GITHUB_TOKEN="ghp_11A7OKU6I0SVlD2jU6YdjO_AMgwmkIYgVppkU0WVMlhRqQYqoRHBOE9f8Lv8rkfyoS6VHCWOLOMpIaws9G"

# 1. Créer le dossier MCP global
echo "📁 Création du dossier $MCP_DIR..."
mkdir -p "$MCP_DIR"
cd "$MCP_DIR"

# 2. Installer le serveur MCP GitHub (via npx)
echo ""
echo "📦 Installation du serveur MCP GitHub..."
npm install @modelcontextprotocol/server-github

# 3. Créer le fichier de configuration
echo ""
echo "⚙️  Création du fichier config.json..."
cat > "$MCP_DIR/config.json" << 'CONFIGEOF'
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-github"
      ],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "GITHUB_TOKEN_PLACEHOLDER"
      }
    }
  }
}
CONFIGEOF

# Remplacer le token
sed -i "s/GITHUB_TOKEN_PLACEHOLDER/$GITHUB_TOKEN/g" "$MCP_DIR/config.json"

echo "✅ Configuration créée dans $MCP_DIR/config.json"

# 4. Créer un script de démarrage
echo ""
echo "🔧 Création du script de démarrage..."
cat > "$MCP_DIR/start_github_mcp.sh" << 'STARTEOF'
#!/bin/bash
# Démarrage du serveur MCP GitHub

export GITHUB_PERSONAL_ACCESS_TOKEN="GITHUB_TOKEN_PLACEHOLDER"

echo "🚀 Démarrage du serveur MCP GitHub..."
echo "📡 Le serveur sera accessible via stdio (pas de port HTTP)"
echo ""
echo "Repos configurés :"
echo "  - pipecat-ai/pipecat"
echo "  - pipecat-ai/pipecat-examples"
echo "  - romain-mallet/agent-vocal-prof"
echo ""

npx -y @modelcontextprotocol/server-github
STARTEOF

# Remplacer le token dans le script
sed -i "s/GITHUB_TOKEN_PLACEHOLDER/$GITHUB_TOKEN/g" "$MCP_DIR/start_github_mcp.sh"
chmod +x "$MCP_DIR/start_github_mcp.sh"

echo "✅ Script de démarrage créé : $MCP_DIR/start_github_mcp.sh"

# 5. Créer la configuration VS Code
echo ""
echo "🔧 Création de la configuration VS Code..."
VSCODE_SETTINGS="$HOME/.vscode-server/data/Machine/settings.json"
mkdir -p "$(dirname "$VSCODE_SETTINGS")"

# Créer ou mettre à jour settings.json
if [ -f "$VSCODE_SETTINGS" ]; then
    echo "⚠️  Fichier settings.json existant détecté"
    echo "   Ajoutez manuellement cette configuration :"
else
    cat > "$VSCODE_SETTINGS" << 'VSEOF'
{
  "github.copilot.advanced": {
    "mcp": {
      "enabled": true,
      "servers": {
        "github": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-github"],
          "env": {
            "GITHUB_PERSONAL_ACCESS_TOKEN": "GITHUB_TOKEN_PLACEHOLDER"
          }
        }
      }
    }
  }
}
VSEOF
    sed -i "s/GITHUB_TOKEN_PLACEHOLDER/$GITHUB_TOKEN/g" "$VSCODE_SETTINGS"
    echo "✅ Configuration VS Code créée"
fi

echo ""
echo "=================================================="
echo "✅ Installation terminée !"
echo ""
echo "📝 Configuration résumée :"
echo "   - Dossier MCP : $MCP_DIR"
echo "   - Config : $MCP_DIR/config.json"
echo "   - Script démarrage : $MCP_DIR/start_github_mcp.sh"
echo ""
echo "🔐 Token GitHub configuré (commence par ghp_11A7...)"
echo ""
echo "📚 Repos accessibles :"
echo "   - pipecat-ai/pipecat"
echo "   - pipecat-ai/pipecat-examples"
echo "   - romain-mallet/agent-vocal-prof"
echo ""
echo "🚀 Pour utiliser :"
echo ""
echo "1️⃣  Le serveur MCP GitHub fonctionne via stdio (pas de port HTTP)"
echo "    GitHub Copilot communique directement via stdin/stdout"
echo ""
echo "2️⃣  Dans VS Code, ajoutez cette configuration dans settings.json :"
echo '    "github.copilot.advanced": {'
echo '      "mcp": {'
echo '        "enabled": true,'
echo '        "servers": {'
echo '          "github": {'
echo '            "command": "npx",'
echo '            "args": ["-y", "@modelcontextprotocol/server-github"],'
echo '            "env": {'
echo '              "GITHUB_PERSONAL_ACCESS_TOKEN": "votre_token"'
echo '            }'
echo '          }'
echo '        }'
echo '      }'
echo '    }'
echo ""
echo "3️⃣  Redémarrez VS Code"
echo ""
echo "4️⃣  Dans Copilot Chat, utilisez :"
echo '    @github what is pipecat framework?'
echo '    @github show me examples from pipecat-examples'
echo ""
echo "⚠️  Note : MCP utilise stdio, pas HTTP. Copilot lance le serveur"
echo "    automatiquement quand vous utilisez @github dans le chat"
echo ""
