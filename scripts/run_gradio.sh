#!/bin/bash
# Run Gradio UI

set -e

echo "================================"
echo "Launching Agent Vocal Prof UI"
echo "================================"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if indexes exist
if [ ! -f "data/maths/index.faiss" ] && [ ! -f "data/physique/index.faiss" ] && [ ! -f "data/anglais/index.faiss" ]; then
    echo "⚠️  Warning: No RAG indexes found."
    echo "Run ./scripts/build_indexes.sh first to create indexes."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Run the Gradio UI
echo "Starting Gradio UI..."
echo "The interface will open at: http://localhost:7860"
echo ""
echo "Press Ctrl+C to stop the server."
echo ""

python -m src.ui_gradio
