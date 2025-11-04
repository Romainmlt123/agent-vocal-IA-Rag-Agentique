#!/bin/bash
# Build FAISS indexes for all subjects

set -e

echo "================================"
echo "Building RAG Indexes"
echo "================================"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if data directories exist
if [ ! -d "data/maths" ] && [ ! -d "data/physique" ] && [ ! -d "data/anglais" ]; then
    echo "⚠️  Warning: No data directories found."
    echo "Creating empty data directories..."
    mkdir -p data/maths data/physique data/anglais
    echo "Please add PDF or TXT files to these directories before building indexes."
    exit 1
fi

# Run the RAG builder
echo "Running RAG index builder..."
python -m src.rag_build

echo ""
echo "================================"
echo "Index Building Complete"
echo "================================"
echo ""
echo "Indexes have been created in:"
echo "  - data/maths/index.faiss"
echo "  - data/physique/index.faiss"
echo "  - data/anglais/index.faiss"
echo ""
