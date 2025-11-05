#!/usr/bin/env python3
"""
Standalone script to build RAG indexes for Colab.
No dependencies on src.config or src.utils.
"""

import os
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks


def build_index_for_subject(subject: str, data_dir: Path, model):
    """Build FAISS index for a subject."""
    subject_dir = data_dir / subject
    
    if not subject_dir.exists():
        print(f"⚠️  Dossier {subject} non trouvé")
        return False
    
    # Lire tous les fichiers .txt
    txt_files = list(subject_dir.glob("*.txt"))
    
    if not txt_files:
        print(f"⚠️  Aucun fichier .txt dans {subject}")
        return False
    
    print(f"\n📚 Traitement {subject}...")
    print(f"   Fichiers trouvés : {len(txt_files)}")
    
    all_chunks = []
    all_metadata = []
    
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Créer des chunks
            chunks = chunk_text(content, chunk_size=500, overlap=50)
            
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metadata.append({
                    "source": txt_file.name,
                    "subject": subject,
                    "text": chunk
                })
            
            print(f"   ✅ {txt_file.name}: {len(chunks)} chunks")
        
        except Exception as e:
            print(f"   ❌ Erreur {txt_file.name}: {e}")
    
    if not all_chunks:
        print(f"⚠️  Aucun chunk créé pour {subject}")
        return False
    
    print(f"\n   🔄 Génération des embeddings ({len(all_chunks)} chunks)...")
    
    # Générer embeddings
    embeddings = model.encode(all_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    
    # Normaliser pour IndexFlatIP (cosine similarity)
    faiss.normalize_L2(embeddings)
    
    # Créer index FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    # Sauvegarder index et métadonnées
    index_path = subject_dir / "index.faiss"
    metadata_path = subject_dir / "metadata.pkl"
    
    faiss.write_index(index, str(index_path))
    
    with open(metadata_path, 'wb') as f:
        pickle.dump(all_metadata, f)
    
    print(f"   ✅ Index sauvegardé : {index_path}")
    print(f"   ✅ Métadonnées : {metadata_path}")
    print(f"   📊 {len(all_chunks)} chunks, dimension {dimension}")
    
    return True


def main():
    """Main function."""
    print("=" * 70)
    print("🗄️  CONSTRUCTION DES INDEX RAG")
    print("=" * 70)
    
    # Trouver le dossier data
    if os.path.exists("/content/agent-vocal-ia-RAG-Agentique"):
        # Sur Colab
        base_dir = Path("/content/agent-vocal-ia-RAG-Agentique")
    else:
        # Local
        base_dir = Path(__file__).parent.parent
    
    data_dir = base_dir / "data"
    
    if not data_dir.exists():
        print(f"❌ Dossier data non trouvé : {data_dir}")
        sys.exit(1)
    
    print(f"\n📁 Dossier data : {data_dir}")
    
    # Charger le modèle d'embeddings
    print("\n🤖 Chargement du modèle d'embeddings...")
    print("   Modèle : sentence-transformers/all-MiniLM-L6-v2")
    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    print("   ✅ Modèle chargé")
    
    # Construire index pour chaque sujet
    subjects = ['maths', 'physique', 'anglais']
    results = {}
    
    for subject in subjects:
        try:
            results[subject] = build_index_for_subject(subject, data_dir, model)
        except Exception as e:
            print(f"\n❌ Erreur pour {subject}: {e}")
            results[subject] = False
    
    # Résumé
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ")
    print("=" * 70)
    
    for subject, success in results.items():
        if success:
            index_path = data_dir / subject / "index.faiss"
            size_kb = index_path.stat().st_size / 1024
            print(f"✅ {subject:10s} : {size_kb:.1f} KB")
        else:
            print(f"❌ {subject:10s} : échec")
    
    success_count = sum(results.values())
    print(f"\n🎯 {success_count}/{len(subjects)} index construits avec succès")
    
    if success_count == len(subjects):
        print("\n✅ Tous les index RAG sont prêts !")
    else:
        print("\n⚠️  Certains index n'ont pas pu être construits")


if __name__ == "__main__":
    main()
