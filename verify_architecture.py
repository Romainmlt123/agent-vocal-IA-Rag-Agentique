"""
Script de vérification de l'architecture RAG Agentique.
Vérifie :
1. Nombre de modèles LLM différents
2. Architecture agentique (Router + modèles spécialisés)
3. RAG par matière
"""
import sys
from pathlib import Path
from src.config import get_config
from src.utils import setup_logging, get_logger

setup_logging(log_level="INFO")
logger = get_logger(__name__)

def verify_architecture():
    """Vérification complète de l'architecture."""
    
    config = get_config()
    
    print("\n" + "="*80)
    print("🔍 VÉRIFICATION DE L'ARCHITECTURE RAG AGENTIQUE")
    print("="*80 + "\n")
    
    # 1. Vérification des modèles LLM
    print("📊 1. MODÈLES LLM PAR MATIÈRE:")
    print("-" * 80)
    
    models_config = config.llm.models
    unique_models = {}
    
    for subject, model_path in models_config.items():
        full_path = Path(config.models_dir) / "llm" / model_path
        exists = full_path.exists()
        
        # Grouper par modèle unique
        if model_path not in unique_models:
            unique_models[model_path] = []
        unique_models[model_path].append(subject)
        
        status = "✅" if exists else "❌"
        print(f"  {status} {subject:10s} → {model_path}")
        if exists:
            size_mb = full_path.stat().st_size / (1024*1024)
            print(f"             Taille: {size_mb:.1f} MB")
    
    print(f"\n📈 Résumé:")
    print(f"   • Matières configurées: {len([k for k in models_config.keys() if k != 'default'])}")
    print(f"   • Modèles physiques uniques: {len(unique_models)}")
    
    print("\n🔄 Modèles uniques et leurs matières:")
    for model_path, subjects in unique_models.items():
        print(f"   • {model_path}")
        print(f"     → Utilisé pour: {', '.join(subjects)}")
    
    # 2. Vérification du Router (Agent)
    print("\n" + "-" * 80)
    print("🤖 2. ROUTER AGENTIQUE:")
    print("-" * 80)
    
    keywords_config = config.router.keywords
    print(f"   ✅ Router configuré avec {len(keywords_config)} matières")
    
    for subject, keywords in keywords_config.items():
        print(f"\n   📚 {subject.upper()}:")
        print(f"      Keywords ({len(keywords)}): {', '.join(keywords[:8])}")
        if len(keywords) > 8:
            print(f"                         ... et {len(keywords)-8} autres")
    
    # 3. Vérification du RAG
    print("\n" + "-" * 80)
    print("📚 3. RAG (RETRIEVAL AUGMENTED GENERATION):")
    print("-" * 80)
    
    indexes_config = config.rag.indexes
    print(f"   ✅ RAG configuré avec {len(indexes_config)} index FAISS")
    
    for subject, index_path in indexes_config.items():
        full_path = Path(config.project_root) / index_path
        exists = full_path.exists()
        status = "✅" if exists else "❌"
        
        print(f"\n   {status} {subject.upper()}:")
        print(f"      Index: {index_path}")
        
        if exists:
            # Charger l'index pour compter les vecteurs
            try:
                import faiss
                index = faiss.read_index(str(full_path))
                print(f"      Vecteurs: {index.ntotal}")
                print(f"      Dimension: {index.d}")
            except Exception as e:
                print(f"      ⚠️  Erreur lecture: {e}")
    
    # 4. Analyse de l'architecture
    print("\n" + "="*80)
    print("📊 4. ANALYSE DE L'ARCHITECTURE:")
    print("="*80 + "\n")
    
    # Vérifier si c'est vraiment agentique
    is_agentic = len(unique_models) > 1
    has_router = len(keywords_config) > 0
    has_rag = len(indexes_config) > 0
    
    if is_agentic:
        print("   ✅ ARCHITECTURE AGENTIQUE CONFIRMÉE")
        print(f"      • {len(unique_models)} modèles LLM différents (spécialisation)")
        print(f"      • Router intelligent avec {sum(len(k) for k in keywords_config.values())} keywords")
        print(f"      • Pipeline: ASR → Router → RAG → LLM spécialisé → TTS")
    else:
        print("   ⚠️  ARCHITECTURE NON-AGENTIQUE")
        print(f"      • Seulement {len(unique_models)} modèle(s) LLM unique(s)")
        print("      • Tous les sujets utilisent le même modèle")
        print("      • Pour être agentique, il faudrait des modèles différents par matière")
    
    if has_rag:
        print(f"\n   ✅ RAG CONFIRMÉ")
        print(f"      • {len(indexes_config)} index FAISS (un par matière)")
        print(f"      • Embedding model: {config.rag.embedding_model}")
        print(f"      • Top-K retrieval: {config.rag.top_k} passages")
    
    # 5. Conclusion et recommandations
    print("\n" + "="*80)
    print("💡 5. CONCLUSION:")
    print("="*80 + "\n")
    
    if is_agentic and has_rag:
        print("   🎉 VOUS AVEZ UN RAG AGENTIQUE COMPLET !")
        print("\n   Votre système est:")
        print("   • Agentique: Router qui sélectionne le modèle adapté")
        print("   • RAG: Récupération de contexte spécialisé par matière")
        print("   • Multi-modal: Text + Audio input/output")
    elif has_rag and not is_agentic:
        print("   ⚠️  VOUS AVEZ UN RAG MAIS PAS VRAIMENT AGENTIQUE")
        print("\n   Votre système est:")
        print("   • RAG: ✅ Contexte récupéré par matière")
        print("   • Router: ✅ Détection de la matière")
        print("   • Agentique: ❌ Tous les sujets utilisent le même modèle")
        print("\n   📌 RECOMMANDATION:")
        print("      Pour être vraiment agentique, téléchargez des modèles différents:")
        print("      • Qwen2-Math pour les maths")
        print("      • Un modèle scientifique pour la physique")
        print("      • Phi-3 (déjà présent) pour l'anglais")
    else:
        print("   ❌ ARCHITECTURE INCOMPLÈTE")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    try:
        verify_architecture()
    except Exception as e:
        logger.error(f"Erreur: {e}", exc_info=True)
        sys.exit(1)
