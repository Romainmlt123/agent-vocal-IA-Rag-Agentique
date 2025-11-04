#!/usr/bin/env python3
"""Test de la question spécifique de l'utilisateur."""

from src.orchestrator import TutoringOrchestrator

def test_math_question():
    """Test avec la question exacte de l'utilisateur."""
    print("🧮 Test : Comment résoudre x² - 5x + 6 = 0?\n")
    print("="*70)
    
    orch = TutoringOrchestrator()
    session_id = orch.create_session()
    
    question = "Comment résoudre x² - 5x + 6 = 0?"
    print(f"📝 Question: {question}\n")
    
    for event in orch.process_text_query(session_id, question):
        if event.type == "subject_detected":
            subject = event.data['subject']
            print(f"✅ Matière détectée: {subject.upper()}")
            
        elif event.type == "rag_results":
            sources = event.data
            print(f"✅ RAG: {len(sources)} sources récupérées")
            for i, src in enumerate(sources[:2], 1):
                print(f"   {i}. {src['source']} (score: {src['score']:.3f})")
                print(f"      {src['text'][:80]}...")
            
        elif event.type == "hints":
            hints = event.data
            print(f"\n💡 HINT NIVEAU 1 (Conceptuel):")
            print(f"   {hints.get('level1', 'N/A')[:200]}")
            print(f"\n💡 HINT NIVEAU 2 (Stratégique):")
            print(f"   {hints.get('level2', 'N/A')[:200]}")
            print(f"\n💡 HINT NIVEAU 3 (Détaillé):")
            print(f"   {hints.get('level3', 'N/A')[:200]}")
            
        elif event.type == "error":
            print(f"❌ ERREUR: {event.data}")
    
    print("\n" + "="*70)
    print("✅ Test terminé!")

if __name__ == "__main__":
    test_math_question()
