#!/usr/bin/env python3
"""Test des 3 matières avec hints complets."""

from src.orchestrator import TutoringOrchestrator

def test_all_subjects():
    """Test chaque matière avec une question."""
    orch = TutoringOrchestrator()
    
    questions = {
        "Maths": "Comment résoudre x² - 5x + 6 = 0?",
        "Physique": "Explique-moi la deuxième loi de Newton",
        "Anglais": "What's the difference between present perfect and past simple?"
    }
    
    for subject_name, question in questions.items():
        print(f"\n{'='*70}")
        print(f"📚 TEST: {subject_name}")
        print(f"❓ Question: {question}")
        print('='*70)
        
        session = orch.create_session()
        
        hints_found = False
        for event in orch.process_text_query(session, question):
            if event.type == 'subject_detected':
                detected = event.data['subject'].upper()
                match = "✅" if detected == subject_name.upper() else "❌"
                print(f"{match} Matière détectée: {detected}")
                
            elif event.type == 'rag_results':
                print(f"✅ RAG: {len(event.data)} sources récupérées")
                
            elif event.type == 'hints':
                hints = event.data
                hints_found = True
                
                if hints.get('level1'):
                    print(f"\n💡 HINT 1 (Conceptuel):")
                    print(f"   {hints['level1'][:200]}...")
                else:
                    print("\n❌ HINT 1: VIDE")
                    
                if hints.get('level2'):
                    print(f"\n💡 HINT 2 (Stratégique):")
                    print(f"   {hints['level2'][:200]}...")
                else:
                    print("\n❌ HINT 2: VIDE")
                    
                if hints.get('level3'):
                    print(f"\n💡 HINT 3 (Détaillé):")
                    print(f"   {hints['level3'][:200]}...")
                else:
                    print("\n❌ HINT 3: VIDE")
                    
            elif event.type == 'error':
                print(f"\n❌ ERREUR: {event.data}")
        
        if not hints_found:
            print("\n❌ AUCUN HINT GÉNÉRÉ!")
        
        print()

if __name__ == "__main__":
    print("\n🧪 TEST COMPLET DES 3 MATIÈRES")
    test_all_subjects()
    print("\n" + "="*70)
    print("✅ Tests terminés!")
