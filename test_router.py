#!/usr/bin/env python3
"""Test rapide du router avec la config mise à jour."""

from src.router import SubjectRouter
from src.config import get_config

def test_router():
    """Test de détection des matières."""
    config = get_config()
    router = SubjectRouter(config)
    
    questions = [
        ("Comment résoudre x² - 5x + 6 = 0?", "maths"),
        ("Explique l'équation du second degré", "maths"),
        ("Quelle est la deuxième loi de Newton?", "physique"),
        ("Qu'est-ce que la force?", "physique"),
        ("What's the present perfect tense?", "anglais"),
        ("Différence entre past simple et present perfect", "anglais"),
        ("Bonjour comment ça va?", "default"),
    ]
    
    print("🧪 Test du Router avec Config Mise à Jour\n")
    print("="*70)
    
    success = 0
    total = len(questions)
    
    for question, expected in questions:
        detected = router.detect_subject(question)
        status = "✅" if detected == expected else "❌"
        success += (detected == expected)
        
        print(f"{status} Question: {question[:50]}")
        print(f"   Attendu: {expected.upper()} | Détecté: {detected.upper()}")
        print()
    
    print("="*70)
    print(f"Résultat: {success}/{total} tests réussis ({100*success/total:.0f}%)")
    
    if success == total:
        print("🎉 Tous les tests passent !")
    else:
        print("⚠️  Certains tests ont échoué")

if __name__ == "__main__":
    test_router()
