#!/usr/bin/env python3
"""Quick test of the tutoring pipeline."""

import sys
from src.orchestrator import TutoringOrchestrator

def test_pipeline():
    """Test the complete pipeline with a sample question."""
    print("🔄 Initializing orchestrator...")
    orch = TutoringOrchestrator()
    
    print("✅ Orchestrator loaded!")
    print("\n" + "="*60)
    
    # Create session
    session_id = orch.create_session()
    print(f"📝 Session created: {session_id}")
    
    # Test questions
    questions = [
        "Comment résoudre x² - 5x + 6 = 0?",
        "Explique-moi la deuxième loi de Newton",
        "What's the difference between present perfect and past simple?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}: {question}")
        print("="*60)
        
        for event in orch.process_text_query(session_id, question):
            if event.type == "subject_detected":
                print(f"📚 Subject: {event.data['subject'].upper()}")
            elif event.type == "rag_results":
                print(f"🔍 Retrieved {len(event.data)} RAG sources")
                for j, src in enumerate(event.data[:2], 1):
                    print(f"   {j}. {src['source']} (score: {src['score']:.3f})")
            elif event.type == "hints":
                hints = event.data
                print(f"\n💡 Level 1 (Conceptual):\n{hints.get('level1', 'N/A')}")
                print(f"\n💡 Level 2 (Strategic):\n{hints.get('level2', 'N/A')}")
                print(f"\n💡 Level 3 (Detailed):\n{hints.get('level3', 'N/A')}")
            elif event.type == "error":
                print(f"❌ Error: {event.data}")
    
    print("\n" + "="*60)
    print("✅ Test complete!")

if __name__ == "__main__":
    try:
        test_pipeline()
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
