"""
Test script for the voice pipeline orchestrator.

This script tests the pipeline with sample questions to verify
that all components are working correctly.
"""

import asyncio
import sys
from loguru import logger

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")

from src.pipeline.voice_pipeline import create_voice_pipeline


async def test_pipeline():
    """Test the voice pipeline with sample questions."""
    
    print("\n" + "="*70)
    print("🧪 Testing Voice Pipeline Orchestrator")
    print("="*70 + "\n")
    
    # Test questions for each subject
    test_questions = [
        {
            "subject": "maths",
            "question": "Comment résoudre l'équation x² + 2x - 8 = 0 ?",
        },
        {
            "subject": "physique",
            "question": "Qu'est-ce que la deuxième loi de Newton ?",
        },
        {
            "subject": "anglais",
            "question": "Comment conjuguer le verbe to be au présent ?",
        },
    ]
    
    try:
        # Create pipeline
        print("🚀 Creating pipeline...")
        pipeline = await create_voice_pipeline(
            stt_model_size="base",
            llm_model="qwen2:1.5b",
            tts_voice="fr_FR-siwis-medium",
            rag_top_k=4,
            enable_metrics=True,
        )
        print("✅ Pipeline created successfully!\n")
        
        # Test each question
        for i, test in enumerate(test_questions, 1):
            print("\n" + "-"*70)
            print(f"Test {i}/{len(test_questions)}: {test['subject'].upper()}")
            print("-"*70)
            
            result = await pipeline.process_question(test["question"])
            
            print(f"\n📝 Question: {result['question']}")
            print(f"🎯 Subject: {result['subject']} (expected: {test['subject']})")
            print(f"📚 Sources: {result['num_sources']} documents")
            print(f"\n💡 Answer:\n{result['answer']}\n")
            
            # Verify routing
            if result['subject'] == test['subject']:
                print("✅ Routing correct!")
            else:
                print(f"⚠️  Routing mismatch: got {result['subject']}, expected {test['subject']}")
        
        # Cleanup
        await pipeline.stop()
        
        print("\n" + "="*70)
        print("✅ All tests completed!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(test_pipeline())
    sys.exit(exit_code)
