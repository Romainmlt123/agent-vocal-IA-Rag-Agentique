"""
Tests for LLM module.
"""

import pytest

from src.llm import LLMEngine, HintLadder


def test_hint_ladder_creation():
    """Test HintLadder creation."""
    hints = HintLadder(
        level1="Think about the concept",
        level2="Use the quadratic formula",
        level3="Substitute values into ax² + bx + c = 0"
    )
    
    assert hints.level1 == "Think about the concept"
    assert hints.level2 == "Use the quadratic formula"
    assert hints.level3 == "Substitute values into ax² + bx + c = 0"
    
    hints_dict = hints.to_dict()
    assert "level1" in hints_dict
    assert "level2" in hints_dict
    assert "level3" in hints_dict


def test_llm_engine_initialization():
    """Test LLM engine initialization."""
    engine = LLMEngine()
    
    assert engine.config is not None
    assert engine.models is not None


def test_build_tutoring_prompt():
    """Test prompt building."""
    engine = LLMEngine()
    
    question = "Comment résoudre x² + 2x + 1 = 0?"
    context = "Les équations du second degré..."
    subject = "maths"
    
    prompt = engine.build_tutoring_prompt(question, context, subject)
    
    assert "HINT LEVEL 1" in prompt
    assert "HINT LEVEL 2" in prompt
    assert "HINT LEVEL 3" in prompt
    assert question in prompt
    assert context in prompt


def test_parse_hint_ladder():
    """Test parsing LLM response into hint ladder."""
    engine = LLMEngine()
    
    response = """
    HINT LEVEL 1 (Conceptual):
    This is a quadratic equation.
    
    HINT LEVEL 2 (Strategic):
    You can factor it or use the formula.
    
    HINT LEVEL 3 (Detailed):
    Notice that this is a perfect square: (x+1)².
    """
    
    hints = engine.parse_hint_ladder(response)
    
    assert "quadratic equation" in hints.level1
    assert "factor" in hints.level2 or "formula" in hints.level2
    assert "perfect square" in hints.level3


def test_hint_ladder_representation():
    """Test HintLadder string representation."""
    hints = HintLadder("a", "b", "c")
    
    repr_str = repr(hints)
    assert "HintLadder" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
