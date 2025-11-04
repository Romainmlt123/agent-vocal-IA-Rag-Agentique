"""
LLM Module - Local Language Model with llama-cpp-python.
Handles model loading, prompt engineering, and streaming generation.
"""

from typing import Iterator, Optional, Dict, Any
from pathlib import Path

from .config import get_config
from .utils import get_logger


logger = get_logger(__name__)


class HintLadder:
    """Represents a 3-level hint ladder response."""
    
    def __init__(self, level1: str = "", level2: str = "", level3: str = ""):
        """
        Initialize hint ladder.
        
        Args:
            level1: Conceptual hint (high-level)
            level2: Strategic hint (method/approach)
            level3: Detailed hint (step-by-step guidance)
        """
        self.level1 = level1
        self.level2 = level2
        self.level3 = level3
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {
            "level1": self.level1,
            "level2": self.level2,
            "level3": self.level3
        }
    
    def __repr__(self) -> str:
        return f"HintLadder(levels={[bool(self.level1), bool(self.level2), bool(self.level3)]})"


class LLMEngine:
    """
    Local LLM engine using llama-cpp-python.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize LLM engine.
        
        Args:
            config_path: Path to config file
        """
        self.config = get_config(config_path)
        self.models = {}  # subject -> Llama instance
    
    def _load_model(self, model_path: Path):
        """
        Load a GGUF model.
        
        Args:
            model_path: Path to GGUF model file
        
        Returns:
            Llama instance
        """
        try:
            from llama_cpp import Llama
            
            logger.info(f"Loading LLM model: {model_path}")
            
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            model = Llama(
                model_path=str(model_path),
                n_ctx=self.config.llm.n_ctx,
                n_threads=self.config.llm.n_threads,
                verbose=False
            )
            
            logger.info(f"Model loaded successfully: {model_path.name}")
            return model
        
        except ImportError:
            logger.error("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            raise
    
    def get_model(self, subject: str):
        """
        Get or load model for a subject.
        
        Args:
            subject: Subject name
        
        Returns:
            Llama model instance
        """
        if subject not in self.models:
            model_path = self.config.get_model_path(subject)
            self.models[subject] = self._load_model(model_path)
        
        return self.models[subject]
    
    def build_tutoring_prompt(self, question: str, context: str, 
                                subject: str) -> str:
        """
        Build a pedagogical tutoring prompt with hint ladder structure.
        
        Args:
            question: Student's question
            context: RAG context passages
            subject: Subject area
        
        Returns:
            Formatted prompt
        """
        system_prompt = self.config.llm.system_prompt
        
        # Detect language from question (simple heuristic)
        is_french = any(word in question.lower() for word in [
            'comment', 'quelle', 'quel', 'pourquoi', 'explique', 'qu\'est-ce',
            'résoudre', 'calculer', 'trouve', 'détermine'
        ])
        
        if is_french:
            prompt = f"""{system_prompt}

Matière: {subject.title()}

Informations du contexte:
{context}

Question de l'étudiant: {question}

Instructions: Fournis exactement 3 niveaux d'indices pour guider l'étudiant (en français):

HINT LEVEL 1 (Conceptuel):
[Fournis un indice de haut niveau sur le concept ou le principe impliqué]

HINT LEVEL 2 (Stratégique):
[Explique l'approche ou la méthode à utiliser]

HINT LEVEL 3 (Détaillé):
[Donne des conseils étape par étape, mais laisse l'étudiant exécuter les étapes]

Rappel: Ne donne jamais la réponse directe. Guide l'étudiant pour qu'il la découvre lui-même.
"""
        else:
            prompt = f"""{system_prompt}

Subject: {subject.title()}

Context Information:
{context}

Student Question: {question}

Instructions: Provide exactly 3 levels of hints to guide the student:

HINT LEVEL 1 (Conceptual):
[Provide a high-level hint about the concept or principle involved]

HINT LEVEL 2 (Strategic):
[Explain the approach or method they should use]

HINT LEVEL 3 (Detailed):
[Give step-by-step guidance, but let the student execute the steps]

Remember: Never give the direct answer. Guide the student to discover it themselves.
"""
        
        return prompt
    
    def generate(self, prompt: str, subject: str = "default", 
                max_tokens: Optional[int] = None) -> str:
        """
        Generate text completion.
        
        Args:
            prompt: Input prompt
            subject: Subject to use for model selection
            max_tokens: Maximum tokens to generate
        
        Returns:
            Generated text
        """
        if max_tokens is None:
            max_tokens = self.config.llm.max_tokens
        
        model = self.get_model(subject)
        
        logger.debug(f"Generating with {subject} model (max_tokens={max_tokens})")
        
        output = model(
            prompt,
            max_tokens=max_tokens,
            temperature=self.config.llm.temperature,
            top_p=self.config.llm.top_p,
            top_k=self.config.llm.top_k,
            repeat_penalty=self.config.llm.repeat_penalty,
            stop=["Student Question:", "USER:", "\n\n\n"],
        )
        
        text = output["choices"][0]["text"]
        logger.debug(f"Generated {len(text)} characters")
        
        return text.strip()
    
    def generate_stream(self, prompt: str, subject: str = "default",
                       max_tokens: Optional[int] = None) -> Iterator[str]:
        """
        Generate text completion with streaming.
        
        Args:
            prompt: Input prompt
            subject: Subject to use for model selection
            max_tokens: Maximum tokens to generate
        
        Yields:
            Text chunks as they are generated
        """
        if max_tokens is None:
            max_tokens = self.config.llm.max_tokens
        
        model = self.get_model(subject)
        
        logger.debug(f"Starting streaming generation with {subject} model")
        
        stream = model(
            prompt,
            max_tokens=max_tokens,
            temperature=self.config.llm.temperature,
            top_p=self.config.llm.top_p,
            top_k=self.config.llm.top_k,
            repeat_penalty=self.config.llm.repeat_penalty,
            stop=["Student Question:", "USER:", "\n\n\n"],
            stream=True
        )
        
        for output in stream:
            chunk = output["choices"][0]["text"]
            yield chunk
    
    def parse_hint_ladder(self, response: str) -> HintLadder:
        """
        Parse LLM response into structured hint ladder.
        
        Args:
            response: LLM generated response
        
        Returns:
            HintLadder object
        """
        hints = HintLadder()
        
        # Simple parsing based on markers
        try:
            # Stop at common end markers
            stop_markers = ["Sources:", "Source:", "Remember:", "Note:", "Instructions:", 
                           "Please note", "The solution is", "Student Question:"]
            response_clean = response
            for marker in stop_markers:
                if marker in response_clean:
                    idx = response_clean.find(marker)
                    response_clean = response_clean[:idx]
            
            # Extract Level 1
            if "HINT LEVEL 1" in response_clean:
                start = response_clean.find("HINT LEVEL 1")
                end = response_clean.find("HINT LEVEL 2", start)
                if end == -1:
                    end = len(response_clean)
                level1_text = response_clean[start:end]
                # Remove the header
                level1_text = level1_text.replace("HINT LEVEL 1 (Conceptual):", "")
                level1_text = level1_text.replace("HINT LEVEL 1 (Conceptuel):", "")
                level1_text = level1_text.replace("HINT LEVEL 1:", "")
                level1_text = level1_text.replace("HINT LEVEL 1", "")
                hints.level1 = level1_text.strip()
            
            # Extract Level 2
            if "HINT LEVEL 2" in response_clean:
                start = response_clean.find("HINT LEVEL 2")
                end = response_clean.find("HINT LEVEL 3", start)
                if end == -1:
                    end = len(response_clean)
                level2_text = response_clean[start:end]
                level2_text = level2_text.replace("HINT LEVEL 2 (Strategic):", "")
                level2_text = level2_text.replace("HINT LEVEL 2 (Stratégique):", "")
                level2_text = level2_text.replace("HINT LEVEL 2:", "")
                level2_text = level2_text.replace("HINT LEVEL 2", "")
                hints.level2 = level2_text.strip()
            
            # Extract Level 3
            if "HINT LEVEL 3" in response_clean:
                start = response_clean.find("HINT LEVEL 3")
                level3_text = response_clean[start:]
                level3_text = level3_text.replace("HINT LEVEL 3 (Detailed):", "")
                level3_text = level3_text.replace("HINT LEVEL 3 (Détaillé):", "")
                level3_text = level3_text.replace("HINT LEVEL 3:", "")
                level3_text = level3_text.replace("HINT LEVEL 3", "")
                hints.level3 = level3_text.strip()
        
        except Exception as e:
            logger.warning(f"Error parsing hint ladder: {e}")
        
        return hints
    
    def generate_tutoring_response(self, question: str, context: str, 
                                  subject: str) -> HintLadder:
        """
        Generate a complete tutoring response with hint ladder.
        
        Args:
            question: Student's question
            context: RAG context
            subject: Subject area
        
        Returns:
            HintLadder object
        """
        prompt = self.build_tutoring_prompt(question, context, subject)
        response = self.generate(prompt, subject)
        hints = self.parse_hint_ladder(response)
        
        return hints
    
    def generate_tutoring_response_stream(self, question: str, context: str,
                                         subject: str) -> Iterator[str]:
        """
        Generate tutoring response with streaming.
        
        Args:
            question: Student's question
            context: RAG context
            subject: Subject area
        
        Yields:
            Text chunks
        """
        prompt = self.build_tutoring_prompt(question, context, subject)
        
        for chunk in self.generate_stream(prompt, subject):
            yield chunk


# Singleton instance
_llm_instance: Optional[LLMEngine] = None


def get_llm_engine(config_path: Optional[str] = None) -> LLMEngine:
    """
    Get or create the global LLM engine instance.
    
    Args:
        config_path: Path to config file
    
    Returns:
        LLMEngine instance
    """
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMEngine(config_path)
    return _llm_instance
