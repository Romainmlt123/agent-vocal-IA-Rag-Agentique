"""
Local LLM Service for Pipecat using Ollama.
Provides local language model inference without API calls.
"""

import asyncio
import json
from typing import Optional, Dict, Any, List
import aiohttp

from pipecat.frames.frames import Frame, TextFrame, LLMMessagesFrame, ErrorFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.ai_services import LLMService
from loguru import logger


class LocalLLMService(LLMService):
    """
    Local LLM service using Ollama for inference.
    
    Features:
    - Multiple model support (llama3.2, mistral, phi, etc.)
    - Streaming responses
    - Context management
    - No API costs
    - GPU acceleration
    
    Args:
        model: Ollama model name (e.g., 'llama3.2:1b', 'mistral')
        base_url: Ollama API URL (default: http://localhost:11434)
        temperature: Sampling temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        system_prompt: System prompt for the model
    """
    
    def __init__(
        self,
        model: str = "llama3.2:1b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 512,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or "Tu es un assistant pédagogique IA."
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._conversation_history: List[Dict[str, str]] = []
        
        logger.info(f"LocalLLMService initialized (model: {model}, url: {base_url})")
    
    async def _ensure_session(self):
        """Ensure aiohttp session is created."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
    
    async def process_frame(self, frame: Frame, direction: str = "down"):
        """
        Process text frames and generate LLM responses.
        
        Args:
            frame: Input frame (TextFrame or LLMMessagesFrame expected)
            direction: Processing direction
        """
        await super().process_frame(frame, direction)
        
        # Handle TextFrame
        if isinstance(frame, TextFrame):
            try:
                # Generate response
                response = await self._generate(frame.text)
                
                if response.strip():
                    # Create response frame
                    response_frame = TextFrame(text=response)
                    await self.push_frame(response_frame, direction)
                    logger.debug(f"LLM response: {response[:100]}...")
                
            except Exception as e:
                logger.error(f"LLM error: {e}")
                error_frame = ErrorFrame(error=str(e))
                await self.push_frame(error_frame, direction)
        
        # Handle LLMMessagesFrame
        elif isinstance(frame, LLMMessagesFrame):
            try:
                messages = frame.messages
                response = await self._generate_from_messages(messages)
                
                if response.strip():
                    response_frame = TextFrame(text=response)
                    await self.push_frame(response_frame, direction)
                
            except Exception as e:
                logger.error(f"LLM messages error: {e}")
                error_frame = ErrorFrame(error=str(e))
                await self.push_frame(error_frame, direction)
        
        else:
            # Pass through other frames
            await self.push_frame(frame, direction)
    
    async def _generate(self, user_message: str, context: Optional[str] = None) -> str:
        """
        Generate response from user message.
        
        Args:
            user_message: User's input text
            context: Optional context from RAG
            
        Returns:
            Generated response text
        """
        await self._ensure_session()
        
        # Build prompt with context if provided
        if context:
            prompt = f"""Contexte pertinent :
{context}

Question de l'utilisateur : {user_message}

Réponds de manière pédagogique en utilisant le contexte fourni."""
        else:
            prompt = user_message
        
        # Add to conversation history
        self._conversation_history.append({"role": "user", "content": prompt})
        
        # Prepare request payload
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                *self._conversation_history
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        # Call Ollama API
        url = f"{self.base_url}/api/chat"
        
        async with self._session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Ollama API error ({response.status}): {error_text}")
            
            result = await response.json()
            assistant_message = result.get("message", {}).get("content", "")
            
            # Add to conversation history
            self._conversation_history.append({"role": "assistant", "content": assistant_message})
            
            return assistant_message
    
    async def _generate_from_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate response from message history.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            
        Returns:
            Generated response text
        """
        await self._ensure_session()
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                *messages
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        url = f"{self.base_url}/api/chat"
        
        async with self._session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Ollama API error: {error_text}")
            
            result = await response.json()
            return result.get("message", {}).get("content", "")
    
    async def generate_streaming(self, user_message: str, context: Optional[str] = None):
        """
        Generate streaming response (yields tokens as they arrive).
        
        Args:
            user_message: User's input text
            context: Optional RAG context
            
        Yields:
            Generated tokens
        """
        await self._ensure_session()
        
        if context:
            prompt = f"Contexte: {context}\n\nQuestion: {user_message}"
        else:
            prompt = user_message
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        url = f"{self.base_url}/api/chat"
        
        async with self._session.post(url, json=payload) as response:
            async for line in response.content:
                if line:
                    try:
                        chunk = json.loads(line)
                        if "message" in chunk:
                            content = chunk["message"].get("content", "")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue
    
    def clear_history(self):
        """Clear conversation history."""
        self._conversation_history.clear()
        logger.info("Conversation history cleared")
    
    def set_system_prompt(self, prompt: str):
        """Update system prompt."""
        self.system_prompt = prompt
        logger.info(f"System prompt updated: {prompt[:50]}...")
    
    async def cleanup(self):
        """Cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()
        
        self._conversation_history.clear()
        logger.info("LocalLLMService cleaned up")


class StreamingLLMService(LocalLLMService):
    """
    Streaming variant of LLMService that yields tokens in real-time.
    Optimized for conversational applications.
    """
    
    async def process_frame(self, frame: Frame, direction: str = "down"):
        """Process frames with streaming response."""
        await super().process_frame(frame, direction)
        
        if isinstance(frame, TextFrame):
            try:
                # Stream response token by token
                full_response = []
                
                async for token in self.generate_streaming(frame.text):
                    # Push each token as it arrives
                    token_frame = TextFrame(text=token)
                    await self.push_frame(token_frame, direction)
                    full_response.append(token)
                
                logger.debug(f"Streamed: {''.join(full_response)[:100]}...")
                
            except Exception as e:
                logger.error(f"Streaming LLM error: {e}")
                error_frame = ErrorFrame(error=str(e))
                await self.push_frame(error_frame, direction)
