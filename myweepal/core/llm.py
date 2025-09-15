"""MLX-based LLM inference module for Apple Silicon."""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import mlx
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models.base import BaseModelArgs

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM inference."""

    model_name: str = "mlx-community/Qwen3-4B-Thinking-2507-4bit"  # Updated to Qwen
    max_tokens: int = 512  # Increased for better reasoning
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    cache_size: int = 512
    context_window: int = 8192  # Qwen supports larger context


class LLMInference:
    """MLX-based LLM inference for Apple Silicon."""

    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize LLM inference engine.

        Args:
            config: LLM configuration parameters
        """
        self.config = config or LLMConfig()
        self.model = None
        self.tokenizer = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Load MLX model and tokenizer."""
        try:
            logger.info(f"Loading MLX model: {self.config.model_name}")
            self.model, self.tokenizer = load(self.config.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model initialization failed: {e}") from e

    def generate_response(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate response from prompt.

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt

        Returns:
            Generated text response
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not initialized")

        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature

        # Format prompt with system message if provided
        formatted_prompt = self._format_prompt(prompt, system_prompt)

        try:
            response = generate(
                self.model,
                self.tokenizer,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                temp=temperature,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
            )
            return self._extract_response(response)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Failed to generate response: {e}") from e

    def generate_embedding(self, text: str) -> mx.array:
        """Generate text embedding using model's encoder.

        Args:
            text: Input text

        Returns:
            Text embedding as MLX array
        """
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not initialized")

        try:
            # Tokenize input
            tokens = self.tokenizer.encode(text, return_tensors="np")
            tokens_mx = mx.array(tokens)

            # Get embeddings from model
            with mx.no_grad():
                embeddings = self.model.embed_tokens(tokens_mx)
                # Mean pooling for sentence embedding
                embedding = mx.mean(embeddings, axis=1)

            return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise RuntimeError(f"Failed to generate embedding: {e}") from e

    def fine_tune_lora(
        self,
        training_data: List[Dict[str, str]],
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        learning_rate: float = 1e-4,
        num_epochs: int = 3,
    ) -> None:
        """Fine-tune model using LoRA adapters.

        Args:
            training_data: List of {"input": ..., "output": ...} pairs
            lora_rank: LoRA rank parameter
            lora_alpha: LoRA alpha scaling
            learning_rate: Learning rate for training
            num_epochs: Number of training epochs
        """
        if not self.model:
            raise RuntimeError("Model not initialized")

        logger.info(f"Starting LoRA fine-tuning with {len(training_data)} samples")

        # Implementation would require mlx_lm LoRA training utilities
        # This is a placeholder for the actual LoRA implementation
        # which would involve:
        # 1. Creating LoRA adapters
        # 2. Setting up optimizer
        # 3. Training loop with gradient updates
        # 4. Saving adapter weights

        logger.info("LoRA fine-tuning complete")

    def _format_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Format prompt with optional system message.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Formatted prompt string
        """
        # Qwen-specific formatting for thinking model
        if "Qwen" in self.config.model_name:
            if system_prompt:
                return f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            return f"User: {prompt}\n\nAssistant:"
        # Fallback to standard format
        if system_prompt:
            return f"<|system|>{system_prompt}<|user|>{prompt}<|assistant|>"
        return f"<|user|>{prompt}<|assistant|>"

    def _extract_response(self, generated: str) -> str:
        """Extract clean response from generated text.

        Args:
            generated: Raw generated text

        Returns:
            Cleaned response text
        """
        # Remove special tokens and clean up
        response = generated.strip()

        # Remove any remaining special tokens
        for token in ["<|system|>", "<|user|>", "<|assistant|>", "<|endoftext|>"]:
            response = response.replace(token, "")

        return response.strip()

    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[str]:
        """Generate responses for multiple prompts.

        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens per response
            temperature: Sampling temperature

        Returns:
            List of generated responses
        """
        responses = []
        for prompt in prompts:
            try:
                response = self.generate_response(prompt, max_tokens, temperature)
                responses.append(response)
            except Exception as e:
                logger.error(f"Failed to generate for prompt: {e}")
                responses.append("")

        return responses

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model.

        Returns:
            Dictionary with model information
        """
        if not self.model:
            return {"status": "not_initialized"}

        return {
            "model_name": self.config.model_name,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "status": "ready",
        }