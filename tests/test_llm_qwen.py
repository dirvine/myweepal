"""Tests for Qwen LLM integration."""

import pytest
from unittest.mock import Mock, patch
import numpy as np
from myweepal.core.llm import LLMInference, LLMConfig


class TestQwenLLM:
    """Test suite for Qwen LLM integration."""

    @pytest.fixture
    def llm_config(self):
        """Create LLM configuration for testing."""
        return LLMConfig(
            model_name="mlx-community/Qwen3-4B-Thinking-2507-4bit",
            max_tokens=512,
            temperature=0.7,
            context_window=8192
        )

    @pytest.fixture
    def llm(self, llm_config):
        """Create LLM instance for testing."""
        with patch("myweepal.core.llm.load") as mock_load:
            # Mock model and tokenizer
            mock_model = Mock()
            mock_tokenizer = Mock()
            mock_load.return_value = (mock_model, mock_tokenizer)

            llm = LLMInference(llm_config)
            return llm

    def test_initialization(self, llm, llm_config):
        """Test LLM initialization with Qwen model."""
        assert llm.config.model_name == "mlx-community/Qwen3-4B-Thinking-2507-4bit"
        assert llm.config.max_tokens == 512
        assert llm.config.context_window == 8192
        assert llm.model is not None
        assert llm.tokenizer is not None

    def test_qwen_prompt_formatting(self, llm):
        """Test Qwen-specific prompt formatting."""
        user_prompt = "What is the meaning of life?"
        system_prompt = "You are a helpful assistant."

        # Format with system prompt
        formatted = llm._format_prompt(user_prompt, system_prompt)
        assert "System: You are a helpful assistant." in formatted
        assert "User: What is the meaning of life?" in formatted
        assert "Assistant:" in formatted

        # Format without system prompt
        formatted = llm._format_prompt(user_prompt)
        assert "User: What is the meaning of life?" in formatted
        assert "Assistant:" in formatted
        assert "System:" not in formatted

    @patch("myweepal.core.llm.generate")
    def test_generate_response(self, mock_generate, llm):
        """Test response generation with Qwen."""
        mock_generate.return_value = "The meaning of life is to find purpose and happiness."

        response = llm.generate_response(
            prompt="What is the meaning of life?",
            max_tokens=100,
            temperature=0.8
        )

        assert isinstance(response, str)
        assert len(response) > 0
        mock_generate.assert_called_once()

    def test_generate_response_with_system_prompt(self, llm):
        """Test response generation with system prompt."""
        with patch("myweepal.core.llm.generate") as mock_generate:
            mock_generate.return_value = "I'm MyWeePal, here to help preserve your memories."

            response = llm.generate_response(
                prompt="Who are you?",
                system_prompt="You are MyWeePal, a memory preservation assistant."
            )

            assert isinstance(response, str)
            # Check that system prompt was included in the call
            call_args = mock_generate.call_args
            assert "System: You are MyWeePal" in call_args[1]["prompt"]

    def test_extract_response_cleaning(self, llm):
        """Test response cleaning for Qwen output."""
        # Test with special tokens
        raw_response = "User: Hello\n\nAssistant: Hello! How can I help you today?"
        cleaned = llm._extract_response(raw_response)
        assert "User:" not in cleaned
        assert "Assistant:" not in cleaned

        # Test with whitespace
        raw_response = "  \n\n  This is the response.  \n\n  "
        cleaned = llm._extract_response(raw_response)
        assert cleaned == "This is the response."

    @patch("myweepal.core.llm.mx")
    def test_generate_embedding(self, mock_mx, llm):
        """Test embedding generation."""
        # Mock MLX operations
        mock_mx.array = Mock(return_value=Mock())
        mock_mx.no_grad = Mock(return_value=Mock(__enter__=Mock(), __exit__=Mock()))
        mock_mx.mean = Mock(return_value=Mock())

        # Mock tokenizer
        llm.tokenizer.encode = Mock(return_value=np.array([1, 2, 3, 4]))

        # Mock model embed_tokens
        llm.model.embed_tokens = Mock(return_value=Mock())

        embedding = llm.generate_embedding("Test text for embedding")
        assert embedding is not None
        llm.tokenizer.encode.assert_called_once_with("Test text for embedding", return_tensors="np")

    def test_batch_generate(self, llm):
        """Test batch generation with multiple prompts."""
        with patch.object(llm, "generate_response") as mock_generate:
            mock_generate.side_effect = ["Response 1", "Response 2", "Response 3"]

            prompts = [
                "Tell me about Scotland",
                "What is MyWeePal?",
                "How do I preserve memories?"
            ]

            responses = llm.batch_generate(prompts)

            assert len(responses) == 3
            assert responses[0] == "Response 1"
            assert responses[1] == "Response 2"
            assert responses[2] == "Response 3"
            assert mock_generate.call_count == 3

    def test_batch_generate_with_error(self, llm):
        """Test batch generation handles errors gracefully."""
        with patch.object(llm, "generate_response") as mock_generate:
            # Make second prompt fail
            mock_generate.side_effect = ["Response 1", Exception("Generation failed"), "Response 3"]

            prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
            responses = llm.batch_generate(prompts)

            assert len(responses) == 3
            assert responses[0] == "Response 1"
            assert responses[1] == ""  # Failed prompt returns empty string
            assert responses[2] == "Response 3"

    def test_get_model_info(self, llm):
        """Test getting model information."""
        info = llm.get_model_info()

        assert info["model_name"] == "mlx-community/Qwen3-4B-Thinking-2507-4bit"
        assert info["max_tokens"] == 512
        assert info["temperature"] == 0.7
        assert info["status"] == "ready"

    def test_get_model_info_not_initialized(self):
        """Test model info when not initialized."""
        with patch("myweepal.core.llm.load") as mock_load:
            mock_load.side_effect = Exception("Failed to load")

            try:
                llm = LLMInference()
            except:
                pass

            llm = LLMInference.__new__(LLMInference)
            llm.model = None
            llm.config = LLMConfig()

            info = llm.get_model_info()
            assert info["status"] == "not_initialized"

    def test_config_defaults(self):
        """Test LLMConfig default values for Qwen."""
        config = LLMConfig()
        assert config.model_name == "mlx-community/Qwen3-4B-Thinking-2507-4bit"
        assert config.max_tokens == 512
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.repetition_penalty == 1.1
        assert config.context_window == 8192

    def test_error_handling_no_model(self):
        """Test error handling when model not initialized."""
        llm = LLMInference.__new__(LLMInference)
        llm.model = None
        llm.tokenizer = None
        llm.config = LLMConfig()

        with pytest.raises(RuntimeError, match="Model not initialized"):
            llm.generate_response("Test prompt")

    def test_error_handling_no_tokenizer(self):
        """Test error handling when tokenizer not initialized."""
        llm = LLMInference.__new__(LLMInference)
        llm.model = Mock()
        llm.tokenizer = None
        llm.config = LLMConfig()

        with pytest.raises(RuntimeError, match="Tokenizer not initialized"):
            llm.generate_embedding("Test text")