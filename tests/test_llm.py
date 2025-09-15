"""Tests for LLM inference module."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from myweepal.core.llm import LLMInference, LLMConfig


class TestLLMInference:
    """Test suite for LLM inference."""

    @pytest.fixture
    def llm_config(self):
        """Create test LLM config."""
        return LLMConfig(
            model_name="test-model",
            max_tokens=100,
            temperature=0.5,
        )

    @pytest.fixture
    def mock_llm(self, llm_config):
        """Create mocked LLM instance."""
        with patch("myweepal.core.llm.load") as mock_load:
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_load.return_value = (mock_model, mock_tokenizer)

            llm = LLMInference(llm_config)
            llm.model = mock_model
            llm.tokenizer = mock_tokenizer
            return llm

    def test_initialization(self, llm_config):
        """Test LLM initialization."""
        with patch("myweepal.core.llm.load") as mock_load:
            mock_model = MagicMock()
            mock_tokenizer = MagicMock()
            mock_load.return_value = (mock_model, mock_tokenizer)

            llm = LLMInference(llm_config)

            assert llm.config == llm_config
            assert llm.model is not None
            assert llm.tokenizer is not None
            mock_load.assert_called_once_with(llm_config.model_name)

    def test_initialization_failure(self, llm_config):
        """Test LLM initialization failure handling."""
        with patch("myweepal.core.llm.load") as mock_load:
            mock_load.side_effect = Exception("Model load failed")

            with pytest.raises(RuntimeError) as exc_info:
                LLMInference(llm_config)

            assert "Model initialization failed" in str(exc_info.value)

    def test_generate_response(self, mock_llm):
        """Test response generation."""
        with patch("myweepal.core.llm.generate") as mock_generate:
            mock_generate.return_value = "Generated response"

            response = mock_llm.generate_response("Test prompt")

            assert response == "Generated response"
            mock_generate.assert_called_once()

    def test_generate_response_with_system_prompt(self, mock_llm):
        """Test response generation with system prompt."""
        with patch("myweepal.core.llm.generate") as mock_generate:
            mock_generate.return_value = "Generated response"

            response = mock_llm.generate_response(
                "User prompt",
                system_prompt="System instructions",
            )

            assert response == "Generated response"
            # Check that system prompt was included in formatted prompt
            call_args = mock_generate.call_args[1]["prompt"]
            assert "System instructions" in call_args
            assert "User prompt" in call_args

    def test_generate_response_failure(self, mock_llm):
        """Test response generation failure handling."""
        with patch("myweepal.core.llm.generate") as mock_generate:
            mock_generate.side_effect = Exception("Generation failed")

            with pytest.raises(RuntimeError) as exc_info:
                mock_llm.generate_response("Test prompt")

            assert "Failed to generate response" in str(exc_info.value)

    def test_generate_embedding(self, mock_llm):
        """Test embedding generation."""
        mock_tokens = np.array([[1, 2, 3, 4]])
        mock_llm.tokenizer.encode.return_value = mock_tokens

        mock_embeddings = MagicMock()
        mock_embeddings.__array__ = lambda self, dtype=None: np.random.randn(1, 4, 768)
        mock_llm.model.embed_tokens.return_value = mock_embeddings

        with patch("myweepal.core.llm.mx.mean") as mock_mean:
            mock_mean.return_value = MagicMock()

            embedding = mock_llm.generate_embedding("Test text")

            assert embedding is not None
            mock_llm.tokenizer.encode.assert_called_once_with("Test text", return_tensors="np")

    def test_batch_generate(self, mock_llm):
        """Test batch generation."""
        with patch.object(mock_llm, "generate_response") as mock_gen:
            mock_gen.side_effect = ["Response 1", "Response 2", "Response 3"]

            prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
            responses = mock_llm.batch_generate(prompts)

            assert len(responses) == 3
            assert responses == ["Response 1", "Response 2", "Response 3"]
            assert mock_gen.call_count == 3

    def test_batch_generate_with_failure(self, mock_llm):
        """Test batch generation with partial failure."""
        with patch.object(mock_llm, "generate_response") as mock_gen:
            mock_gen.side_effect = ["Response 1", Exception("Failed"), "Response 3"]

            prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
            responses = mock_llm.batch_generate(prompts)

            assert len(responses) == 3
            assert responses[0] == "Response 1"
            assert responses[1] == ""  # Failed response
            assert responses[2] == "Response 3"

    def test_format_prompt(self, mock_llm):
        """Test prompt formatting."""
        # Without system prompt
        formatted = mock_llm._format_prompt("User message")
        assert "<|user|>User message<|assistant|>" in formatted

        # With system prompt
        formatted = mock_llm._format_prompt("User message", "System message")
        assert "<|system|>System message" in formatted
        assert "<|user|>User message" in formatted
        assert "<|assistant|>" in formatted

    def test_extract_response(self, mock_llm):
        """Test response extraction and cleaning."""
        raw_response = "<|assistant|>This is the response<|endoftext|>"
        clean = mock_llm._extract_response(raw_response)

        assert clean == "This is the response"
        assert "<|assistant|>" not in clean
        assert "<|endoftext|>" not in clean

    def test_get_model_info(self, mock_llm):
        """Test getting model information."""
        info = mock_llm.get_model_info()

        assert info["status"] == "ready"
        assert info["model_name"] == "test-model"
        assert info["max_tokens"] == 100
        assert info["temperature"] == 0.5

    def test_get_model_info_not_initialized(self):
        """Test getting model info when not initialized."""
        llm = LLMInference.__new__(LLMInference)
        llm.model = None

        info = llm.get_model_info()

        assert info["status"] == "not_initialized"