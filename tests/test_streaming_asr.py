"""Tests for StreamingASR module."""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from myweepal.audio.streaming import StreamingASR, StreamingTranscriptionSegment


class TestStreamingASR:
    """Test suite for StreamingASR."""

    @pytest.fixture
    def streaming_asr(self):
        """Create StreamingASR instance for testing."""
        with patch("myweepal.audio.streaming.StreamingASR._load_model"):
            asr = StreamingASR(
                model_id="test-model",
                sample_rate=16000,
                chunk_duration=2.0,
                use_mlx=False
            )
            # Mock the model
            asr.model = Mock()
            asr.model.transcribe = Mock(return_value={
                "text": "test transcription",
                "avg_logprob": 0.95,
                "word_timestamps": []
            })
            return asr

    def test_initialization(self, streaming_asr):
        """Test StreamingASR initialization."""
        assert streaming_asr.sample_rate == 16000
        assert streaming_asr.chunk_duration == 2.0
        assert streaming_asr.model is not None

    @pytest.mark.asyncio
    async def test_stream_transcribe(self, streaming_asr):
        """Test async stream transcription."""
        # Create mock audio stream
        async def mock_audio_stream():
            for i in range(3):
                # Generate 2 seconds of audio at 16kHz
                audio_chunk = np.random.randn(32000).astype(np.float32)
                yield audio_chunk

        # Collect results
        results = []
        async for segment in streaming_asr.stream_transcribe(mock_audio_stream()):
            results.append(segment)

        # Verify results
        assert len(results) > 0
        assert isinstance(results[0], StreamingTranscriptionSegment)
        assert results[0].text == "test transcription"

    def test_transcribe_sync(self, streaming_asr):
        """Test synchronous transcription."""
        # Generate test audio
        audio_data = np.random.randn(16000).astype(np.float32)

        # Transcribe
        result = streaming_asr._transcribe_sync(audio_data)

        # Verify
        assert result["text"] == "test transcription"
        assert result["avg_logprob"] == 0.95

    def test_voice_activity_detection(self, streaming_asr):
        """Test voice activity detection in capture."""
        # Test with silence (low energy)
        silence = np.zeros(16000).astype(np.float32)
        energy = np.sqrt(np.mean(silence**2))
        assert energy < streaming_asr.energy_threshold

        # Test with speech (higher energy)
        speech = np.random.randn(16000).astype(np.float32) * 0.5
        energy = np.sqrt(np.mean(speech**2))
        assert energy > streaming_asr.energy_threshold

    def test_start_stop_streaming(self, streaming_asr):
        """Test starting and stopping streaming."""
        callback = Mock()

        # Start streaming
        streaming_asr.start_streaming(callback)
        assert streaming_asr.is_streaming

        # Stop streaming
        streaming_asr.stop_streaming()
        assert not streaming_asr.is_streaming

    def test_get_results(self, streaming_asr):
        """Test getting transcription results from queue."""
        # Add some results to queue
        for i in range(5):
            segment = StreamingTranscriptionSegment(
                text=f"segment {i}",
                start_time=i * 2.0,
                end_time=(i + 1) * 2.0,
                confidence=0.9,
                is_final=True
            )
            streaming_asr.result_queue.put(segment)

        # Get results
        results = streaming_asr.get_results(limit=3)

        # Verify
        assert len(results) == 3
        assert results[0].text == "segment 0"
        assert results[2].text == "segment 2"

    @pytest.mark.asyncio
    async def test_transcribe_chunk_error_handling(self, streaming_asr):
        """Test error handling in chunk transcription."""
        # Set model to None to trigger error
        streaming_asr.model = None

        # Try to transcribe
        audio_data = np.random.randn(16000).astype(np.float32)
        result = await streaming_asr._transcribe_chunk(audio_data, 0.0, 1.0)

        # Should return None on error
        assert result is None

    def test_audio_normalization(self, streaming_asr):
        """Test audio normalization in transcription."""
        # Create audio with values > 1
        audio_data = np.random.randn(16000).astype(np.float32) * 2.0

        # Mock transcribe to check normalized input
        def check_normalized(audio):
            assert np.max(np.abs(audio)) <= 1.0
            return {"text": "normalized", "avg_logprob": 0.9}

        streaming_asr.model.transcribe = check_normalized

        # Transcribe
        result = streaming_asr._transcribe_sync(audio_data)
        assert result["text"] == "normalized"


class TestStreamingTranscriptionSegment:
    """Test StreamingTranscriptionSegment dataclass."""

    def test_segment_creation(self):
        """Test creating transcription segment."""
        segment = StreamingTranscriptionSegment(
            text="Hello world",
            start_time=0.0,
            end_time=2.0,
            confidence=0.95,
            is_final=True,
            word_timestamps=[{"word": "Hello", "start": 0.0, "end": 0.5}]
        )

        assert segment.text == "Hello world"
        assert segment.start_time == 0.0
        assert segment.end_time == 2.0
        assert segment.confidence == 0.95
        assert segment.is_final
        assert len(segment.word_timestamps) == 1

    def test_segment_duration(self):
        """Test calculating segment duration."""
        segment = StreamingTranscriptionSegment(
            text="Test",
            start_time=1.0,
            end_time=3.5,
            confidence=0.9,
            is_final=False
        )

        duration = segment.end_time - segment.start_time
        assert duration == 2.5