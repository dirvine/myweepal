"""Tests for EmotionAwareTTS module."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
import tempfile
from myweepal.audio.emotion_tts import EmotionAwareTTS, VoiceProfile, EmotionSettings


class TestEmotionAwareTTS:
    """Test suite for EmotionAwareTTS."""

    @pytest.fixture
    def tts(self):
        """Create EmotionAwareTTS instance for testing."""
        with patch("myweepal.audio.emotion_tts.EmotionAwareTTS._initialize_model"):
            tts = EmotionAwareTTS(
                model_name="test-model",
                sample_rate=24000,
                device="cpu",
                cache_dir=tempfile.mkdtemp()
            )
            # Mock the model
            tts.model = Mock()
            tts.model.generate = Mock(return_value=np.random.randn(24000))
            return tts

    def test_initialization(self, tts):
        """Test EmotionAwareTTS initialization."""
        assert tts.sample_rate == 24000
        assert tts.device == "cpu"
        assert len(tts.emotion_map) > 0
        assert "happy" in tts.emotion_map
        assert "sad" in tts.emotion_map

    def test_synthesize_with_emotion(self, tts):
        """Test synthesis with different emotions."""
        text = "Hello, this is a test."

        # Test different emotions
        for emotion in ["happy", "sad", "neutral", "excited"]:
            audio = tts.synthesize_with_emotion(text, emotion=emotion)
            assert isinstance(audio, np.ndarray)
            assert len(audio) > 0

    def test_emotion_intensity_mapping(self, tts):
        """Test emotion to intensity mapping."""
        assert tts.emotion_map["happy"] == 0.8
        assert tts.emotion_map["sad"] == 0.6
        assert tts.emotion_map["neutral"] == 0.5
        assert tts.emotion_map["excited"] == 0.9
        assert tts.emotion_map["calm"] == 0.3

    def test_voice_profile_creation(self, tts):
        """Test creating voice profiles."""
        profile = VoiceProfile(
            name="test_voice",
            reference_audio="/path/to/audio.wav",
            emotion_default=0.7,
            speed=1.1,
            pitch=0.95
        )

        assert profile.name == "test_voice"
        assert profile.reference_audio == "/path/to/audio.wav"
        assert profile.emotion_default == 0.7
        assert profile.speed == 1.1
        assert profile.pitch == 0.95

    @patch("torchaudio.load")
    def test_clone_voice(self, mock_load, tts):
        """Test voice cloning functionality."""
        # Mock audio loading
        mock_waveform = np.random.randn(1, 24000 * 10)  # 10 seconds
        mock_load.return_value = (mock_waveform, 24000)

        # Clone voice
        success = tts.clone_voice(
            audio_path="/path/to/reference.wav",
            profile_name="cloned_voice",
            min_duration=6.0
        )

        assert success
        assert "cloned_voice" in tts.voice_profiles
        profile = tts.voice_profiles["cloned_voice"]
        assert profile.name == "cloned_voice"
        assert profile.reference_audio == "/path/to/reference.wav"

    def test_emotion_effects(self, tts):
        """Test emotion-based audio effects."""
        audio = np.random.randn(24000).astype(np.float32)

        # Test happy effect (increased pitch and energy)
        happy_audio = tts._apply_emotion_effects(audio, "happy", 0.8)
        assert isinstance(happy_audio, np.ndarray)

        # Test sad effect (lowered pitch and energy)
        sad_audio = tts._apply_emotion_effects(audio, "sad", 0.6)
        assert isinstance(sad_audio, np.ndarray)

        # Test angry effect (increased energy)
        angry_audio = tts._apply_emotion_effects(audio, "angry", 0.7)
        assert isinstance(angry_audio, np.ndarray)

    def test_adjust_pitch(self, tts):
        """Test pitch adjustment."""
        audio = np.random.randn(24000).astype(np.float32)

        # Increase pitch
        high_pitch = tts._adjust_pitch(audio, 1.2)
        assert isinstance(high_pitch, np.ndarray)

        # Decrease pitch
        low_pitch = tts._adjust_pitch(audio, 0.8)
        assert isinstance(low_pitch, np.ndarray)

    def test_adjust_energy(self, tts):
        """Test energy/amplitude adjustment."""
        audio = np.random.randn(24000).astype(np.float32) * 0.5

        # Increase energy
        high_energy = tts._adjust_energy(audio, 1.5)
        assert np.mean(np.abs(high_energy)) > np.mean(np.abs(audio))

        # Decrease energy
        low_energy = tts._adjust_energy(audio, 0.5)
        assert np.mean(np.abs(low_energy)) < np.mean(np.abs(audio))

        # Check clipping
        assert np.max(np.abs(high_energy)) <= 1.0

    def test_watermarking(self, tts):
        """Test audio watermarking."""
        audio = np.random.randn(24000).astype(np.float32) * 0.5
        tts.enable_watermark = True

        watermarked = tts._add_watermark(audio)

        assert isinstance(watermarked, np.ndarray)
        assert len(watermarked) == len(audio)
        # Watermark should be subtle
        difference = np.mean(np.abs(watermarked - audio))
        assert difference < 0.01

    def test_save_load_voice_profiles(self, tts):
        """Test saving and loading voice profiles."""
        # Add test profiles
        tts.voice_profiles["test1"] = VoiceProfile(
            name="test1",
            reference_audio="/path/1.wav"
        )
        tts.voice_profiles["test2"] = VoiceProfile(
            name="test2",
            reference_audio="/path/2.wav"
        )

        # Save profiles
        tts._save_voice_profiles()

        # Clear and reload
        tts.voice_profiles.clear()
        tts._load_voice_profiles()

        # Check loaded profiles
        assert "test1" in tts.voice_profiles
        assert "test2" in tts.voice_profiles
        assert tts.voice_profiles["test1"].reference_audio == "/path/1.wav"

    def test_speak_blocking(self, tts):
        """Test blocking speech synthesis."""
        with patch.object(tts, "_play_audio") as mock_play:
            success = tts.speak("Test speech", emotion="happy", blocking=True)
            assert success
            mock_play.assert_called_once()

    def test_speak_non_blocking(self, tts):
        """Test non-blocking speech synthesis."""
        with patch.object(tts, "_ensure_speaking_thread") as mock_thread:
            success = tts.speak("Test speech", emotion="neutral", blocking=False)
            assert success
            mock_thread.assert_called_once()
            assert not tts.speech_queue.empty()

    def test_stop_speaking(self, tts):
        """Test stopping speech output."""
        # Add items to queue
        for i in range(3):
            tts.speech_queue.put(np.random.randn(1000))

        # Start speaking thread
        tts.is_speaking = True

        # Stop speaking
        tts.stop_speaking()

        assert not tts.is_speaking
        assert tts.speech_queue.empty()

    def test_get_emotion_list(self, tts):
        """Test getting available emotions."""
        emotions = tts.get_emotion_list()
        assert isinstance(emotions, list)
        assert "happy" in emotions
        assert "sad" in emotions
        assert "neutral" in emotions
        assert len(emotions) == len(tts.emotion_map)

    def test_set_current_profile(self, tts):
        """Test setting current voice profile."""
        # Add a test profile
        tts.voice_profiles["custom"] = VoiceProfile(name="custom")

        # Set valid profile
        success = tts.set_current_profile("custom")
        assert success
        assert tts.current_profile == "custom"

        # Set invalid profile
        success = tts.set_current_profile("nonexistent")
        assert not success
        assert tts.current_profile == "custom"  # Should not change

    def test_fallback_generation(self, tts):
        """Test fallback audio generation when model fails."""
        # Set model to None to trigger fallback
        tts.model = None

        audio = tts.synthesize_with_emotion("Test", emotion="neutral")
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0

    def test_emotion_settings_dataclass(self):
        """Test EmotionSettings dataclass."""
        settings = EmotionSettings(
            emotion_type="happy",
            intensity=0.8,
            exaggeration=0.7
        )

        assert settings.emotion_type == "happy"
        assert settings.intensity == 0.8
        assert settings.exaggeration == 0.7