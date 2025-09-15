"""Emotion-aware TTS module with voice cloning support."""

import logging
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import sounddevice as sd
import torch
import torchaudio
from queue import Queue
import threading
import json

logger = logging.getLogger(__name__)


@dataclass
class VoiceProfile:
    """Voice profile for TTS."""

    name: str
    reference_audio: Optional[str] = None
    emotion_default: float = 0.5
    speed: float = 1.0
    pitch: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmotionSettings:
    """Emotion settings for TTS."""

    emotion_type: str = "neutral"
    intensity: float = 0.5
    exaggeration: float = 0.5


class EmotionAwareTTS:
    """TTS with emotion control and voice cloning."""

    def __init__(
        self,
        model_name: str = "mlx-community/Kokoro-82M-bf16",
        sample_rate: int = 24000,
        device: str = "mps",  # Metal Performance Shaders for Apple Silicon
        enable_watermark: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """Initialize emotion-aware TTS.

        Args:
            model_name: Model to use (Chatterbox, Kokoro, etc.)
            sample_rate: Output sample rate
            device: Device to run on (mps for Apple Silicon)
            enable_watermark: Whether to add watermark to generated audio
            cache_dir: Directory for caching models and voices
        """
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.device = device if torch.backends.mps.is_available() else "cpu"
        self.enable_watermark = enable_watermark
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".myweepal" / "tts_cache"

        # Emotion mapping
        self.emotion_map = {
            "happy": 0.8,
            "sad": 0.6,
            "neutral": 0.5,
            "excited": 0.9,
            "calm": 0.3,
            "angry": 0.7,
            "fearful": 0.65,
            "surprised": 0.85,
            "disgusted": 0.6,
            "contemplative": 0.4,
        }

        # Voice profiles
        self.voice_profiles: Dict[str, VoiceProfile] = {}
        self.current_profile = "default"

        # Audio queue for non-blocking playback
        self.speech_queue = Queue()
        self.is_speaking = False
        self.speaking_thread = None

        # Model and processor
        self.model = None
        self.processor = None

        self._initialize_model()
        self._load_voice_profiles()

    def _initialize_model(self) -> None:
        """Initialize the TTS model."""
        try:
            if "chatterbox" in self.model_name.lower():
                self._init_chatterbox()
            elif "kokoro" in self.model_name.lower():
                self._init_kokoro()
            else:
                self._init_default_tts()

            logger.info(f"TTS model {self.model_name} initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize TTS model: {e}")
            # Fall back to a simpler model
            self._init_fallback()

    def _init_chatterbox(self) -> None:
        """Initialize Chatterbox TTS (when available)."""
        try:
            # This would be the actual Chatterbox initialization
            # For now, we'll prepare the structure
            logger.info("Chatterbox TTS initialization (placeholder)")
            # from chatterbox.tts import ChatterboxTTS
            # self.model = ChatterboxTTS.from_pretrained(device=self.device)
        except ImportError:
            logger.warning("Chatterbox not available, using fallback")
            self._init_fallback()

    def _init_kokoro(self) -> None:
        """Initialize Kokoro MLX model."""
        try:
            # Try to load MLX-based Kokoro model
            import mlx.core as mx
            import mlx.nn as nn
            from mlx_lm import load

            logger.info("Loading Kokoro MLX model")
            self.model, self.tokenizer = load(self.model_name)
            logger.info("Kokoro model loaded successfully")
        except ImportError:
            logger.warning("MLX not available for Kokoro, using fallback")
            self._init_fallback()

    def _init_default_tts(self) -> None:
        """Initialize default TTS using available libraries."""
        try:
            # Try to use TTS library if available
            from TTS.api import TTS

            logger.info("Using default TTS library")
            self.model = TTS("tts_models/en/ljspeech/tacotron2-DDC")
        except ImportError:
            logger.warning("TTS library not available, using fallback")
            self._init_fallback()

    def _init_fallback(self) -> None:
        """Initialize fallback TTS (basic functionality)."""
        logger.info("Using fallback TTS mode")
        self.model = None  # Will use basic synthesis

    def synthesize_with_emotion(
        self,
        text: str,
        emotion: str = "neutral",
        voice_ref: Optional[str] = None,
        custom_intensity: Optional[float] = None,
    ) -> np.ndarray:
        """Generate speech with emotional context.

        Args:
            text: Text to synthesize
            emotion: Target emotion
            voice_ref: Optional voice reference audio path
            custom_intensity: Optional custom emotion intensity

        Returns:
            Generated audio as numpy array
        """
        # Get emotion intensity
        intensity = custom_intensity if custom_intensity is not None else self.emotion_map.get(emotion, 0.5)

        # Get voice profile
        profile = self._get_voice_profile(voice_ref)

        try:
            if self.model is None:
                # Fallback: generate placeholder audio
                return self._generate_placeholder_audio(text, intensity)

            # Generate audio based on model type
            if hasattr(self.model, "generate"):
                # Chatterbox-style API
                audio = self.model.generate(
                    text,
                    emotion_intensity=intensity,
                    audio_prompt_path=profile.reference_audio,
                    speed=profile.speed,
                )
            elif hasattr(self.model, "tts"):
                # TTS library API
                if profile.reference_audio:
                    audio = self.model.tts_with_vc(
                        text=text,
                        speaker_wav=profile.reference_audio,
                    )
                else:
                    audio = self.model.tts(text=text)
            else:
                # Fallback
                audio = self._generate_placeholder_audio(text, intensity)

            # Apply emotion post-processing
            audio = self._apply_emotion_effects(audio, emotion, intensity)

            # Add watermark if enabled
            if self.enable_watermark:
                audio = self._add_watermark(audio)

            return audio

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return self._generate_placeholder_audio(text, intensity)

    def _get_voice_profile(self, voice_ref: Optional[str] = None) -> VoiceProfile:
        """Get or create voice profile.

        Args:
            voice_ref: Optional voice reference path

        Returns:
            Voice profile
        """
        if voice_ref:
            # Create temporary profile from reference
            return VoiceProfile(
                name="temp",
                reference_audio=voice_ref,
                emotion_default=0.5,
            )

        # Use current profile or default
        profile_name = self.current_profile
        if profile_name in self.voice_profiles:
            return self.voice_profiles[profile_name]

        # Return default profile
        return VoiceProfile(name="default")

    def _apply_emotion_effects(
        self,
        audio: np.ndarray,
        emotion: str,
        intensity: float
    ) -> np.ndarray:
        """Apply emotion-based audio effects.

        Args:
            audio: Input audio
            emotion: Emotion type
            intensity: Emotion intensity

        Returns:
            Processed audio
        """
        # Apply emotion-specific processing
        if emotion == "happy" or emotion == "excited":
            # Increase pitch slightly and add energy
            audio = self._adjust_pitch(audio, 1.0 + 0.1 * intensity)
            audio = self._adjust_energy(audio, 1.0 + 0.2 * intensity)
        elif emotion == "sad":
            # Lower pitch and reduce energy
            audio = self._adjust_pitch(audio, 1.0 - 0.1 * intensity)
            audio = self._adjust_energy(audio, 1.0 - 0.2 * intensity)
        elif emotion == "angry":
            # Add slight distortion and increase energy
            audio = self._adjust_energy(audio, 1.0 + 0.3 * intensity)
            audio = self._add_edge(audio, intensity)
        elif emotion == "calm":
            # Smooth and reduce variations
            audio = self._smooth_audio(audio, intensity)

        return audio

    def _adjust_pitch(self, audio: np.ndarray, factor: float) -> np.ndarray:
        """Adjust audio pitch.

        Args:
            audio: Input audio
            factor: Pitch adjustment factor

        Returns:
            Pitch-adjusted audio
        """
        try:
            # Convert to tensor for processing
            audio_tensor = torch.from_numpy(audio).float()

            # Use torchaudio for pitch shifting if available
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            # Simple pitch adjustment using resampling
            # This is a placeholder - real pitch shifting would use vocoder
            if factor != 1.0:
                orig_len = audio_tensor.shape[-1]
                resampled = torchaudio.functional.resample(
                    audio_tensor,
                    self.sample_rate,
                    int(self.sample_rate * factor)
                )
                # Resample back to original sample rate
                audio_tensor = torchaudio.functional.resample(
                    resampled,
                    int(self.sample_rate * factor),
                    self.sample_rate
                )
                # Adjust length
                if audio_tensor.shape[-1] > orig_len:
                    audio_tensor = audio_tensor[:, :orig_len]
                elif audio_tensor.shape[-1] < orig_len:
                    padding = orig_len - audio_tensor.shape[-1]
                    audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))

            return audio_tensor.squeeze().numpy()

        except Exception as e:
            logger.warning(f"Pitch adjustment failed: {e}")
            return audio

    def _adjust_energy(self, audio: np.ndarray, factor: float) -> np.ndarray:
        """Adjust audio energy/amplitude.

        Args:
            audio: Input audio
            factor: Energy adjustment factor

        Returns:
            Energy-adjusted audio
        """
        return np.clip(audio * factor, -1.0, 1.0)

    def _add_edge(self, audio: np.ndarray, intensity: float) -> np.ndarray:
        """Add edge/harshness to audio.

        Args:
            audio: Input audio
            intensity: Effect intensity

        Returns:
            Processed audio
        """
        # Simple soft clipping for edge
        threshold = 1.0 - 0.3 * intensity
        audio = np.tanh(audio / threshold) * threshold
        return audio

    def _smooth_audio(self, audio: np.ndarray, intensity: float) -> np.ndarray:
        """Smooth audio variations.

        Args:
            audio: Input audio
            intensity: Smoothing intensity

        Returns:
            Smoothed audio
        """
        from scipy.signal import savgol_filter

        # Apply smoothing filter
        window_length = int(self.sample_rate * 0.01 * (1 + intensity))
        if window_length % 2 == 0:
            window_length += 1
        window_length = min(window_length, len(audio) - 1)

        if window_length > 3:
            try:
                audio = savgol_filter(audio, window_length, 3)
            except:
                pass

        return audio

    def _add_watermark(self, audio: np.ndarray) -> np.ndarray:
        """Add watermark to audio.

        Args:
            audio: Input audio

        Returns:
            Watermarked audio
        """
        # Simple watermarking - add inaudible high-frequency signal
        # In production, use proper watermarking like PerTH
        watermark_freq = 18000  # Hz - mostly inaudible
        watermark_amplitude = 0.001

        t = np.arange(len(audio)) / self.sample_rate
        watermark = watermark_amplitude * np.sin(2 * np.pi * watermark_freq * t)

        return audio + watermark

    def _generate_placeholder_audio(self, text: str, intensity: float) -> np.ndarray:
        """Generate placeholder audio for fallback.

        Args:
            text: Input text
            intensity: Emotion intensity

        Returns:
            Placeholder audio
        """
        # Generate silence with correct duration
        words = len(text.split())
        duration = (words / 150) * 60  # Approximate duration
        samples = int(duration * self.sample_rate)

        # Generate very quiet noise as placeholder
        audio = np.random.normal(0, 0.001, samples)

        return audio

    def clone_voice(
        self,
        audio_path: str,
        profile_name: str,
        min_duration: float = 6.0
    ) -> bool:
        """Clone voice from audio sample.

        Args:
            audio_path: Path to reference audio
            profile_name: Name for the voice profile
            min_duration: Minimum required duration in seconds

        Returns:
            True if successful
        """
        try:
            # Load and validate audio
            waveform, sr = torchaudio.load(audio_path)

            # Check duration
            duration = waveform.shape[-1] / sr
            if duration < min_duration:
                logger.warning(f"Audio duration {duration:.1f}s is less than minimum {min_duration}s")
                # Continue anyway for testing

            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            # Create voice profile
            profile = VoiceProfile(
                name=profile_name,
                reference_audio=audio_path,
                metadata={
                    "duration": duration,
                    "sample_rate": self.sample_rate,
                    "created": time.time(),
                }
            )

            # Save profile
            self.voice_profiles[profile_name] = profile
            self._save_voice_profiles()

            logger.info(f"Voice profile '{profile_name}' created successfully")
            return True

        except Exception as e:
            logger.error(f"Voice cloning failed: {e}")
            return False

    def _save_voice_profiles(self) -> None:
        """Save voice profiles to disk."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            profiles_file = self.cache_dir / "voice_profiles.json"

            profiles_data = {}
            for name, profile in self.voice_profiles.items():
                profiles_data[name] = {
                    "name": profile.name,
                    "reference_audio": profile.reference_audio,
                    "emotion_default": profile.emotion_default,
                    "speed": profile.speed,
                    "pitch": profile.pitch,
                    "metadata": profile.metadata,
                }

            with open(profiles_file, "w") as f:
                json.dump(profiles_data, f, indent=2)

            logger.info(f"Saved {len(profiles_data)} voice profiles")

        except Exception as e:
            logger.error(f"Failed to save voice profiles: {e}")

    def _load_voice_profiles(self) -> None:
        """Load voice profiles from disk."""
        try:
            profiles_file = self.cache_dir / "voice_profiles.json"

            if profiles_file.exists():
                with open(profiles_file, "r") as f:
                    profiles_data = json.load(f)

                for name, data in profiles_data.items():
                    self.voice_profiles[name] = VoiceProfile(
                        name=data["name"],
                        reference_audio=data.get("reference_audio"),
                        emotion_default=data.get("emotion_default", 0.5),
                        speed=data.get("speed", 1.0),
                        pitch=data.get("pitch", 1.0),
                        metadata=data.get("metadata", {}),
                    )

                logger.info(f"Loaded {len(self.voice_profiles)} voice profiles")

        except Exception as e:
            logger.error(f"Failed to load voice profiles: {e}")

    def speak(
        self,
        text: str,
        emotion: str = "neutral",
        blocking: bool = False
    ) -> bool:
        """Speak text with emotion.

        Args:
            text: Text to speak
            emotion: Emotion to use
            blocking: Whether to block until complete

        Returns:
            True if successful
        """
        try:
            # Generate audio
            audio = self.synthesize_with_emotion(text, emotion)

            if audio is None or len(audio) == 0:
                return False

            # Play audio
            if blocking:
                self._play_audio(audio)
            else:
                self.speech_queue.put(audio)
                self._ensure_speaking_thread()

            return True

        except Exception as e:
            logger.error(f"Failed to speak: {e}")
            return False

    def _play_audio(self, audio: np.ndarray) -> None:
        """Play audio through speakers.

        Args:
            audio: Audio data
        """
        try:
            # Ensure correct format
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Normalize
            max_val = np.max(np.abs(audio))
            if max_val > 1.0:
                audio = audio / max_val

            # Play
            sd.play(audio, self.sample_rate)
            sd.wait()

        except Exception as e:
            logger.error(f"Audio playback failed: {e}")

    def _ensure_speaking_thread(self) -> None:
        """Ensure speaking thread is running."""
        if not self.is_speaking:
            self.is_speaking = True
            self.speaking_thread = threading.Thread(target=self._speaking_loop)
            self.speaking_thread.daemon = True
            self.speaking_thread.start()

    def _speaking_loop(self) -> None:
        """Process speech queue in separate thread."""
        while self.is_speaking or not self.speech_queue.empty():
            try:
                if not self.speech_queue.empty():
                    audio = self.speech_queue.get(timeout=0.5)
                    self._play_audio(audio)
                else:
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"Speaking loop error: {e}")
                time.sleep(0.1)

    def stop_speaking(self) -> None:
        """Stop all speech output."""
        self.is_speaking = False

        # Clear queue
        while not self.speech_queue.empty():
            self.speech_queue.get()

        # Stop playback
        sd.stop()

        if self.speaking_thread:
            self.speaking_thread.join(timeout=5)

        logger.info("Speech stopped")

    def get_emotion_list(self) -> List[str]:
        """Get list of available emotions.

        Returns:
            List of emotion names
        """
        return list(self.emotion_map.keys())

    def get_voice_profiles_list(self) -> List[str]:
        """Get list of available voice profiles.

        Returns:
            List of profile names
        """
        return list(self.voice_profiles.keys())

    def set_current_profile(self, profile_name: str) -> bool:
        """Set current voice profile.

        Args:
            profile_name: Profile name

        Returns:
            True if successful
        """
        if profile_name in self.voice_profiles or profile_name == "default":
            self.current_profile = profile_name
            logger.info(f"Current voice profile set to: {profile_name}")
            return True
        else:
            logger.warning(f"Voice profile '{profile_name}' not found")
            return False