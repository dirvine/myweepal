"""Text-to-speech module with voice cloning support."""

import logging
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
import sounddevice as sd
from TTS.api import TTS
import threading
from queue import Queue
import tempfile
import os

logger = logging.getLogger(__name__)


@dataclass
class TTSConfig:
    """TTS configuration."""

    model_name: str = "tts_models/en/ljspeech/tacotron2-DDC"
    vocoder_name: Optional[str] = "vocoder_models/en/ljspeech/hifigan_v2"
    use_cuda: bool = False
    sample_rate: int = 22050


class TextToSpeech:
    """Text-to-speech with voice cloning capabilities."""

    def __init__(self, config: Optional[TTSConfig] = None):
        """Initialize TTS engine.

        Args:
            config: TTS configuration
        """
        self.config = config or TTSConfig()
        self.tts = None
        self.is_speaking = False
        self.speech_queue = Queue()
        self.speaking_thread = None
        self.voice_profile = None

        self._initialize_tts()

    def _initialize_tts(self) -> None:
        """Initialize TTS model."""
        try:
            logger.info(f"Initializing TTS model: {self.config.model_name}")

            # Initialize TTS
            self.tts = TTS(
                model_name=self.config.model_name,
                vocoder_name=self.config.vocoder_name,
                use_cuda=self.config.use_cuda,
            )

            logger.info("TTS initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TTS: {e}")
            raise RuntimeError(f"TTS initialization failed: {e}") from e

    def speak(self, text: str, blocking: bool = False) -> bool:
        """Convert text to speech and play.

        Args:
            text: Text to speak
            blocking: Whether to block until speech completes

        Returns:
            True if successful
        """
        if not self.tts:
            logger.error("TTS not initialized")
            return False

        try:
            # Generate audio
            audio = self._generate_audio(text)

            if audio is None:
                return False

            # Play audio
            if blocking:
                self._play_audio(audio)
            else:
                self.speech_queue.put((text, audio))
                self._ensure_speaking_thread()

            return True

        except Exception as e:
            logger.error(f"Failed to speak: {e}")
            return False

    def _generate_audio(self, text: str) -> Optional[np.ndarray]:
        """Generate audio from text.

        Args:
            text: Input text

        Returns:
            Audio array or None
        """
        try:
            # Generate audio with TTS
            if self.voice_profile:
                # Use voice cloning if profile exists
                audio = self.tts.tts_with_vc(
                    text=text,
                    speaker_wav=self.voice_profile,
                )
            else:
                # Use default voice
                audio = self.tts.tts(text=text)

            # Convert to numpy array if needed
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)

            return audio

        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            return None

    def _play_audio(self, audio: np.ndarray) -> None:
        """Play audio through speakers.

        Args:
            audio: Audio data
        """
        try:
            # Ensure audio is in correct format
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Normalize if needed
            max_val = np.max(np.abs(audio))
            if max_val > 1.0:
                audio = audio / max_val

            # Play audio
            sd.play(audio, self.config.sample_rate)
            sd.wait()  # Wait for playback to complete

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
                    text, audio = self.speech_queue.get(timeout=0.5)
                    self._play_audio(audio)
                else:
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"Speaking loop error: {e}")
                time.sleep(0.1)

    def clone_voice(self, audio_samples: list[str], profile_name: str = "user") -> bool:
        """Clone voice from audio samples.

        Args:
            audio_samples: List of audio file paths
            profile_name: Name for voice profile

        Returns:
            True if successful
        """
        if not audio_samples:
            logger.error("No audio samples provided")
            return False

        try:
            logger.info(f"Creating voice profile: {profile_name}")

            # For now, just use the first sample as reference
            # A more sophisticated approach would combine multiple samples
            self.voice_profile = audio_samples[0]

            # You could also save the profile for later use
            # self._save_voice_profile(profile_name)

            logger.info("Voice profile created successfully")
            return True

        except Exception as e:
            logger.error(f"Voice cloning failed: {e}")
            return False

    def save_to_file(self, text: str, output_file: str) -> bool:
        """Save TTS output to audio file.

        Args:
            text: Text to convert
            output_file: Output file path

        Returns:
            True if successful
        """
        try:
            # Generate audio
            audio = self._generate_audio(text)

            if audio is None:
                return False

            # Save to file using TTS library
            self.tts.save_wav(audio, output_file)

            logger.info(f"Audio saved to {output_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            return False

    def stop_speaking(self) -> None:
        """Stop all speech output."""
        self.is_speaking = False

        # Clear queue
        while not self.speech_queue.empty():
            self.speech_queue.get()

        # Stop current playback
        sd.stop()

        if self.speaking_thread:
            self.speaking_thread.join(timeout=5)

        logger.info("Speech stopped")

    def set_voice_parameters(
        self,
        speed: float = 1.0,
        pitch: float = 1.0,
        emotion: Optional[str] = None,
    ) -> None:
        """Adjust voice parameters.

        Args:
            speed: Speech speed multiplier
            pitch: Pitch adjustment
            emotion: Target emotion (if supported by model)
        """
        # These would be model-specific adjustments
        # Store for use during generation
        self.voice_speed = speed
        self.voice_pitch = pitch
        self.voice_emotion = emotion

        logger.info(f"Voice parameters updated: speed={speed}, pitch={pitch}, emotion={emotion}")

    def get_available_voices(self) -> list[str]:
        """Get list of available voices.

        Returns:
            List of voice names
        """
        if not self.tts:
            return []

        try:
            # This would depend on the TTS library implementation
            # For now, return a placeholder list
            return ["default", "cloned"]
        except Exception as e:
            logger.error(f"Failed to get voices: {e}")
            return []

    def estimate_duration(self, text: str) -> float:
        """Estimate speech duration for text.

        Args:
            text: Input text

        Returns:
            Estimated duration in seconds
        """
        # Simple estimation based on average speaking rate
        # ~150 words per minute is typical
        words = len(text.split())
        duration = (words / 150) * 60  # Convert to seconds

        # Adjust for speech speed if set
        if hasattr(self, "voice_speed"):
            duration = duration / self.voice_speed

        return duration