"""Speech recognition module using Whisper."""

import logging
import time
from typing import Optional, Callable, Any
from dataclasses import dataclass
import numpy as np
import sounddevice as sd
import whisper
import threading
from queue import Queue
import wave
import tempfile
import os

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Speech transcription result."""

    text: str
    confidence: float
    timestamp: float
    duration: float
    language: str


class SpeechRecognizer:
    """Speech recognition using Whisper."""

    def __init__(
        self,
        model_size: str = "base",
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration: float = 5.0,
        energy_threshold: float = 0.01,
    ):
        """Initialize speech recognizer.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            sample_rate: Audio sample rate
            channels: Number of audio channels
            chunk_duration: Duration of audio chunks for processing
            energy_threshold: Minimum energy for voice activity detection
        """
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.energy_threshold = energy_threshold

        self.model = None
        self.is_listening = False
        self.audio_queue = Queue()
        self.transcription_queue = Queue(maxsize=100)
        self.listening_thread = None
        self.processing_thread = None
        self.callback = None

        self._load_model()

    def _load_model(self) -> None:
        """Load Whisper model."""
        try:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def start_listening(self, callback: Optional[Callable[[TranscriptionResult], Any]] = None) -> None:
        """Start continuous listening.

        Args:
            callback: Optional callback for transcription results
        """
        if self.is_listening:
            logger.warning("Already listening")
            return

        self.callback = callback
        self.is_listening = True

        # Start audio capture thread
        self.listening_thread = threading.Thread(target=self._audio_capture_loop)
        self.listening_thread.daemon = True
        self.listening_thread.start()

        # Start processing thread
        self.processing_thread = threading.Thread(target=self._audio_processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        logger.info("Started listening")

    def stop_listening(self) -> None:
        """Stop continuous listening."""
        self.is_listening = False

        if self.listening_thread:
            self.listening_thread.join(timeout=5)

        if self.processing_thread:
            self.processing_thread.join(timeout=5)

        # Clear queues
        while not self.audio_queue.empty():
            self.audio_queue.get()

        logger.info("Stopped listening")

    def _audio_capture_loop(self) -> None:
        """Capture audio in separate thread."""
        chunk_samples = int(self.chunk_duration * self.sample_rate)

        def audio_callback(indata, frames, time_info, status):
            """Callback for audio stream."""
            if status:
                logger.warning(f"Audio status: {status}")

            # Copy audio data
            audio_chunk = indata.copy().flatten()

            # Simple voice activity detection
            energy = np.sqrt(np.mean(audio_chunk**2))
            if energy > self.energy_threshold:
                self.audio_queue.put((audio_chunk, time.time()))

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=audio_callback,
                blocksize=chunk_samples,
            ):
                while self.is_listening:
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Audio capture error: {e}")

    def _audio_processing_loop(self) -> None:
        """Process audio chunks in separate thread."""
        audio_buffer = []
        buffer_start_time = None

        while self.is_listening or not self.audio_queue.empty():
            try:
                # Get audio chunk
                if not self.audio_queue.empty():
                    chunk, timestamp = self.audio_queue.get(timeout=0.5)

                    if buffer_start_time is None:
                        buffer_start_time = timestamp

                    audio_buffer.append(chunk)

                    # Process when buffer is full
                    buffer_duration = len(audio_buffer) * self.chunk_duration
                    if buffer_duration >= self.chunk_duration:
                        # Concatenate audio chunks
                        audio_data = np.concatenate(audio_buffer)

                        # Transcribe
                        result = self._transcribe_audio(audio_data, buffer_start_time)

                        if result and result.text.strip():
                            # Add to queue
                            if not self.transcription_queue.full():
                                self.transcription_queue.put(result)

                            # Call callback if provided
                            if self.callback:
                                self.callback(result)

                        # Clear buffer
                        audio_buffer = []
                        buffer_start_time = None

                else:
                    time.sleep(0.1)

            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                time.sleep(0.1)

    def _transcribe_audio(
        self,
        audio_data: np.ndarray,
        timestamp: float,
    ) -> Optional[TranscriptionResult]:
        """Transcribe audio data.

        Args:
            audio_data: Audio samples
            timestamp: Start timestamp

        Returns:
            Transcription result or None
        """
        if not self.model:
            logger.error("Model not loaded")
            return None

        try:
            # Normalize audio
            audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))

            # Transcribe
            result = self.model.transcribe(
                audio_data,
                language=None,  # Auto-detect
                task="transcribe",
                fp16=False,
            )

            if result and "text" in result:
                return TranscriptionResult(
                    text=result["text"],
                    confidence=result.get("avg_logprob", 0.0),
                    timestamp=timestamp,
                    duration=len(audio_data) / self.sample_rate,
                    language=result.get("language", "en"),
                )

            return None

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None

    def transcribe_file(self, audio_file: str) -> Optional[TranscriptionResult]:
        """Transcribe an audio file.

        Args:
            audio_file: Path to audio file

        Returns:
            Transcription result or None
        """
        if not self.model:
            logger.error("Model not loaded")
            return None

        try:
            result = self.model.transcribe(audio_file)

            if result and "text" in result:
                return TranscriptionResult(
                    text=result["text"],
                    confidence=result.get("avg_logprob", 0.0),
                    timestamp=time.time(),
                    duration=0.0,  # Would need to read file to get duration
                    language=result.get("language", "en"),
                )

            return None

        except Exception as e:
            logger.error(f"File transcription error: {e}")
            return None

    def record_and_transcribe(self, duration: float = 5.0) -> Optional[TranscriptionResult]:
        """Record audio for specified duration and transcribe.

        Args:
            duration: Recording duration in seconds

        Returns:
            Transcription result or None
        """
        try:
            logger.info(f"Recording for {duration} seconds...")

            # Record audio
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
            )
            sd.wait()  # Wait for recording to complete

            logger.info("Recording complete, transcribing...")

            # Transcribe
            return self._transcribe_audio(audio_data.flatten(), time.time())

        except Exception as e:
            logger.error(f"Record and transcribe error: {e}")
            return None

    def get_transcription_history(self, limit: int = 10) -> list[TranscriptionResult]:
        """Get recent transcriptions.

        Args:
            limit: Maximum number of results

        Returns:
            List of transcription results
        """
        history = []
        while not self.transcription_queue.empty() and len(history) < limit:
            try:
                history.append(self.transcription_queue.get_nowait())
            except:
                break
        return history

    def save_audio(self, audio_data: np.ndarray, filename: str) -> bool:
        """Save audio data to file.

        Args:
            audio_data: Audio samples
            filename: Output filename

        Returns:
            True if saved successfully
        """
        try:
            # Normalize and convert to int16
            audio_data = audio_data * 32767
            audio_data = audio_data.astype(np.int16)

            # Save as WAV
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data.tobytes())

            logger.info(f"Audio saved to {filename}")
            return True

        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            return False