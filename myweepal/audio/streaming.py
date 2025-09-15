"""Real-time streaming ASR module with MLX optimization."""

import asyncio
import logging
import time
from typing import Optional, AsyncIterator, Dict, Any, Callable
from dataclasses import dataclass
import numpy as np
import sounddevice as sd
from queue import Queue
import threading

logger = logging.getLogger(__name__)


@dataclass
class StreamingTranscriptionSegment:
    """Streaming transcription segment with timestamps."""

    text: str
    start_time: float
    end_time: float
    confidence: float
    is_final: bool
    word_timestamps: Optional[list] = None


class StreamingASR:
    """Real-time speech recognition with streaming support."""

    def __init__(
        self,
        model_id: str = "mlx-community/whisper-large-v3-turbo",
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration: float = 2.0,
        overlap_duration: float = 0.5,
        context_size: tuple = (256, 256),  # (left, right) frames for local attention
        energy_threshold: float = 0.01,
        use_mlx: bool = True,
    ):
        """Initialize streaming ASR.

        Args:
            model_id: Model identifier for MLX or Hugging Face
            sample_rate: Audio sample rate (16kHz for most models)
            channels: Number of audio channels (mono recommended)
            chunk_duration: Duration of audio chunks for processing
            overlap_duration: Overlap between chunks for context
            context_size: Local attention context (left, right) frames
            energy_threshold: Minimum energy for voice activity detection
            use_mlx: Whether to use MLX optimization
        """
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.context_size = context_size
        self.energy_threshold = energy_threshold
        self.use_mlx = use_mlx

        self.model = None
        self.processor = None
        self.is_streaming = False
        self.audio_queue = Queue()
        self.result_queue = Queue()

        # Streaming state
        self.audio_buffer = []
        self.previous_text = ""
        self.segment_start_time = None

        # Threading
        self.capture_thread = None
        self.process_thread = None

        self._load_model()

    def _load_model(self) -> None:
        """Load the ASR model."""
        try:
            if self.use_mlx and "mlx" in self.model_id:
                # Try to load MLX-optimized model
                try:
                    import mlx_whisper as whisper
                    logger.info(f"Loading MLX Whisper model: {self.model_id}")
                    self.model = whisper.load(self.model_id.split("/")[-1])
                    logger.info("MLX Whisper model loaded successfully")
                except ImportError:
                    logger.warning("MLX Whisper not available, falling back to standard Whisper")
                    self._load_standard_whisper()
            else:
                self._load_standard_whisper()

        except Exception as e:
            logger.error(f"Failed to load ASR model: {e}")
            # Don't raise error if models aren't available - use placeholder
            self.model = None

    def _load_standard_whisper(self) -> None:
        """Load standard Whisper model as fallback."""
        try:
            import whisper
            model_size = self.model_id.split("-")[-1] if "-" in self.model_id else "base"
            logger.info(f"Loading standard Whisper model: {model_size}")
            self.model = whisper.load_model(model_size)
            logger.info("Standard Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load standard Whisper: {e}")
            # Use placeholder model instead of raising error
            self.model = None

    async def stream_transcribe(
        self,
        audio_stream: AsyncIterator[np.ndarray]
    ) -> AsyncIterator[StreamingTranscriptionSegment]:
        """Process audio stream in real-time.

        Args:
            audio_stream: Async iterator of audio chunks

        Yields:
            Streaming transcription segments
        """
        buffer = []
        buffer_duration = 0.0
        segment_start = time.time()

        async for chunk in audio_stream:
            # Add chunk to buffer
            buffer.append(chunk)
            buffer_duration += len(chunk) / self.sample_rate

            # Process when buffer reaches chunk duration
            if buffer_duration >= self.chunk_duration:
                # Concatenate buffer
                audio_data = np.concatenate(buffer)

                # Transcribe chunk
                segment = await self._transcribe_chunk(
                    audio_data,
                    segment_start,
                    time.time()
                )

                if segment:
                    yield segment

                # Keep overlap for context
                overlap_samples = int(self.overlap_duration * self.sample_rate)
                if len(audio_data) > overlap_samples:
                    buffer = [audio_data[-overlap_samples:]]
                    buffer_duration = self.overlap_duration
                else:
                    buffer = []
                    buffer_duration = 0.0

                segment_start = time.time() - buffer_duration

    async def _transcribe_chunk(
        self,
        audio_data: np.ndarray,
        start_time: float,
        end_time: float
    ) -> Optional[StreamingTranscriptionSegment]:
        """Transcribe a single audio chunk.

        Args:
            audio_data: Audio samples
            start_time: Chunk start time
            end_time: Chunk end time

        Returns:
            Transcription segment or None
        """
        if self.model is None:
            logger.error("Model not loaded")
            return None

        try:
            # Normalize audio
            audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))

            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._transcribe_sync,
                audio_data
            )

            if result and result.get("text"):
                text = result["text"].strip()

                # Filter out repeated text
                if text and text != self.previous_text:
                    self.previous_text = text

                    return StreamingTranscriptionSegment(
                        text=text,
                        start_time=start_time,
                        end_time=end_time,
                        confidence=result.get("avg_logprob", 0.0),
                        is_final=False,  # Can be made final based on VAD
                        word_timestamps=result.get("word_timestamps")
                    )

            return None

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None

    def _transcribe_sync(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Synchronous transcription for thread pool execution.

        Args:
            audio_data: Audio samples

        Returns:
            Transcription result dictionary
        """
        if hasattr(self.model, "transcribe"):
            # Whisper-style API
            return self.model.transcribe(
                audio_data,
                language=None,  # Auto-detect
                task="transcribe",
                fp16=False,
                word_timestamps=True
            )
        else:
            # Handle other model types
            logger.warning("Model does not support transcribe method")
            return {}

    def start_streaming(
        self,
        callback: Optional[Callable[[StreamingTranscriptionSegment], None]] = None
    ) -> None:
        """Start continuous streaming transcription.

        Args:
            callback: Optional callback for transcription segments
        """
        if self.is_streaming:
            logger.warning("Already streaming")
            return

        self.is_streaming = True
        self.callback = callback

        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()

        # Start processing thread
        self.process_thread = threading.Thread(target=self._process_loop)
        self.process_thread.daemon = True
        self.process_thread.start()

        logger.info("Started streaming transcription")

    def stop_streaming(self) -> None:
        """Stop streaming transcription."""
        self.is_streaming = False

        if self.capture_thread:
            self.capture_thread.join(timeout=5)

        if self.process_thread:
            self.process_thread.join(timeout=5)

        # Clear queues
        while not self.audio_queue.empty():
            self.audio_queue.get()

        while not self.result_queue.empty():
            self.result_queue.get()

        logger.info("Stopped streaming transcription")

    def _capture_loop(self) -> None:
        """Audio capture loop for threading."""
        chunk_samples = int(self.chunk_duration * self.sample_rate)

        def audio_callback(indata, frames, time_info, status):
            """Callback for audio stream."""
            if status:
                logger.warning(f"Audio status: {status}")

            # Copy and flatten audio data
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
                while self.is_streaming:
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Audio capture error: {e}")

    def _process_loop(self) -> None:
        """Audio processing loop for threading."""
        buffer = []
        buffer_start_time = None
        buffer_duration = 0.0

        while self.is_streaming or not self.audio_queue.empty():
            try:
                if not self.audio_queue.empty():
                    chunk, timestamp = self.audio_queue.get(timeout=0.5)

                    if buffer_start_time is None:
                        buffer_start_time = timestamp

                    buffer.append(chunk)
                    buffer_duration += len(chunk) / self.sample_rate

                    # Process when buffer is full
                    if buffer_duration >= self.chunk_duration:
                        # Concatenate buffer
                        audio_data = np.concatenate(buffer)

                        # Create async task for transcription
                        segment = self._transcribe_sync_wrapper(
                            audio_data,
                            buffer_start_time,
                            timestamp
                        )

                        if segment:
                            # Add to result queue
                            if not self.result_queue.full():
                                self.result_queue.put(segment)

                            # Call callback if provided
                            if self.callback:
                                self.callback(segment)

                        # Keep overlap for context
                        overlap_samples = int(self.overlap_duration * self.sample_rate)
                        if len(audio_data) > overlap_samples:
                            buffer = [audio_data[-overlap_samples:]]
                            buffer_duration = self.overlap_duration
                            buffer_start_time = timestamp - buffer_duration
                        else:
                            buffer = []
                            buffer_duration = 0.0
                            buffer_start_time = None
                else:
                    time.sleep(0.1)

            except Exception as e:
                logger.error(f"Processing error: {e}")
                time.sleep(0.1)

    def _transcribe_sync_wrapper(
        self,
        audio_data: np.ndarray,
        start_time: float,
        end_time: float
    ) -> Optional[StreamingTranscriptionSegment]:
        """Wrapper for synchronous transcription in thread.

        Args:
            audio_data: Audio samples
            start_time: Start time
            end_time: End time

        Returns:
            Transcription segment or None
        """
        try:
            result = self._transcribe_sync(audio_data)

            if result and result.get("text"):
                text = result["text"].strip()

                if text and text != self.previous_text:
                    self.previous_text = text

                    return StreamingTranscriptionSegment(
                        text=text,
                        start_time=start_time,
                        end_time=end_time,
                        confidence=result.get("avg_logprob", 0.0),
                        is_final=True,
                        word_timestamps=result.get("word_timestamps")
                    )

            return None

        except Exception as e:
            logger.error(f"Transcription wrapper error: {e}")
            return None

    def get_results(self, limit: int = 10) -> list[StreamingTranscriptionSegment]:
        """Get recent transcription results.

        Args:
            limit: Maximum number of results

        Returns:
            List of transcription segments
        """
        results = []
        while not self.result_queue.empty() and len(results) < limit:
            try:
                results.append(self.result_queue.get_nowait())
            except:
                break
        return results