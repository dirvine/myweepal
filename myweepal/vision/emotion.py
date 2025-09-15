"""Emotion detection module using webcam."""

import logging
import time
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import cv2
import numpy as np
import threading
from queue import Queue

# Try to import FER, but provide fallback if not available
try:
    from fer import FER
    FER_AVAILABLE = True
except ImportError:
    FER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("FER library not available. Emotion detection will use fallback mode.")

logger = logging.getLogger(__name__)


@dataclass
class EmotionState:
    """Represents detected emotional state."""

    dominant_emotion: str
    emotion_scores: Dict[str, float]
    timestamp: float
    face_detected: bool
    confidence: float


class EmotionDetector:
    """Real-time emotion detection from webcam."""

    def __init__(
        self,
        camera_index: int = 0,
        detection_interval: float = 5.0,
        enable_display: bool = False,
    ):
        """Initialize emotion detector.

        Args:
            camera_index: Camera device index
            detection_interval: Seconds between emotion analyses
            enable_display: Whether to show video feed
        """
        self.camera_index = camera_index
        self.detection_interval = detection_interval
        self.enable_display = enable_display

        self.detector = FER(mtcnn=True) if FER_AVAILABLE else None
        self.capture = None
        self.is_running = False
        self.emotion_queue = Queue(maxsize=10)
        self.detection_thread = None
        self.current_emotion = None

    def start(self) -> None:
        """Start emotion detection."""
        if self.is_running:
            logger.warning("Emotion detector already running")
            return

        try:
            self.capture = cv2.VideoCapture(self.camera_index)
            if not self.capture.isOpened():
                raise RuntimeError(f"Failed to open camera {self.camera_index}")

            self.is_running = True
            self.detection_thread = threading.Thread(target=self._detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()

            logger.info("Emotion detector started")
        except Exception as e:
            logger.error(f"Failed to start emotion detector: {e}")
            raise RuntimeError(f"Emotion detector startup failed: {e}") from e

    def stop(self) -> None:
        """Stop emotion detection."""
        self.is_running = False

        if self.detection_thread:
            self.detection_thread.join(timeout=5)

        if self.capture:
            self.capture.release()

        if self.enable_display:
            cv2.destroyAllWindows()

        logger.info("Emotion detector stopped")

    def _detection_loop(self) -> None:
        """Main detection loop running in separate thread."""
        last_detection_time = 0

        while self.is_running:
            try:
                ret, frame = self.capture.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    time.sleep(0.1)
                    continue

                current_time = time.time()

                # Perform emotion detection at specified interval
                if current_time - last_detection_time >= self.detection_interval:
                    emotion_state = self._analyze_frame(frame, current_time)

                    if emotion_state:
                        self.current_emotion = emotion_state

                        # Add to queue if not full
                        if not self.emotion_queue.full():
                            self.emotion_queue.put(emotion_state)

                    last_detection_time = current_time

                # Display frame if enabled
                if self.enable_display:
                    self._display_frame(frame)

                # Small delay to prevent CPU overload
                time.sleep(0.03)  # ~30 FPS

            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                time.sleep(0.1)

    def _analyze_frame(self, frame: np.ndarray, timestamp: float) -> Optional[EmotionState]:
        """Analyze frame for emotions.

        Args:
            frame: Video frame
            timestamp: Current timestamp

        Returns:
            Detected emotion state or None
        """
        try:
            # Check if detector is available
            if not self.detector:
                # Fallback: return neutral emotion
                return EmotionState(
                    dominant_emotion="neutral",
                    emotion_scores={"neutral": 1.0},
                    timestamp=timestamp,
                    face_detected=False,
                    confidence=0.5,
                )

            # Detect emotions
            result = self.detector.detect_emotions(frame)

            if not result:
                return EmotionState(
                    dominant_emotion="neutral",
                    emotion_scores={},
                    timestamp=timestamp,
                    face_detected=False,
                    confidence=0.0,
                )

            # Get the first face (assuming single user)
            face_result = result[0]
            emotions = face_result.get("emotions", {})

            if not emotions:
                return None

            # Find dominant emotion
            dominant_emotion = max(emotions, key=emotions.get)
            confidence = emotions[dominant_emotion]

            return EmotionState(
                dominant_emotion=dominant_emotion,
                emotion_scores=emotions,
                timestamp=timestamp,
                face_detected=True,
                confidence=confidence,
            )

        except Exception as e:
            logger.error(f"Failed to analyze frame: {e}")
            return None

    def _display_frame(self, frame: np.ndarray) -> None:
        """Display frame with emotion overlay.

        Args:
            frame: Video frame
        """
        try:
            if self.current_emotion and self.current_emotion.face_detected:
                # Add emotion text to frame
                text = f"Emotion: {self.current_emotion.dominant_emotion}"
                confidence_text = f"Confidence: {self.current_emotion.confidence:.2f}"

                cv2.putText(
                    frame,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    confidence_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("MyWeePal - Emotion Detection", frame)

            # Check for 'q' key to quit display
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.enable_display = False

        except Exception as e:
            logger.error(f"Failed to display frame: {e}")

    def get_current_emotion(self) -> Optional[EmotionState]:
        """Get the most recent emotion state.

        Returns:
            Current emotion state or None
        """
        return self.current_emotion

    def get_emotion_history(self, limit: int = 10) -> list[EmotionState]:
        """Get recent emotion history.

        Args:
            limit: Maximum number of states to return

        Returns:
            List of recent emotion states
        """
        history = []
        while not self.emotion_queue.empty() and len(history) < limit:
            try:
                history.append(self.emotion_queue.get_nowait())
            except:
                break
        return history

    def get_emotion_summary(self) -> Dict[str, Any]:
        """Get summary of emotion detection.

        Returns:
            Summary statistics
        """
        history = self.get_emotion_history(100)

        if not history:
            return {
                "total_detections": 0,
                "face_detection_rate": 0.0,
                "dominant_emotions": {},
                "average_confidence": 0.0,
            }

        face_detections = sum(1 for e in history if e.face_detected)
        emotion_counts = {}

        total_confidence = 0.0
        confidence_count = 0

        for emotion_state in history:
            if emotion_state.face_detected:
                emotion = emotion_state.dominant_emotion
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                total_confidence += emotion_state.confidence
                confidence_count += 1

        return {
            "total_detections": len(history),
            "face_detection_rate": face_detections / len(history) if history else 0.0,
            "dominant_emotions": emotion_counts,
            "average_confidence": total_confidence / confidence_count if confidence_count > 0 else 0.0,
        }

    def calibrate(self, duration: float = 10.0) -> Dict[str, float]:
        """Calibrate emotion baseline.

        Args:
            duration: Calibration duration in seconds

        Returns:
            Baseline emotion scores
        """
        logger.info(f"Starting emotion calibration for {duration} seconds")

        if not self.is_running:
            self.start()

        start_time = time.time()
        emotion_accumulator = {}
        sample_count = 0

        while time.time() - start_time < duration:
            emotion = self.get_current_emotion()

            if emotion and emotion.face_detected:
                for emo, score in emotion.emotion_scores.items():
                    if emo not in emotion_accumulator:
                        emotion_accumulator[emo] = 0.0
                    emotion_accumulator[emo] += score
                sample_count += 1

            time.sleep(0.5)

        if sample_count > 0:
            baseline = {emo: score / sample_count for emo, score in emotion_accumulator.items()}
        else:
            baseline = {}

        logger.info(f"Calibration complete. Baseline: {baseline}")
        return baseline