"""Speaker recognition and identification module with voice biometrics."""

import logging
import time
import hashlib
import json
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
import librosa
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import advanced libraries, provide fallbacks
try:
    import torch
    import torchaudio
    from speechbrain.pretrained import EncoderClassifier, SpeakerRecognition
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    logger.warning("SpeechBrain not available. Using fallback speaker recognition.")

try:
    from pyannote.audio import Pipeline as PyannotePipeline
    from pyannote.audio import Model as PyannoteModel
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    logger.warning("Pyannote not available. Speaker diarization disabled.")


class UserRole(Enum):
    """User role enumeration."""
    PRIME = "prime"  # Primary user (David)
    FAMILY = "family"  # Family members
    GUEST = "guest"  # Temporary guests
    UNKNOWN = "unknown"  # Unidentified speakers


@dataclass
class VoiceProfile:
    """Voice profile for a registered user."""
    user_id: str
    name: str
    role: UserRole
    embeddings: List[np.ndarray] = field(default_factory=list)
    enrollment_samples: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_seen: Optional[datetime] = None
    verification_threshold: float = 0.85
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpeakerSegment:
    """Identified speaker segment."""
    speaker_id: str
    name: str
    role: UserRole
    start_time: float
    end_time: float
    confidence: float
    text: Optional[str] = None


class SpeakerRecognitionSystem:
    """Speaker recognition system with enrollment and identification."""

    def __init__(
        self,
        model_name: str = "speechbrain/spkrec-ecapa-voxceleb",
        sample_rate: int = 16000,
        embedding_size: int = 192,
        min_enrollment_duration: float = 10.0,
        max_enrollment_duration: float = 30.0,
        identification_threshold: float = 0.85,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize speaker recognition system.

        Args:
            model_name: SpeechBrain model for embeddings
            sample_rate: Audio sample rate
            embedding_size: Size of speaker embeddings
            min_enrollment_duration: Minimum enrollment audio duration
            max_enrollment_duration: Maximum enrollment audio duration
            identification_threshold: Confidence threshold for identification
            cache_dir: Directory for caching profiles
        """
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.embedding_size = embedding_size
        self.min_enrollment_duration = min_enrollment_duration
        self.max_enrollment_duration = max_enrollment_duration
        self.identification_threshold = identification_threshold
        self.cache_dir = cache_dir or Path.home() / ".myweepal" / "speaker_profiles"

        # Initialize models
        self.embedding_model = None
        self.verification_model = None
        self.diarization_pipeline = None

        # Speaker profiles database
        self.profiles: Dict[str, VoiceProfile] = {}

        # Real-time tracking
        self.active_speakers: Dict[str, float] = {}  # speaker_id -> last_seen
        self.speaker_history: List[SpeakerSegment] = []

        self._initialize_models()
        self._load_profiles()

    def _initialize_models(self) -> None:
        """Initialize speaker recognition models."""
        if SPEECHBRAIN_AVAILABLE:
            try:
                logger.info(f"Loading SpeechBrain model: {self.model_name}")
                self.embedding_model = EncoderClassifier.from_hparams(
                    source=self.model_name,
                    savedir=self.cache_dir / "models",
                    run_opts={"device": "cpu"}
                )

                # Load verification model for similarity scoring
                self.verification_model = SpeakerRecognition.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=self.cache_dir / "verification",
                    run_opts={"device": "cpu"}
                )

                logger.info("SpeechBrain models loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load SpeechBrain models: {e}")
                self.embedding_model = None
                self.verification_model = None

        if PYANNOTE_AVAILABLE:
            try:
                logger.info("Loading Pyannote diarization pipeline")
                # Note: Requires auth token for some models
                # self.diarization_pipeline = PyannotePipeline.from_pretrained(
                #     "pyannote/speaker-diarization"
                # )
                logger.info("Pyannote pipeline ready")
            except Exception as e:
                logger.warning(f"Failed to load Pyannote: {e}")

    def _load_profiles(self) -> None:
        """Load saved speaker profiles from disk."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        profiles_file = self.cache_dir / "profiles.json"

        if profiles_file.exists():
            try:
                with open(profiles_file, "r") as f:
                    data = json.load(f)

                for user_id, profile_data in data.items():
                    # Convert embeddings from lists to numpy arrays
                    embeddings = [np.array(emb) for emb in profile_data.get("embeddings", [])]

                    profile = VoiceProfile(
                        user_id=user_id,
                        name=profile_data["name"],
                        role=UserRole(profile_data["role"]),
                        embeddings=embeddings,
                        enrollment_samples=profile_data.get("enrollment_samples", 0),
                        created_at=datetime.fromisoformat(profile_data["created_at"]),
                        last_seen=datetime.fromisoformat(profile_data["last_seen"]) if profile_data.get("last_seen") else None,
                        verification_threshold=profile_data.get("verification_threshold", 0.85),
                        metadata=profile_data.get("metadata", {})
                    )
                    self.profiles[user_id] = profile

                logger.info(f"Loaded {len(self.profiles)} speaker profiles")
            except Exception as e:
                logger.error(f"Failed to load profiles: {e}")

    def _save_profiles(self) -> None:
        """Save speaker profiles to disk."""
        profiles_file = self.cache_dir / "profiles.json"

        try:
            data = {}
            for user_id, profile in self.profiles.items():
                data[user_id] = {
                    "name": profile.name,
                    "role": profile.role.value,
                    "embeddings": [emb.tolist() for emb in profile.embeddings],
                    "enrollment_samples": profile.enrollment_samples,
                    "created_at": profile.created_at.isoformat(),
                    "last_seen": profile.last_seen.isoformat() if profile.last_seen else None,
                    "verification_threshold": profile.verification_threshold,
                    "metadata": profile.metadata
                }

            with open(profiles_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self.profiles)} speaker profiles")
        except Exception as e:
            logger.error(f"Failed to save profiles: {e}")

    def enroll_speaker(
        self,
        audio_path: str,
        name: str,
        role: UserRole = UserRole.FAMILY,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """Enroll a new speaker with voice samples.

        Args:
            audio_path: Path to enrollment audio
            name: Speaker's name
            role: User role
            metadata: Additional metadata

        Returns:
            Success status and user_id or error message
        """
        try:
            # Load and validate audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            duration = len(audio) / sr

            if duration < self.min_enrollment_duration:
                return False, f"Audio too short ({duration:.1f}s), need at least {self.min_enrollment_duration}s"

            if duration > self.max_enrollment_duration:
                # Trim to max duration
                audio = audio[:int(self.max_enrollment_duration * sr)]

            # Generate embeddings
            embeddings = self._extract_embeddings(audio)

            if not embeddings:
                return False, "Failed to extract voice embeddings"

            # Generate user ID
            user_id = self._generate_user_id(name)

            # Create or update profile
            if user_id in self.profiles:
                profile = self.profiles[user_id]
                profile.embeddings.extend(embeddings)
                profile.enrollment_samples += len(embeddings)
                logger.info(f"Updated profile for {name} with {len(embeddings)} new embeddings")
            else:
                profile = VoiceProfile(
                    user_id=user_id,
                    name=name,
                    role=role,
                    embeddings=embeddings,
                    enrollment_samples=len(embeddings),
                    metadata=metadata or {}
                )
                self.profiles[user_id] = profile
                logger.info(f"Created new profile for {name} with {len(embeddings)} embeddings")

            # Save profiles
            self._save_profiles()

            return True, user_id

        except Exception as e:
            logger.error(f"Enrollment failed: {e}")
            return False, str(e)

    def identify_speaker(
        self,
        audio: np.ndarray,
        return_all_scores: bool = False
    ) -> Tuple[Optional[str], float, Optional[Dict[str, float]]]:
        """Identify speaker from audio.

        Args:
            audio: Audio samples
            return_all_scores: Return scores for all speakers

        Returns:
            Identified user_id, confidence, and optionally all scores
        """
        if not self.profiles:
            logger.warning("No enrolled speakers")
            return None, 0.0, None

        try:
            # Extract embeddings from input audio
            embeddings = self._extract_embeddings(audio)

            if not embeddings:
                return None, 0.0, None

            # Compare with all enrolled speakers
            scores = {}
            for user_id, profile in self.profiles.items():
                if not profile.embeddings:
                    continue

                # Calculate similarity with each enrolled embedding
                similarities = []
                for test_emb in embeddings:
                    for ref_emb in profile.embeddings[:5]:  # Use top 5 reference embeddings
                        sim = self._calculate_similarity(test_emb, ref_emb)
                        similarities.append(sim)

                # Average similarity as score
                scores[user_id] = np.mean(similarities) if similarities else 0.0

            if not scores:
                return None, 0.0, None

            # Find best match
            best_user_id = max(scores, key=scores.get)
            best_score = scores[best_user_id]

            # Check threshold
            if best_score >= self.identification_threshold:
                # Update last seen
                self.profiles[best_user_id].last_seen = datetime.now()
                self.active_speakers[best_user_id] = time.time()

                if return_all_scores:
                    return best_user_id, best_score, scores
                return best_user_id, best_score, None

            return None, best_score, scores if return_all_scores else None

        except Exception as e:
            logger.error(f"Speaker identification failed: {e}")
            return None, 0.0, None

    def _extract_embeddings(self, audio: np.ndarray) -> List[np.ndarray]:
        """Extract speaker embeddings from audio.

        Args:
            audio: Audio samples

        Returns:
            List of embedding vectors
        """
        if self.embedding_model is None:
            # Fallback: Use MFCC features as simple embeddings
            return self._extract_mfcc_embeddings(audio)

        try:
            # Convert to tensor
            audio_tensor = torch.tensor(audio).unsqueeze(0)

            # Extract embeddings using SpeechBrain
            with torch.no_grad():
                embeddings = self.embedding_model.encode_batch(audio_tensor)

            return [embeddings.squeeze().numpy()]

        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return self._extract_mfcc_embeddings(audio)

    def _extract_mfcc_embeddings(self, audio: np.ndarray) -> List[np.ndarray]:
        """Extract MFCC-based embeddings as fallback.

        Args:
            audio: Audio samples

        Returns:
            List of MFCC-based embeddings
        """
        try:
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=20,
                n_fft=2048,
                hop_length=512
            )

            # Create embeddings from MFCC statistics
            embeddings = []

            # Sliding window for multiple embeddings
            window_size = int(3.0 * self.sample_rate)  # 3-second windows
            hop_size = int(1.5 * self.sample_rate)  # 1.5-second hop

            for i in range(0, len(audio) - window_size, hop_size):
                window = audio[i:i + window_size]
                window_mfcc = librosa.feature.mfcc(
                    y=window,
                    sr=self.sample_rate,
                    n_mfcc=20
                )

                # Compute statistics as embedding
                embedding = np.concatenate([
                    np.mean(window_mfcc, axis=1),
                    np.std(window_mfcc, axis=1),
                    np.min(window_mfcc, axis=1),
                    np.max(window_mfcc, axis=1)
                ])

                # Pad or truncate to fixed size
                if len(embedding) < self.embedding_size:
                    embedding = np.pad(embedding, (0, self.embedding_size - len(embedding)))
                else:
                    embedding = embedding[:self.embedding_size]

                embeddings.append(embedding)

            return embeddings if embeddings else [np.zeros(self.embedding_size)]

        except Exception as e:
            logger.error(f"MFCC embedding extraction failed: {e}")
            return [np.zeros(self.embedding_size)]

    def _calculate_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate similarity between two embeddings.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Similarity score (0-1)
        """
        if self.verification_model is not None:
            try:
                # Use SpeechBrain verification model
                score = self.verification_model.verify_batch(
                    torch.tensor(emb1).unsqueeze(0),
                    torch.tensor(emb2).unsqueeze(0)
                )
                return float(score)
            except:
                pass

        # Fallback: Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        # Convert from [-1, 1] to [0, 1]
        return (similarity + 1) / 2

    def _generate_user_id(self, name: str) -> str:
        """Generate unique user ID from name.

        Args:
            name: User's name

        Returns:
            Unique user ID
        """
        base_id = name.lower().replace(" ", "_")
        timestamp = str(int(time.time()))
        unique_string = f"{base_id}_{timestamp}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:12]

    def remove_speaker(self, user_id: str) -> bool:
        """Remove a speaker profile.

        Args:
            user_id: User ID to remove

        Returns:
            Success status
        """
        if user_id in self.profiles:
            del self.profiles[user_id]
            self._save_profiles()
            logger.info(f"Removed speaker profile: {user_id}")
            return True
        return False

    def get_active_speakers(self, timeout: float = 30.0) -> List[str]:
        """Get list of recently active speakers.

        Args:
            timeout: Seconds before considering speaker inactive

        Returns:
            List of active user IDs
        """
        current_time = time.time()
        active = []

        for user_id, last_seen in self.active_speakers.items():
            if current_time - last_seen < timeout:
                active.append(user_id)

        return active

    def get_speaker_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get speaker information.

        Args:
            user_id: User ID

        Returns:
            Speaker information dictionary
        """
        if user_id not in self.profiles:
            return None

        profile = self.profiles[user_id]
        return {
            "user_id": user_id,
            "name": profile.name,
            "role": profile.role.value,
            "enrollment_samples": profile.enrollment_samples,
            "created_at": profile.created_at.isoformat(),
            "last_seen": profile.last_seen.isoformat() if profile.last_seen else None,
            "metadata": profile.metadata
        }

    def authenticate_voice(
        self,
        audio: np.ndarray,
        claimed_user_id: str,
        threshold: Optional[float] = None
    ) -> Tuple[bool, float]:
        """Authenticate user by voice.

        Args:
            audio: Audio samples
            claimed_user_id: Claimed user identity
            threshold: Optional custom threshold

        Returns:
            Authentication result and confidence score
        """
        if claimed_user_id not in self.profiles:
            return False, 0.0

        profile = self.profiles[claimed_user_id]
        threshold = threshold or profile.verification_threshold

        # Identify speaker
        identified_id, confidence, _ = self.identify_speaker(audio)

        # Check if identified speaker matches claimed identity
        if identified_id == claimed_user_id and confidence >= threshold:
            return True, confidence

        return False, confidence

    def process_conversation(
        self,
        audio_segments: List[Tuple[np.ndarray, float, float]]
    ) -> List[SpeakerSegment]:
        """Process conversation with multiple speakers.

        Args:
            audio_segments: List of (audio, start_time, end_time) tuples

        Returns:
            List of identified speaker segments
        """
        segments = []

        for audio, start_time, end_time in audio_segments:
            user_id, confidence, _ = self.identify_speaker(audio)

            if user_id:
                profile = self.profiles[user_id]
                segment = SpeakerSegment(
                    speaker_id=user_id,
                    name=profile.name,
                    role=profile.role,
                    start_time=start_time,
                    end_time=end_time,
                    confidence=confidence
                )
                segments.append(segment)
                self.speaker_history.append(segment)

        return segments