"""Dialogue management system integrating all components."""

import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from pathlib import Path

from myweepal.core.llm import LLMInference, LLMConfig
from myweepal.core.memory import MemoryStore, Memory
from myweepal.core.personalized_memory import PersonalizedMemorySystem, PrivacyLevel
from myweepal.vision.emotion import EmotionDetector, EmotionState
from myweepal.audio.asr import SpeechRecognizer, TranscriptionResult
from myweepal.audio.tts import TextToSpeech, TTSConfig
from myweepal.audio.speaker_recognition import SpeakerRecognitionSystem, UserRole

logger = logging.getLogger(__name__)


class ConversationPhase(Enum):
    """Phases of life story conversation."""

    INTRODUCTION = "introduction"
    CHILDHOOD = "childhood"
    EDUCATION = "education"
    CAREER = "career"
    RELATIONSHIPS = "relationships"
    ACHIEVEMENTS = "achievements"
    CHALLENGES = "challenges"
    ASPIRATIONS = "aspirations"
    REFLECTION = "reflection"
    COMPLETE = "complete"


@dataclass
class ConversationState:
    """Current state of the conversation."""

    phase: ConversationPhase
    questions_asked: int
    current_topic: str
    user_mood: str
    engagement_level: float
    session_start: float
    last_interaction: float
    current_speaker_id: Optional[str] = None
    current_speaker_name: Optional[str] = None


class DialogueManager:
    """Manages the conversational flow and integrates all components."""

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        tts_config: Optional[TTSConfig] = None,
        memory_persist_dir: str = "./data/chroma",
        question_bank_path: Optional[str] = None,
        enable_speaker_recognition: bool = True,
    ):
        """Initialize dialogue manager.

        Args:
            llm_config: LLM configuration
            tts_config: TTS configuration
            memory_persist_dir: Directory for memory storage
            question_bank_path: Path to question bank JSON
            enable_speaker_recognition: Enable speaker recognition
        """
        # Initialize components
        self.llm = LLMInference(llm_config)
        self.memory_store = MemoryStore(memory_persist_dir)  # Legacy memory store
        self.emotion_detector = EmotionDetector()
        self.speech_recognizer = SpeechRecognizer()
        self.tts = TextToSpeech(tts_config)

        # Initialize new components
        self.enable_speaker_recognition = enable_speaker_recognition
        if enable_speaker_recognition:
            self.speaker_system = SpeakerRecognitionSystem()
            self.personalized_memory = PersonalizedMemorySystem(
                db_path=Path(memory_persist_dir).parent / "personalized_memory"
            )
        else:
            self.speaker_system = None
            self.personalized_memory = None

        # Conversation state
        self.state = ConversationState(
            phase=ConversationPhase.INTRODUCTION,
            questions_asked=0,
            current_topic="introduction",
            user_mood="neutral",
            engagement_level=1.0,
            session_start=time.time(),
            last_interaction=time.time(),
        )

        # Question bank
        self.question_bank = self._load_question_bank(question_bank_path)
        self.asked_questions = set()

        # Session data
        self.session_memories = []
        self.is_active = False

    def _load_question_bank(self, path: Optional[str]) -> Dict[str, List[str]]:
        """Load question bank from file or use defaults.

        Args:
            path: Path to question bank JSON

        Returns:
            Question bank organized by phase
        """
        if path and Path(path).exists():
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load question bank: {e}")

        # Default question bank
        return {
            "introduction": [
                "Hello! I'm here to help capture your life story. What's your name?",
                "It's wonderful to meet you. How are you feeling today?",
                "Before we begin, is there anything specific about your life you'd like to share?",
            ],
            "childhood": [
                "Where were you born and raised?",
                "What are your earliest childhood memories?",
                "Can you tell me about your family growing up?",
                "What was your favorite thing to do as a child?",
                "Did you have any childhood dreams or aspirations?",
            ],
            "education": [
                "What was school like for you?",
                "Who was your favorite teacher and why?",
                "What subjects did you enjoy most?",
                "Did you pursue higher education? Tell me about that experience.",
                "What's the most important lesson you learned during your education?",
            ],
            "career": [
                "What was your first job?",
                "How did you choose your career path?",
                "What has been your most rewarding professional experience?",
                "Have you had any mentors who influenced your career?",
                "What professional achievement are you most proud of?",
            ],
            "relationships": [
                "Tell me about the important people in your life.",
                "How did you meet your closest friends?",
                "What relationships have shaped who you are today?",
                "Is there someone who had a profound impact on your life?",
                "What does family mean to you?",
            ],
            "achievements": [
                "What accomplishment are you most proud of?",
                "Have you achieved any long-term goals you set for yourself?",
                "What personal milestones stand out in your memory?",
                "Have you made a difference in someone else's life?",
                "What legacy would you like to leave?",
            ],
            "challenges": [
                "What has been your biggest challenge in life?",
                "How have you overcome difficult times?",
                "What life lessons have you learned from adversity?",
                "Is there anything you would do differently?",
                "How have challenges shaped your character?",
            ],
            "aspirations": [
                "What are your current goals and dreams?",
                "Is there something you still want to accomplish?",
                "What gives your life meaning and purpose?",
                "How do you want to be remembered?",
                "What advice would you give to your younger self?",
            ],
            "reflection": [
                "Looking back, what are you most grateful for?",
                "What has brought you the most joy in life?",
                "If you could relive one moment, what would it be?",
                "What wisdom would you like to pass on?",
                "How would you summarize your life story?",
            ],
        }

    def start_session(self) -> None:
        """Start a new conversation session."""
        logger.info("Starting dialogue session")

        self.is_active = True
        self.state.session_start = time.time()
        self.state.last_interaction = time.time()

        # Start component services
        self.emotion_detector.start()
        self.speech_recognizer.start_listening(self._handle_transcription)

        # Initial greeting
        self._speak_and_record("Hello! I'm MyWeePal, and I'm here to help capture your life story. This is a safe space where you can share your experiences, memories, and wisdom. Shall we begin?")

    def stop_session(self) -> None:
        """Stop the current session."""
        logger.info("Stopping dialogue session")

        self.is_active = False

        # Stop component services
        self.emotion_detector.stop()
        self.speech_recognizer.stop_listening()
        self.tts.stop_speaking()

        # Save session summary
        self._save_session_summary()

    def _handle_transcription(self, transcription: TranscriptionResult) -> None:
        """Handle incoming transcription.

        Args:
            transcription: Speech transcription result
        """
        if not self.is_active:
            return

        logger.info(f"User said: {transcription.text}")

        # Update interaction time
        self.state.last_interaction = time.time()

        # Get current emotion
        emotion = self.emotion_detector.get_current_emotion()
        if emotion:
            self.state.user_mood = emotion.dominant_emotion

        # Process response
        self._process_user_response(transcription.text, emotion)

    def _process_user_response(self, text: str, emotion: Optional[EmotionState]) -> None:
        """Process user's response and generate follow-up.

        Args:
            text: User's response text
            emotion: Current emotion state
        """
        # Generate embedding
        embedding = self._generate_text_embedding(text)

        # Store memory
        current_question = self._get_current_question()
        memory_id = self.memory_store.add_memory(
            question=current_question,
            answer=text,
            embedding=embedding,
            category=self.state.phase.value,
            emotion=emotion.dominant_emotion if emotion else "neutral",
            metadata={
                "session_time": time.time() - self.state.session_start,
                "engagement": self.state.engagement_level,
            },
        )

        self.session_memories.append(memory_id)

        # Generate and speak response
        response = self._generate_response(text, emotion)
        self._speak_and_record(response)

        # Decide next action
        self._update_conversation_state()
        next_question = self._select_next_question()

        if next_question:
            time.sleep(2)  # Natural pause
            self._speak_and_record(next_question)

    def _generate_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text.

        Args:
            text: Input text

        Returns:
            Text embedding
        """
        try:
            embedding = self.llm.generate_embedding(text)
            return np.array(embedding)
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return random embedding as fallback
            return np.random.randn(768)

    def _generate_response(self, user_text: str, emotion: Optional[EmotionState]) -> str:
        """Generate contextual response.

        Args:
            user_text: User's input
            emotion: Current emotion

        Returns:
            Generated response
        """
        # Build context from recent memories
        context = self._build_conversation_context()

        # Adjust prompt based on emotion
        emotion_context = ""
        if emotion and emotion.dominant_emotion != "neutral":
            emotion_context = f"The user seems {emotion.dominant_emotion}. Respond empathetically."

        prompt = f"""You are a compassionate life story interviewer.

Context: {context}
User's response: {user_text}
{emotion_context}

Generate a brief, empathetic acknowledgment of what they shared, showing you're listening and care about their story. Keep it natural and conversational."""

        try:
            response = self.llm.generate_response(
                prompt,
                max_tokens=100,
                temperature=0.7,
            )
            return response
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return "Thank you for sharing that with me. It means a lot."

    def _select_next_question(self) -> Optional[str]:
        """Select the next question to ask.

        Returns:
            Next question or None
        """
        phase_questions = self.question_bank.get(self.state.phase.value, [])

        # Filter out asked questions
        available = [q for q in phase_questions if q not in self.asked_questions]

        if not available:
            # Move to next phase
            self._advance_phase()
            if self.state.phase == ConversationPhase.COMPLETE:
                return None
            return self._select_next_question()

        # Check if question is too similar to recent memories
        for question in available:
            embedding = self._generate_text_embedding(question)
            similar_memories = self.memory_store.search_memories(embedding, top_k=3)

            # If not too similar to recent memories, use it
            if not similar_memories or len(similar_memories) == 0:
                self.asked_questions.add(question)
                self.state.questions_asked += 1
                return question

        # Fallback to first available
        question = available[0]
        self.asked_questions.add(question)
        self.state.questions_asked += 1
        return question

    def _build_conversation_context(self) -> str:
        """Build context from recent conversation.

        Returns:
            Context string
        """
        # Get recent memories
        recent_memories = self.memory_store.get_all_memories(limit=5)

        context_parts = []
        for memory in recent_memories:
            context_parts.append(f"Q: {memory.question}\nA: {memory.answer}")

        return "\n\n".join(context_parts[-3:])  # Last 3 exchanges

    def _update_conversation_state(self) -> None:
        """Update conversation state based on progress."""
        # Update engagement based on response length and emotion
        # This is a simplified heuristic
        self.state.engagement_level = min(1.0, self.state.engagement_level * 0.95 + 0.05)

        # Check if should advance phase
        if self.state.questions_asked >= 5:  # 5 questions per phase
            self._advance_phase()

    def _advance_phase(self) -> None:
        """Advance to next conversation phase."""
        phases = list(ConversationPhase)
        current_index = phases.index(self.state.phase)

        if current_index < len(phases) - 1:
            self.state.phase = phases[current_index + 1]
            self.state.questions_asked = 0
            logger.info(f"Advanced to phase: {self.state.phase.value}")

    def _speak_and_record(self, text: str) -> None:
        """Speak text and record what was said.

        Args:
            text: Text to speak
        """
        logger.info(f"Speaking: {text}")
        self.tts.speak(text, blocking=True)
        self._current_question = text

    def _get_current_question(self) -> str:
        """Get the current question being asked.

        Returns:
            Current question
        """
        return getattr(self, "_current_question", "")

    def _save_session_summary(self) -> None:
        """Save summary of the session."""
        summary = {
            "session_start": self.state.session_start,
            "session_duration": time.time() - self.state.session_start,
            "questions_asked": self.state.questions_asked,
            "memories_created": len(self.session_memories),
            "final_phase": self.state.phase.value,
            "memory_ids": self.session_memories,
        }

        summary_path = Path("./data/sessions") / f"session_{int(self.state.session_start)}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Session summary saved to {summary_path}")

    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics.

        Returns:
            Session statistics
        """
        return {
            "phase": self.state.phase.value,
            "questions_asked": self.state.questions_asked,
            "session_duration": time.time() - self.state.session_start,
            "memories_created": len(self.session_memories),
            "user_mood": self.state.user_mood,
            "engagement_level": self.state.engagement_level,
            "is_active": self.is_active,
            "current_speaker": self.state.current_speaker_name,
        }

    def enroll_speaker(self, audio_path: str, name: str, role: str = "family") -> Tuple[bool, str]:
        """Enroll a new speaker for recognition.

        Args:
            audio_path: Path to enrollment audio (10-30 seconds)
            name: Speaker's name
            role: User role (prime, family, guest)

        Returns:
            Success status and user ID or error message
        """
        if not self.speaker_system:
            return False, "Speaker recognition not enabled"

        # Map role string to enum
        role_map = {
            "prime": UserRole.PRIME,
            "family": UserRole.FAMILY,
            "guest": UserRole.GUEST,
        }
        user_role = role_map.get(role.lower(), UserRole.FAMILY)

        # Enroll speaker
        success, result = self.speaker_system.enroll_speaker(
            audio_path, name, user_role
        )

        if success:
            # Create personalized memory profile
            if self.personalized_memory:
                self.personalized_memory.create_user_profile(
                    user_id=result,
                    name=name,
                    preferences={
                        "role": role,
                        "enrolled_at": time.time()
                    }
                )
            logger.info(f"Successfully enrolled speaker: {name} ({result})")
        else:
            logger.error(f"Failed to enroll speaker: {result}")

        return success, result

    def identify_current_speaker(self, audio: np.ndarray) -> Optional[str]:
        """Identify speaker from audio.

        Args:
            audio: Audio samples

        Returns:
            Speaker ID or None
        """
        if not self.speaker_system:
            return None

        user_id, confidence, _ = self.speaker_system.identify_speaker(audio)

        if user_id and confidence > 0.7:
            # Update conversation state
            speaker_info = self.speaker_system.get_speaker_info(user_id)
            if speaker_info:
                self.state.current_speaker_id = user_id
                self.state.current_speaker_name = speaker_info["name"]
                logger.info(f"Identified speaker: {speaker_info['name']} (confidence: {confidence:.2f})")
            return user_id

        return None

    def _process_user_response_with_speaker(self, text: str, emotion: Optional[EmotionState]) -> None:
        """Process user response with speaker identification.

        Args:
            text: User's response text
            emotion: Current emotion state
        """
        # Store memory based on speaker
        if self.personalized_memory and self.state.current_speaker_id:
            # Store in personalized memory
            current_question = self._get_current_question()

            # Determine privacy level based on content
            privacy = PrivacyLevel.PRIVATE
            if "family" in text.lower() or "everyone" in text.lower():
                privacy = PrivacyLevel.FAMILY

            success, memory_id = self.personalized_memory.store_memory(
                user_id=self.state.current_speaker_id,
                content=f"Q: {current_question}\nA: {text}",
                context={
                    "phase": self.state.phase.value,
                    "emotion": emotion.dominant_emotion if emotion else "neutral",
                    "session_time": time.time() - self.state.session_start,
                },
                privacy=privacy,
                importance=0.8,
                tags=[self.state.phase.value, "life_story"]
            )

            if success:
                self.session_memories.append(memory_id)
        else:
            # Fall back to legacy memory store
            self._process_user_response(text, emotion)

    def get_personalized_context(self, user_id: str, topic: str) -> str:
        """Get personalized context for a user.

        Args:
            user_id: User identifier
            topic: Topic to search for

        Returns:
            Personalized context string
        """
        if not self.personalized_memory:
            return ""

        memories = self.personalized_memory.retrieve_memories(
            user_id=user_id,
            query=topic,
            include_shared=True,
            max_results=5
        )

        if not memories:
            return ""

        context_parts = []
        for memory in memories:
            context_parts.append(memory.content)

        return "\n\n".join(context_parts)

    def list_enrolled_speakers(self) -> List[Dict[str, Any]]:
        """List all enrolled speakers.

        Returns:
            List of speaker information
        """
        if not self.speaker_system:
            return []

        speakers = []
        for user_id, profile in self.speaker_system.profiles.items():
            speakers.append({
                "user_id": user_id,
                "name": profile.name,
                "role": profile.role.value,
                "enrolled_at": profile.created_at.isoformat(),
                "last_seen": profile.last_seen.isoformat() if profile.last_seen else None,
            })

        return speakers

    def remove_speaker(self, user_id: str) -> bool:
        """Remove an enrolled speaker.

        Args:
            user_id: User ID to remove

        Returns:
            Success status
        """
        if not self.speaker_system:
            return False

        # Remove from speaker system
        removed = self.speaker_system.remove_speaker(user_id)

        # Clear personalized memories if requested
        if removed and self.personalized_memory:
            self.personalized_memory.clear_user_memories(user_id, confirm=True)

        return removed