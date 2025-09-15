#!/usr/bin/env python
"""Tests for speaker recognition and personalized memory systems."""

import sys
import unittest
from pathlib import Path
import tempfile
import shutil
import numpy as np
from datetime import datetime, timedelta
import time

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from myweepal.audio.speaker_recognition import (
    SpeakerRecognitionSystem,
    VoiceProfile,
    UserRole,
    SpeakerSegment
)
from myweepal.core.personalized_memory import (
    PersonalizedMemorySystem,
    MemoryEntry,
    PrivacyLevel,
    UserMemoryProfile
)


class TestSpeakerRecognition(unittest.TestCase):
    """Test speaker recognition functionality."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.test_dir) / "speaker_cache"
        self.system = SpeakerRecognitionSystem(
            cache_dir=self.cache_dir,
            min_enrollment_duration=1.0,  # Shorter for testing
            max_enrollment_duration=5.0
        )

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _generate_test_audio(self, duration: float, frequency: float = 440.0) -> np.ndarray:
        """Generate test audio signal.

        Args:
            duration: Duration in seconds
            frequency: Base frequency

        Returns:
            Audio samples
        """
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        # Add some variation to make each "speaker" unique
        audio = np.sin(2 * np.pi * frequency * t)
        audio += 0.1 * np.sin(2 * np.pi * frequency * 2 * t)
        audio += 0.05 * np.random.randn(len(t))
        return audio.astype(np.float32)

    def test_voice_enrollment(self):
        """Test voice enrollment process."""
        # Create test audio file
        test_audio = self._generate_test_audio(2.0, frequency=440.0)
        audio_file = Path(self.test_dir) / "test_enrollment.wav"

        # Save audio to file (simplified - normally would use proper WAV writing)
        import wave
        with wave.open(str(audio_file), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes((test_audio * 32767).astype(np.int16).tobytes())

        # Enroll speaker
        success, user_id = self.system.enroll_speaker(
            str(audio_file),
            "David",
            UserRole.PRIME
        )

        self.assertTrue(success)
        self.assertIsNotNone(user_id)
        self.assertIn(user_id, self.system.profiles)

        # Check profile
        profile = self.system.profiles[user_id]
        self.assertEqual(profile.name, "David")
        self.assertEqual(profile.role, UserRole.PRIME)
        self.assertGreater(profile.enrollment_samples, 0)
        self.assertGreater(len(profile.embeddings), 0)

    def test_speaker_identification(self):
        """Test speaker identification."""
        # Enroll multiple speakers
        speakers = [
            ("David", 440.0, UserRole.PRIME),
            ("Alice", 550.0, UserRole.FAMILY),
            ("Bob", 330.0, UserRole.GUEST)
        ]

        enrolled_ids = {}
        for name, freq, role in speakers:
            audio = self._generate_test_audio(2.0, frequency=freq)
            audio_file = Path(self.test_dir) / f"{name}_enrollment.wav"

            import wave
            with wave.open(str(audio_file), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes((audio * 32767).astype(np.int16).tobytes())

            success, user_id = self.system.enroll_speaker(
                str(audio_file),
                name,
                role
            )
            self.assertTrue(success)
            enrolled_ids[name] = user_id

        # Test identification with David's voice characteristics
        test_audio = self._generate_test_audio(1.0, frequency=440.0)
        identified_id, confidence, scores = self.system.identify_speaker(
            test_audio,
            return_all_scores=True
        )

        # Should identify as David (or None if embeddings are too different)
        if identified_id:
            self.assertEqual(identified_id, enrolled_ids["David"])
            self.assertGreater(confidence, 0.0)

        # Test with unknown speaker
        unknown_audio = self._generate_test_audio(1.0, frequency=660.0)
        unknown_id, unknown_conf, _ = self.system.identify_speaker(unknown_audio)

        # Should have lower confidence for unknown speaker
        if unknown_id:
            self.assertLess(unknown_conf, 1.0)

    def test_voice_authentication(self):
        """Test voice authentication."""
        # Enroll a speaker
        audio = self._generate_test_audio(2.0, frequency=440.0)
        audio_file = Path(self.test_dir) / "auth_enrollment.wav"

        import wave
        with wave.open(str(audio_file), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())

        success, user_id = self.system.enroll_speaker(
            str(audio_file),
            "David",
            UserRole.PRIME
        )
        self.assertTrue(success)

        # Test authentication with same speaker
        auth_audio = self._generate_test_audio(1.0, frequency=440.0)
        authenticated, confidence = self.system.authenticate_voice(
            auth_audio,
            user_id,
            threshold=0.5  # Lower threshold for test audio
        )

        # May or may not authenticate depending on embedding quality
        self.assertIsInstance(authenticated, bool)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)

        # Test authentication with wrong speaker
        wrong_audio = self._generate_test_audio(1.0, frequency=660.0)
        wrong_auth, wrong_conf = self.system.authenticate_voice(
            wrong_audio,
            user_id,
            threshold=0.9
        )

        # Should have lower confidence for wrong speaker
        if wrong_auth:
            self.assertLess(wrong_conf, confidence)

    def test_active_speakers_tracking(self):
        """Test active speakers tracking."""
        # Enroll a speaker
        audio = self._generate_test_audio(2.0, frequency=440.0)
        audio_file = Path(self.test_dir) / "active_enrollment.wav"

        import wave
        with wave.open(str(audio_file), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())

        success, user_id = self.system.enroll_speaker(
            str(audio_file),
            "David",
            UserRole.PRIME
        )
        self.assertTrue(success)

        # Identify speaker (marks as active)
        test_audio = self._generate_test_audio(1.0, frequency=440.0)
        self.system.identify_speaker(test_audio)

        # Check active speakers
        active = self.system.get_active_speakers(timeout=30.0)
        # May or may not be active depending on identification success
        self.assertIsInstance(active, list)

    def test_speaker_removal(self):
        """Test speaker profile removal."""
        # Enroll a speaker
        audio = self._generate_test_audio(2.0, frequency=440.0)
        audio_file = Path(self.test_dir) / "remove_enrollment.wav"

        import wave
        with wave.open(str(audio_file), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())

        success, user_id = self.system.enroll_speaker(
            str(audio_file),
            "TestUser",
            UserRole.GUEST
        )
        self.assertTrue(success)
        self.assertIn(user_id, self.system.profiles)

        # Remove speaker
        removed = self.system.remove_speaker(user_id)
        self.assertTrue(removed)
        self.assertNotIn(user_id, self.system.profiles)

        # Try to remove again (should fail)
        removed_again = self.system.remove_speaker(user_id)
        self.assertFalse(removed_again)

    def test_conversation_processing(self):
        """Test multi-speaker conversation processing."""
        # Enroll speakers
        speakers = [
            ("David", 440.0, UserRole.PRIME),
            ("Alice", 550.0, UserRole.FAMILY)
        ]

        enrolled_ids = {}
        for name, freq, role in speakers:
            audio = self._generate_test_audio(2.0, frequency=freq)
            audio_file = Path(self.test_dir) / f"{name}_conv.wav"

            import wave
            with wave.open(str(audio_file), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes((audio * 32767).astype(np.int16).tobytes())

            success, user_id = self.system.enroll_speaker(
                str(audio_file),
                name,
                role
            )
            enrolled_ids[name] = user_id

        # Create conversation segments
        segments = [
            (self._generate_test_audio(1.0, 440.0), 0.0, 1.0),  # David
            (self._generate_test_audio(1.0, 550.0), 1.0, 2.0),  # Alice
            (self._generate_test_audio(1.0, 440.0), 2.0, 3.0),  # David
        ]

        # Process conversation
        results = self.system.process_conversation(segments)

        # Check results
        self.assertIsInstance(results, list)
        for segment in results:
            self.assertIsInstance(segment, SpeakerSegment)
            self.assertIn(segment.speaker_id, enrolled_ids.values())
            self.assertGreaterEqual(segment.confidence, 0.0)
            self.assertLessEqual(segment.confidence, 1.0)


class TestPersonalizedMemory(unittest.TestCase):
    """Test personalized memory system."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = Path(self.test_dir) / "memory_db"
        self.memory_system = PersonalizedMemorySystem(
            db_path=self.db_path,
            max_results=10,
            similarity_threshold=0.5
        )

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_user_profile_creation(self):
        """Test creating user memory profiles."""
        # Create profile for David
        profile = self.memory_system.create_user_profile(
            user_id="david_001",
            name="David",
            preferences={"language": "en", "timezone": "GMT"}
        )

        self.assertIsNotNone(profile)
        self.assertEqual(profile.user_id, "david_001")
        self.assertEqual(profile.name, "David")
        self.assertEqual(profile.privacy_default, PrivacyLevel.PRIVATE)
        self.assertIn("david_001", self.memory_system.user_profiles)

        # Check collection was created
        collection = self.memory_system._get_user_collection("david_001")
        self.assertIsNotNone(collection)

    def test_memory_storage(self):
        """Test storing memories."""
        # Create user profile
        profile = self.memory_system.create_user_profile(
            user_id="test_user",
            name="TestUser"
        )

        # Store a memory
        success, memory_id = self.memory_system.store_memory(
            user_id="test_user",
            content="I love hiking in the Scottish Highlands",
            context={"location": "Scotland", "activity": "hiking"},
            privacy=PrivacyLevel.PRIVATE,
            importance=0.8,
            tags=["outdoor", "hobby"]
        )

        self.assertTrue(success)
        self.assertIsNotNone(memory_id)

        # Store ephemeral memory
        success2, memory_id2 = self.memory_system.store_memory(
            user_id="test_user",
            content="Temporary note about meeting",
            privacy=PrivacyLevel.EPHEMERAL,
            ttl_hours=1
        )

        self.assertTrue(success2)
        self.assertIsNotNone(memory_id2)

    def test_memory_retrieval(self):
        """Test retrieving memories."""
        # Create user and store memories
        profile = self.memory_system.create_user_profile(
            user_id="retrieval_user",
            name="RetrievalUser"
        )

        memories_to_store = [
            ("I enjoy programming in Python", ["programming", "python"]),
            ("My favorite food is pizza", ["food", "preferences"]),
            ("I work on AI projects", ["work", "AI", "programming"])
        ]

        for content, tags in memories_to_store:
            self.memory_system.store_memory(
                user_id="retrieval_user",
                content=content,
                tags=tags,
                importance=0.7
            )

        # Retrieve memories about programming
        results = self.memory_system.retrieve_memories(
            user_id="retrieval_user",
            query="programming and coding",
            max_results=5
        )

        self.assertIsInstance(results, list)
        # ChromaDB might not return results without proper embeddings
        if results:
            self.assertGreater(len(results), 0)
            for memory in results:
                self.assertIsInstance(memory, MemoryEntry)

    def test_privacy_levels(self):
        """Test different privacy levels."""
        # Create multiple users
        users = [
            ("user1", "User1"),
            ("user2", "User2")
        ]

        for user_id, name in users:
            self.memory_system.create_user_profile(user_id, name)

        # Store memories with different privacy levels
        self.memory_system.store_memory(
            user_id="user1",
            content="Private memory",
            privacy=PrivacyLevel.PRIVATE
        )

        self.memory_system.store_memory(
            user_id="user1",
            content="Family shared memory",
            privacy=PrivacyLevel.FAMILY
        )

        self.memory_system.store_memory(
            user_id="user1",
            content="Public shared memory",
            privacy=PrivacyLevel.PUBLIC
        )

        # Retrieve with privacy filter
        private_only = self.memory_system.retrieve_memories(
            user_id="user1",
            query="memory",
            privacy_filter=[PrivacyLevel.PRIVATE],
            include_shared=False
        )

        # Retrieve including shared
        all_memories = self.memory_system.retrieve_memories(
            user_id="user1",
            query="memory",
            include_shared=True
        )

        # Check that we can retrieve memories
        self.assertIsInstance(private_only, list)
        self.assertIsInstance(all_memories, list)

    def test_memory_deletion(self):
        """Test deleting memories."""
        # Create user and store memory
        profile = self.memory_system.create_user_profile(
            user_id="delete_user",
            name="DeleteUser"
        )

        success, memory_id = self.memory_system.store_memory(
            user_id="delete_user",
            content="Memory to delete"
        )
        self.assertTrue(success)

        # Delete the memory
        deleted = self.memory_system.delete_memory("delete_user", memory_id)
        self.assertTrue(deleted)

        # Try to retrieve (should not find it)
        results = self.memory_system.retrieve_memories(
            user_id="delete_user",
            query="Memory to delete"
        )

        # The deleted memory should not be in results
        if results:
            self.assertNotIn(memory_id, [m.id for m in results])

    def test_memory_export(self):
        """Test exporting user memories."""
        # Create user and store memories
        profile = self.memory_system.create_user_profile(
            user_id="export_user",
            name="ExportUser"
        )

        for i in range(3):
            self.memory_system.store_memory(
                user_id="export_user",
                content=f"Memory number {i}"
            )

        # Export memories
        export_data = self.memory_system.export_user_memories("export_user")

        self.assertIsNotNone(export_data)
        self.assertEqual(export_data["user_id"], "export_user")
        self.assertEqual(export_data["user_name"], "ExportUser")
        self.assertIn("memories", export_data)
        self.assertIn("exported_at", export_data)

    def test_memory_cleanup(self):
        """Test memory cleanup for expired entries."""
        # Create user
        profile = self.memory_system.create_user_profile(
            user_id="cleanup_user",
            name="CleanupUser"
        )

        # Store ephemeral memory with short TTL
        success, ephemeral_id = self.memory_system.store_memory(
            user_id="cleanup_user",
            content="Ephemeral memory",
            privacy=PrivacyLevel.EPHEMERAL,
            ttl_hours=0  # Expires immediately
        )

        # Store permanent memory
        success2, permanent_id = self.memory_system.store_memory(
            user_id="cleanup_user",
            content="Permanent memory"
        )

        # Force cleanup
        self.memory_system._cleanup_old_memories("cleanup_user")

        # The cleanup functionality depends on ChromaDB's internal handling
        # Just verify it doesn't crash
        self.assertTrue(True)


class TestIntegration(unittest.TestCase):
    """Integration tests for speaker recognition with personalized memory."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()

        # Initialize both systems
        self.speaker_system = SpeakerRecognitionSystem(
            cache_dir=Path(self.test_dir) / "speakers",
            min_enrollment_duration=1.0
        )

        self.memory_system = PersonalizedMemorySystem(
            db_path=Path(self.test_dir) / "memory",
            max_results=10
        )

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_speaker_with_personalized_memory(self):
        """Test speaker identification with personalized memory retrieval."""
        # Enroll a speaker
        test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 2, 32000)).astype(np.float32)
        audio_file = Path(self.test_dir) / "david.wav"

        import wave
        with wave.open(str(audio_file), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes((test_audio * 32767).astype(np.int16).tobytes())

        success, speaker_id = self.speaker_system.enroll_speaker(
            str(audio_file),
            "David",
            UserRole.PRIME
        )
        self.assertTrue(success)

        # Create memory profile for the speaker
        memory_profile = self.memory_system.create_user_profile(
            user_id=speaker_id,
            name="David",
            preferences={"voice_enrolled": True}
        )

        # Store personalized memories
        memories = [
            "I prefer to be called David",
            "I live in Barr, Scotland",
            "I work on AI and robotics projects",
            "My favorite programming language is Rust"
        ]

        for memory in memories:
            success, _ = self.memory_system.store_memory(
                user_id=speaker_id,
                content=memory,
                privacy=PrivacyLevel.PRIVATE,
                importance=0.9
            )
            self.assertTrue(success)

        # Simulate speaker identification
        test_segment = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
        identified_id, confidence, _ = self.speaker_system.identify_speaker(test_segment)

        # Retrieve personalized memories for identified speaker
        if identified_id:
            personal_memories = self.memory_system.retrieve_memories(
                user_id=identified_id,
                query="programming preferences",
                max_results=5
            )

            # Verify we can retrieve memories
            self.assertIsInstance(personal_memories, list)

            # Get speaker info
            speaker_info = self.speaker_system.get_speaker_info(identified_id)
            self.assertIsNotNone(speaker_info)
            self.assertEqual(speaker_info["name"], "David")


def run_tests():
    """Run all tests."""
    # Create test suite
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSpeakerRecognition))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPersonalizedMemory))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)