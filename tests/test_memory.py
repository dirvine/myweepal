"""Tests for memory storage module."""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from myweepal.core.memory import MemoryStore, Memory


class TestMemoryStore:
    """Test suite for memory storage."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def memory_store(self, temp_dir):
        """Create memory store instance."""
        return MemoryStore(persist_directory=temp_dir)

    @pytest.fixture
    def sample_embedding(self):
        """Create sample embedding."""
        return np.random.randn(768)

    def test_initialization(self, temp_dir):
        """Test memory store initialization."""
        store = MemoryStore(persist_directory=temp_dir)

        assert store.persist_directory == temp_dir
        assert store.collection_name == "myweepal_memories"
        assert store.client is not None
        assert store.collection is not None

    def test_initialization_failure(self):
        """Test memory store initialization failure."""
        with patch("myweepal.core.memory.chromadb.PersistentClient") as mock_client:
            mock_client.side_effect = Exception("DB init failed")

            with pytest.raises(RuntimeError) as exc_info:
                MemoryStore("/invalid/path")

            assert "Memory store initialization failed" in str(exc_info.value)

    def test_add_memory(self, memory_store, sample_embedding):
        """Test adding a memory."""
        memory_id = memory_store.add_memory(
            question="What is your name?",
            answer="My name is Test",
            embedding=sample_embedding,
            category="introduction",
            emotion="happy",
            metadata={"session": "test"},
        )

        assert memory_id is not None
        assert isinstance(memory_id, str)

        # Verify memory was added
        memory = memory_store.get_memory_by_id(memory_id)
        assert memory is not None
        assert memory.question == "What is your name?"
        assert memory.answer == "My name is Test"
        assert memory.category == "introduction"
        assert memory.emotion == "happy"

    def test_add_memory_failure(self, memory_store, sample_embedding):
        """Test memory addition failure handling."""
        with patch.object(memory_store.collection, "add") as mock_add:
            mock_add.side_effect = Exception("Add failed")

            with pytest.raises(RuntimeError) as exc_info:
                memory_store.add_memory(
                    question="Test?",
                    answer="Test",
                    embedding=sample_embedding,
                )

            assert "Failed to store memory" in str(exc_info.value)

    def test_search_memories(self, memory_store, sample_embedding):
        """Test searching memories."""
        # Add some memories
        for i in range(5):
            memory_store.add_memory(
                question=f"Question {i}?",
                answer=f"Answer {i}",
                embedding=sample_embedding + np.random.randn(768) * 0.1,
                category="test",
                emotion="neutral",
            )

        # Search
        results = memory_store.search_memories(sample_embedding, top_k=3)

        assert len(results) <= 3
        for memory in results:
            assert isinstance(memory, Memory)
            assert memory.question.startswith("Question")
            assert memory.answer.startswith("Answer")

    def test_search_memories_with_category_filter(self, memory_store, sample_embedding):
        """Test searching with category filter."""
        # Add memories with different categories
        memory_store.add_memory(
            question="Q1?",
            answer="A1",
            embedding=sample_embedding,
            category="cat1",
        )
        memory_store.add_memory(
            question="Q2?",
            answer="A2",
            embedding=sample_embedding + np.random.randn(768) * 0.1,
            category="cat2",
        )

        # Search with category filter
        results = memory_store.search_memories(
            sample_embedding,
            top_k=5,
            category="cat1",
        )

        # Should only return memories from cat1
        for memory in results:
            assert memory.category == "cat1"

    def test_search_memories_failure(self, memory_store, sample_embedding):
        """Test memory search failure handling."""
        with patch.object(memory_store.collection, "query") as mock_query:
            mock_query.side_effect = Exception("Query failed")

            with pytest.raises(RuntimeError) as exc_info:
                memory_store.search_memories(sample_embedding)

            assert "Memory search failed" in str(exc_info.value)

    def test_get_memory_by_id(self, memory_store, sample_embedding):
        """Test retrieving memory by ID."""
        # Add a memory
        memory_id = memory_store.add_memory(
            question="Test question?",
            answer="Test answer",
            embedding=sample_embedding,
        )

        # Retrieve it
        memory = memory_store.get_memory_by_id(memory_id)

        assert memory is not None
        assert memory.id == memory_id
        assert memory.question == "Test question?"
        assert memory.answer == "Test answer"

    def test_get_memory_by_id_not_found(self, memory_store):
        """Test retrieving non-existent memory."""
        memory = memory_store.get_memory_by_id("non-existent-id")
        assert memory is None

    def test_get_all_memories(self, memory_store, sample_embedding):
        """Test getting all memories."""
        # Add multiple memories
        num_memories = 10
        for i in range(num_memories):
            memory_store.add_memory(
                question=f"Q{i}?",
                answer=f"A{i}",
                embedding=sample_embedding + np.random.randn(768) * 0.1,
            )

        # Get all
        memories = memory_store.get_all_memories()

        assert len(memories) == num_memories
        for memory in memories:
            assert isinstance(memory, Memory)

    def test_get_all_memories_with_pagination(self, memory_store, sample_embedding):
        """Test getting memories with pagination."""
        # Add memories
        for i in range(10):
            memory_store.add_memory(
                question=f"Q{i}?",
                answer=f"A{i}",
                embedding=sample_embedding,
            )

        # Get with limit
        memories = memory_store.get_all_memories(limit=5)
        assert len(memories) == 5

        # Get with offset
        memories = memory_store.get_all_memories(limit=5, offset=5)
        assert len(memories) == 5

    def test_delete_memory(self, memory_store, sample_embedding):
        """Test deleting a memory."""
        # Add a memory
        memory_id = memory_store.add_memory(
            question="To delete?",
            answer="Yes",
            embedding=sample_embedding,
        )

        # Delete it
        result = memory_store.delete_memory(memory_id)
        assert result is True

        # Verify it's gone
        memory = memory_store.get_memory_by_id(memory_id)
        assert memory is None

    def test_delete_memory_failure(self, memory_store):
        """Test deletion failure handling."""
        with patch.object(memory_store.collection, "delete") as mock_delete:
            mock_delete.side_effect = Exception("Delete failed")

            result = memory_store.delete_memory("some-id")
            assert result is False

    def test_get_memory_stats(self, memory_store, sample_embedding):
        """Test getting memory statistics."""
        # Add memories with different categories and emotions
        categories = ["childhood", "education", "career"]
        emotions = ["happy", "sad", "neutral"]

        for i in range(9):
            memory_store.add_memory(
                question=f"Q{i}?",
                answer=f"A{i}",
                embedding=sample_embedding,
                category=categories[i % 3],
                emotion=emotions[i % 3],
            )

        stats = memory_store.get_memory_stats()

        assert stats["total_memories"] == 9
        assert len(stats["categories"]) == 3
        assert len(stats["emotions"]) == 3
        assert all(count == 3 for count in stats["categories"].values())
        assert all(count == 3 for count in stats["emotions"].values())

    def test_get_memory_stats_empty(self, memory_store):
        """Test getting stats for empty store."""
        stats = memory_store.get_memory_stats()

        assert stats["total_memories"] == 0
        assert stats["categories"] == {}
        assert stats["emotions"] == {}

    def test_memory_dataclass(self):
        """Test Memory dataclass."""
        memory = Memory(
            id="test-id",
            question="Q?",
            answer="A",
            timestamp=123.456,
            category="test",
            emotion="happy",
            metadata={"key": "value"},
        )

        # Test to_dict
        memory_dict = memory.to_dict()
        assert memory_dict["id"] == "test-id"
        assert memory_dict["question"] == "Q?"
        assert memory_dict["answer"] == "A"
        assert memory_dict["timestamp"] == 123.456
        assert memory_dict["category"] == "test"
        assert memory_dict["emotion"] == "happy"
        assert memory_dict["metadata"] == {"key": "value"}