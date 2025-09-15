"""Memory storage and retrieval using ChromaDB."""

import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import chromadb
from chromadb.config import Settings
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Memory:
    """Represents a single memory entry."""

    id: str
    question: str
    answer: str
    timestamp: float
    category: Optional[str] = None
    emotion: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary."""
        return asdict(self)


class MemoryStore:
    """Vector database for storing and retrieving memories."""

    def __init__(
        self,
        persist_directory: str = "./data/chroma",
        collection_name: str = "myweepal_memories",
    ):
        """Initialize memory store.

        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of the ChromaDB collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Initialized memory store: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize memory store: {e}")
            raise RuntimeError(f"Memory store initialization failed: {e}") from e

    def add_memory(
        self,
        question: str,
        answer: str,
        embedding: np.ndarray,
        category: Optional[str] = None,
        emotion: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a new memory to the store.

        Args:
            question: The question asked
            answer: The user's answer
            embedding: Vector embedding of the Q&A pair
            category: Optional category (e.g., "childhood", "career")
            emotion: Optional detected emotion
            metadata: Additional metadata

        Returns:
            Memory ID
        """
        memory_id = str(uuid.uuid4())
        timestamp = time.time()

        memory = Memory(
            id=memory_id,
            question=question,
            answer=answer,
            timestamp=timestamp,
            category=category,
            emotion=emotion,
            metadata=metadata or {},
        )

        try:
            self.collection.add(
                embeddings=[embedding.tolist()],
                documents=[f"Q: {question}\nA: {answer}"],
                metadatas=[
                    {
                        "question": question,
                        "answer": answer,
                        "timestamp": str(timestamp),
                        "category": category or "",
                        "emotion": emotion or "",
                        **(metadata or {}),
                    }
                ],
                ids=[memory_id],
            )
            logger.info(f"Added memory: {memory_id}")
            return memory_id
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            raise RuntimeError(f"Failed to store memory: {e}") from e

    def search_memories(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        category: Optional[str] = None,
        time_range: Optional[Tuple[float, float]] = None,
    ) -> List[Memory]:
        """Search for similar memories.

        Args:
            query_embedding: Query vector embedding
            top_k: Number of results to return
            category: Filter by category
            time_range: Filter by time range (start, end)

        Returns:
            List of matching memories
        """
        try:
            # Build where clause for filtering
            where_clause = {}
            if category:
                where_clause["category"] = category

            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where_clause if where_clause else None,
            )

            memories = []
            if results["ids"] and results["ids"][0]:
                for i, memory_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i]

                    # Apply time range filter if specified
                    timestamp = float(metadata.get("timestamp", 0))
                    if time_range and not (time_range[0] <= timestamp <= time_range[1]):
                        continue

                    memory = Memory(
                        id=memory_id,
                        question=metadata.get("question", ""),
                        answer=metadata.get("answer", ""),
                        timestamp=timestamp,
                        category=metadata.get("category") or None,
                        emotion=metadata.get("emotion") or None,
                        metadata={
                            k: v
                            for k, v in metadata.items()
                            if k not in ["question", "answer", "timestamp", "category", "emotion"]
                        },
                    )
                    memories.append(memory)

            return memories
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            raise RuntimeError(f"Memory search failed: {e}") from e

    def get_memory_by_id(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a specific memory by ID.

        Args:
            memory_id: Memory ID

        Returns:
            Memory object or None if not found
        """
        try:
            results = self.collection.get(ids=[memory_id])

            if results["ids"]:
                metadata = results["metadatas"][0]
                return Memory(
                    id=memory_id,
                    question=metadata.get("question", ""),
                    answer=metadata.get("answer", ""),
                    timestamp=float(metadata.get("timestamp", 0)),
                    category=metadata.get("category") or None,
                    emotion=metadata.get("emotion") or None,
                    metadata={
                        k: v
                        for k, v in metadata.items()
                        if k not in ["question", "answer", "timestamp", "category", "emotion"]
                    },
                )
            return None
        except Exception as e:
            logger.error(f"Failed to get memory: {e}")
            return None

    def get_all_memories(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Memory]:
        """Get all memories with pagination.

        Args:
            limit: Maximum number of memories to return
            offset: Number of memories to skip

        Returns:
            List of memories
        """
        try:
            # ChromaDB doesn't have direct pagination, so we get all and slice
            results = self.collection.get()

            memories = []
            if results["ids"]:
                for i, memory_id in enumerate(results["ids"]):
                    metadata = results["metadatas"][i]
                    memory = Memory(
                        id=memory_id,
                        question=metadata.get("question", ""),
                        answer=metadata.get("answer", ""),
                        timestamp=float(metadata.get("timestamp", 0)),
                        category=metadata.get("category") or None,
                        emotion=metadata.get("emotion") or None,
                        metadata={
                            k: v
                            for k, v in metadata.items()
                            if k not in ["question", "answer", "timestamp", "category", "emotion"]
                        },
                    )
                    memories.append(memory)

            # Sort by timestamp
            memories.sort(key=lambda m: m.timestamp)

            # Apply pagination
            if limit:
                memories = memories[offset : offset + limit]
            else:
                memories = memories[offset:]

            return memories
        except Exception as e:
            logger.error(f"Failed to get all memories: {e}")
            raise RuntimeError(f"Failed to retrieve memories: {e}") from e

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory.

        Args:
            memory_id: Memory ID to delete

        Returns:
            True if deleted, False otherwise
        """
        try:
            self.collection.delete(ids=[memory_id])
            logger.info(f"Deleted memory: {memory_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory: {e}")
            return False

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories.

        Returns:
            Dictionary with memory statistics
        """
        try:
            count = self.collection.count()
            results = self.collection.get()

            categories = {}
            emotions = {}
            if results["metadatas"]:
                for metadata in results["metadatas"]:
                    cat = metadata.get("category", "uncategorized")
                    categories[cat] = categories.get(cat, 0) + 1

                    emo = metadata.get("emotion", "neutral")
                    emotions[emo] = emotions.get(emo, 0) + 1

            return {
                "total_memories": count,
                "categories": categories,
                "emotions": emotions,
            }
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {"total_memories": 0, "categories": {}, "emotions": {}}