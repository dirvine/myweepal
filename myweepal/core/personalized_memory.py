"""Personalized memory system with per-user ChromaDB collections."""

import logging
import hashlib
import json
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import chromadb
from chromadb.config import Settings
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class PrivacyLevel(Enum):
    """Privacy levels for memory storage."""
    PUBLIC = "public"  # Shared across all users
    FAMILY = "family"  # Shared within family
    PRIVATE = "private"  # User-specific only
    EPHEMERAL = "ephemeral"  # Temporary, auto-delete


@dataclass
class MemoryEntry:
    """Individual memory entry."""
    id: str
    user_id: str
    content: str
    embedding: Optional[List[float]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    privacy: PrivacyLevel = PrivacyLevel.PRIVATE
    ttl_hours: Optional[int] = None  # Time to live for ephemeral memories
    importance: float = 0.5  # 0-1 importance score
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class UserMemoryProfile:
    """User-specific memory configuration."""
    user_id: str
    name: str
    collection_name: str
    max_memories: int = 10000
    retention_days: int = 365
    privacy_default: PrivacyLevel = PrivacyLevel.PRIVATE
    auto_summarize: bool = True
    preferences: Dict[str, Any] = field(default_factory=dict)


class PersonalizedMemorySystem:
    """Multi-user memory system with ChromaDB."""

    def __init__(
        self,
        db_path: Optional[Path] = None,
        embedding_model: Optional[Any] = None,
        max_results: int = 10,
        similarity_threshold: float = 0.7,
    ):
        """Initialize personalized memory system.

        Args:
            db_path: Path to ChromaDB storage
            embedding_model: Model for generating embeddings
            max_results: Maximum search results
            similarity_threshold: Minimum similarity for retrieval
        """
        self.db_path = db_path or Path.home() / ".myweepal" / "memory_db"
        self.embedding_model = embedding_model
        self.max_results = max_results
        self.similarity_threshold = similarity_threshold

        # Initialize ChromaDB
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )

        # User profiles
        self.user_profiles: Dict[str, UserMemoryProfile] = {}

        # Collections cache
        self.collections: Dict[str, chromadb.Collection] = {}

        # Shared collections
        self.shared_collection = None
        self.family_collection = None

        self._initialize_collections()
        self._load_user_profiles()

    def _initialize_collections(self) -> None:
        """Initialize shared collections."""
        try:
            # Shared public collection
            self.shared_collection = self.client.get_or_create_collection(
                name="shared_memories",
                metadata={"type": "public", "created": datetime.now().isoformat()}
            )

            # Family collection
            self.family_collection = self.client.get_or_create_collection(
                name="family_memories",
                metadata={"type": "family", "created": datetime.now().isoformat()}
            )

            logger.info("Initialized shared memory collections")
        except Exception as e:
            logger.error(f"Failed to initialize collections: {e}")

    def _load_user_profiles(self) -> None:
        """Load user profiles from disk."""
        profiles_file = self.db_path / "user_profiles.json"

        if profiles_file.exists():
            try:
                with open(profiles_file, "r") as f:
                    data = json.load(f)

                for user_id, profile_data in data.items():
                    profile = UserMemoryProfile(
                        user_id=user_id,
                        name=profile_data["name"],
                        collection_name=profile_data["collection_name"],
                        max_memories=profile_data.get("max_memories", 10000),
                        retention_days=profile_data.get("retention_days", 365),
                        privacy_default=PrivacyLevel(profile_data.get("privacy_default", "private")),
                        auto_summarize=profile_data.get("auto_summarize", True),
                        preferences=profile_data.get("preferences", {})
                    )
                    self.user_profiles[user_id] = profile

                    # Load user collection
                    self._get_user_collection(user_id)

                logger.info(f"Loaded {len(self.user_profiles)} user memory profiles")
            except Exception as e:
                logger.error(f"Failed to load user profiles: {e}")

    def _save_user_profiles(self) -> None:
        """Save user profiles to disk."""
        profiles_file = self.db_path / "user_profiles.json"

        try:
            data = {}
            for user_id, profile in self.user_profiles.items():
                data[user_id] = {
                    "name": profile.name,
                    "collection_name": profile.collection_name,
                    "max_memories": profile.max_memories,
                    "retention_days": profile.retention_days,
                    "privacy_default": profile.privacy_default.value,
                    "auto_summarize": profile.auto_summarize,
                    "preferences": profile.preferences
                }

            with open(profiles_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self.user_profiles)} user profiles")
        except Exception as e:
            logger.error(f"Failed to save user profiles: {e}")

    def create_user_profile(
        self,
        user_id: str,
        name: str,
        preferences: Optional[Dict[str, Any]] = None
    ) -> UserMemoryProfile:
        """Create a new user memory profile.

        Args:
            user_id: Unique user identifier
            name: User's name
            preferences: User preferences

        Returns:
            Created user profile
        """
        # Generate unique collection name
        collection_name = f"user_{user_id}_memories"

        profile = UserMemoryProfile(
            user_id=user_id,
            name=name,
            collection_name=collection_name,
            preferences=preferences or {}
        )

        self.user_profiles[user_id] = profile

        # Create ChromaDB collection
        self._get_user_collection(user_id)

        # Save profiles
        self._save_user_profiles()

        logger.info(f"Created memory profile for user: {name} ({user_id})")
        return profile

    def _get_user_collection(self, user_id: str) -> Optional[chromadb.Collection]:
        """Get or create user-specific collection.

        Args:
            user_id: User identifier

        Returns:
            ChromaDB collection or None
        """
        if user_id not in self.user_profiles:
            logger.error(f"No profile for user: {user_id}")
            return None

        profile = self.user_profiles[user_id]

        if user_id not in self.collections:
            try:
                collection = self.client.get_or_create_collection(
                    name=profile.collection_name,
                    metadata={
                        "user_id": user_id,
                        "user_name": profile.name,
                        "created": datetime.now().isoformat()
                    }
                )
                self.collections[user_id] = collection
                logger.info(f"Loaded collection for user: {profile.name}")
            except Exception as e:
                logger.error(f"Failed to get collection for user {user_id}: {e}")
                return None

        return self.collections.get(user_id)

    def store_memory(
        self,
        user_id: str,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        privacy: Optional[PrivacyLevel] = None,
        ttl_hours: Optional[int] = None,
        importance: float = 0.5,
        tags: Optional[List[str]] = None
    ) -> Tuple[bool, str]:
        """Store a memory for a user.

        Args:
            user_id: User identifier
            content: Memory content
            context: Additional context
            privacy: Privacy level
            ttl_hours: Time to live for ephemeral memories
            importance: Importance score (0-1)
            tags: Memory tags

        Returns:
            Success status and memory ID or error message
        """
        if user_id not in self.user_profiles:
            return False, f"No profile for user: {user_id}"

        profile = self.user_profiles[user_id]
        collection = self._get_user_collection(user_id)

        if not collection:
            return False, "Failed to access user collection"

        try:
            # Generate memory ID
            memory_id = self._generate_memory_id(user_id, content)

            # Use default privacy if not specified
            privacy = privacy or profile.privacy_default

            # Create memory entry
            memory = MemoryEntry(
                id=memory_id,
                user_id=user_id,
                content=content,
                timestamp=datetime.now(),
                context=context or {},
                privacy=privacy,
                ttl_hours=ttl_hours,
                importance=importance,
                tags=tags or []
            )

            # Generate embedding
            embedding = self._generate_embedding(content)

            # Store in appropriate collection(s)
            metadata = {
                "user_id": user_id,
                "timestamp": memory.timestamp.isoformat(),
                "privacy": privacy.value,
                "importance": importance,
                "tags": json.dumps(tags or []),
                "context": json.dumps(context or {})
            }

            if ttl_hours:
                metadata["ttl_hours"] = ttl_hours
                metadata["expires_at"] = (
                    datetime.now() + timedelta(hours=ttl_hours)
                ).isoformat()

            # Store in user collection
            collection.add(
                ids=[memory_id],
                embeddings=[embedding] if embedding else None,
                documents=[content],
                metadatas=[metadata]
            )

            # Store in shared collections based on privacy
            if privacy == PrivacyLevel.PUBLIC and self.shared_collection:
                self.shared_collection.add(
                    ids=[memory_id],
                    embeddings=[embedding] if embedding else None,
                    documents=[content],
                    metadatas=[metadata]
                )
            elif privacy == PrivacyLevel.FAMILY and self.family_collection:
                self.family_collection.add(
                    ids=[memory_id],
                    embeddings=[embedding] if embedding else None,
                    documents=[content],
                    metadatas=[metadata]
                )

            # Clean up old memories if needed
            self._cleanup_old_memories(user_id)

            logger.info(f"Stored memory for {profile.name}: {memory_id[:8]}...")
            return True, memory_id

        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return False, str(e)

    def retrieve_memories(
        self,
        user_id: str,
        query: str,
        include_shared: bool = True,
        privacy_filter: Optional[List[PrivacyLevel]] = None,
        max_results: Optional[int] = None
    ) -> List[MemoryEntry]:
        """Retrieve relevant memories for a user.

        Args:
            user_id: User identifier
            query: Search query
            include_shared: Include shared/family memories
            privacy_filter: Filter by privacy levels
            max_results: Maximum results to return

        Returns:
            List of relevant memory entries
        """
        if user_id not in self.user_profiles:
            logger.error(f"No profile for user: {user_id}")
            return []

        memories = []
        max_results = max_results or self.max_results

        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)

            # Search user's personal collection
            collection = self._get_user_collection(user_id)
            if collection:
                results = collection.query(
                    query_embeddings=[query_embedding] if query_embedding else None,
                    query_texts=[query] if not query_embedding else None,
                    n_results=max_results,
                    where=self._build_privacy_filter(privacy_filter) if privacy_filter else None
                )

                memories.extend(self._parse_query_results(results))

            # Search shared collections if requested
            if include_shared:
                # Family memories
                if self.family_collection:
                    family_results = self.family_collection.query(
                        query_embeddings=[query_embedding] if query_embedding else None,
                        query_texts=[query] if not query_embedding else None,
                        n_results=max_results // 2
                    )
                    memories.extend(self._parse_query_results(family_results))

                # Public memories
                if self.shared_collection:
                    public_results = self.shared_collection.query(
                        query_embeddings=[query_embedding] if query_embedding else None,
                        query_texts=[query] if not query_embedding else None,
                        n_results=max_results // 2
                    )
                    memories.extend(self._parse_query_results(public_results))

            # Sort by relevance/importance
            memories.sort(key=lambda m: m.importance, reverse=True)

            # Update access counts
            for memory in memories:
                self._update_access_count(user_id, memory.id)

            return memories[:max_results]

        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            return []

    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text.

        Args:
            text: Input text

        Returns:
            Embedding vector or None
        """
        if self.embedding_model:
            try:
                # Use provided embedding model
                embedding = self.embedding_model.encode(text)
                return embedding.tolist()
            except:
                pass

        # Fallback: Use ChromaDB's default embedding
        return None

    def _generate_memory_id(self, user_id: str, content: str) -> str:
        """Generate unique memory ID.

        Args:
            user_id: User identifier
            content: Memory content

        Returns:
            Unique memory ID
        """
        timestamp = str(datetime.now().timestamp())
        unique_string = f"{user_id}_{content[:50]}_{timestamp}"
        return hashlib.md5(unique_string.encode()).hexdigest()

    def _build_privacy_filter(self, privacy_levels: List[PrivacyLevel]) -> Dict[str, Any]:
        """Build ChromaDB filter for privacy levels.

        Args:
            privacy_levels: List of privacy levels to filter

        Returns:
            ChromaDB where clause
        """
        return {
            "privacy": {"$in": [p.value for p in privacy_levels]}
        }

    def _parse_query_results(self, results: Dict[str, Any]) -> List[MemoryEntry]:
        """Parse ChromaDB query results into MemoryEntry objects.

        Args:
            results: ChromaDB query results

        Returns:
            List of memory entries
        """
        memories = []

        if not results or not results.get("ids"):
            return memories

        for i, doc_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i] if results.get("metadatas") else {}

            memory = MemoryEntry(
                id=doc_id,
                user_id=metadata.get("user_id", ""),
                content=results["documents"][0][i] if results.get("documents") else "",
                timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.now().isoformat())),
                context=json.loads(metadata.get("context", "{}")),
                privacy=PrivacyLevel(metadata.get("privacy", "private")),
                ttl_hours=metadata.get("ttl_hours"),
                importance=float(metadata.get("importance", 0.5)),
                tags=json.loads(metadata.get("tags", "[]"))
            )
            memories.append(memory)

        return memories

    def _update_access_count(self, user_id: str, memory_id: str) -> None:
        """Update access count for a memory.

        Args:
            user_id: User identifier
            memory_id: Memory identifier
        """
        # This would update the metadata in ChromaDB
        # Implementation depends on ChromaDB version
        pass

    def _cleanup_old_memories(self, user_id: str) -> None:
        """Clean up old or expired memories.

        Args:
            user_id: User identifier
        """
        if user_id not in self.user_profiles:
            return

        profile = self.user_profiles[user_id]
        collection = self._get_user_collection(user_id)

        if not collection:
            return

        try:
            # Get all memories
            all_memories = collection.get()

            if not all_memories or not all_memories.get("ids"):
                return

            current_time = datetime.now()
            ids_to_delete = []

            for i, memory_id in enumerate(all_memories["ids"]):
                metadata = all_memories["metadatas"][i] if all_memories.get("metadatas") else {}

                # Check TTL expiration
                if metadata.get("expires_at"):
                    expires_at = datetime.fromisoformat(metadata["expires_at"])
                    if current_time > expires_at:
                        ids_to_delete.append(memory_id)
                        continue

                # Check retention period
                if metadata.get("timestamp"):
                    created_at = datetime.fromisoformat(metadata["timestamp"])
                    if (current_time - created_at).days > profile.retention_days:
                        ids_to_delete.append(memory_id)

            # Delete expired memories
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                logger.info(f"Cleaned up {len(ids_to_delete)} expired memories for {profile.name}")

            # Check max memories limit
            total_memories = collection.count()
            if total_memories > profile.max_memories:
                # Delete oldest memories
                # This would require sorting by timestamp and deleting oldest
                pass

        except Exception as e:
            logger.error(f"Failed to cleanup memories: {e}")

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """Delete a specific memory.

        Args:
            user_id: User identifier
            memory_id: Memory identifier

        Returns:
            Success status
        """
        collection = self._get_user_collection(user_id)

        if not collection:
            return False

        try:
            collection.delete(ids=[memory_id])

            # Also delete from shared collections
            if self.shared_collection:
                self.shared_collection.delete(ids=[memory_id])
            if self.family_collection:
                self.family_collection.delete(ids=[memory_id])

            logger.info(f"Deleted memory: {memory_id[:8]}...")
            return True

        except Exception as e:
            logger.error(f"Failed to delete memory: {e}")
            return False

    def export_user_memories(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Export all memories for a user.

        Args:
            user_id: User identifier

        Returns:
            Exported data or None
        """
        if user_id not in self.user_profiles:
            return None

        profile = self.user_profiles[user_id]
        collection = self._get_user_collection(user_id)

        if not collection:
            return None

        try:
            all_memories = collection.get()

            return {
                "user_id": user_id,
                "user_name": profile.name,
                "exported_at": datetime.now().isoformat(),
                "total_memories": collection.count(),
                "memories": all_memories
            }

        except Exception as e:
            logger.error(f"Failed to export memories: {e}")
            return None

    def clear_user_memories(self, user_id: str, confirm: bool = False) -> bool:
        """Clear all memories for a user.

        Args:
            user_id: User identifier
            confirm: Confirmation flag

        Returns:
            Success status
        """
        if not confirm:
            logger.warning("Clear operation requires confirmation")
            return False

        collection = self._get_user_collection(user_id)

        if not collection:
            return False

        try:
            # Delete the collection
            self.client.delete_collection(collection.name)

            # Remove from cache
            if user_id in self.collections:
                del self.collections[user_id]

            # Recreate empty collection
            self._get_user_collection(user_id)

            logger.info(f"Cleared all memories for user: {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to clear memories: {e}")
            return False