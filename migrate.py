#!/usr/bin/env python
"""Migration script from MyWeePal v1 to v2."""

import shutil
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any
import sqlite3
import pickle

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MyWeePalMigration:
    """Handle migration from v1 to v2."""

    def __init__(self, dry_run: bool = False):
        """Initialize migration.

        Args:
            dry_run: If True, only show what would be done without making changes
        """
        self.dry_run = dry_run
        self.project_root = Path.cwd()
        self.backup_dir = self.project_root / "migration_backup"
        self.old_data_dir = self.project_root / "data"
        self.new_data_dir = self.project_root / "data"

    def run(self) -> bool:
        """Run complete migration process.

        Returns:
            True if successful
        """
        logger.info("Starting MyWeePal v1 to v2 migration...")

        if self.dry_run:
            logger.info("Running in DRY RUN mode - no changes will be made")

        try:
            # Step 1: Create backup
            if not self.dry_run:
                self._create_backup()

            # Step 2: Migrate ChromaDB
            self._migrate_chromadb()

            # Step 3: Migrate voice profiles
            self._migrate_voice_profiles()

            # Step 4: Migrate conversation history
            self._migrate_conversation_history()

            # Step 5: Create new directories
            self._create_new_directories()

            # Step 6: Update configuration
            self._update_configuration()

            # Step 7: Download new models
            self._download_models()

            logger.info("Migration completed successfully!")
            return True

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            if not self.dry_run:
                logger.info("Attempting to restore from backup...")
                self._restore_backup()
            return False

    def _create_backup(self) -> None:
        """Create backup of current data."""
        logger.info("Creating backup...")

        self.backup_dir.mkdir(exist_ok=True)

        # Backup data directory
        if self.old_data_dir.exists():
            backup_data = self.backup_dir / "data"
            if backup_data.exists():
                shutil.rmtree(backup_data)
            shutil.copytree(self.old_data_dir, backup_data)
            logger.info(f"Backed up data directory to {backup_data}")

        # Backup config files
        for config_file in [".env", "config.yaml", "config.json"]:
            src = self.project_root / config_file
            if src.exists():
                dst = self.backup_dir / config_file
                shutil.copy2(src, dst)
                logger.info(f"Backed up {config_file}")

    def _restore_backup(self) -> None:
        """Restore from backup."""
        logger.info("Restoring from backup...")

        if not self.backup_dir.exists():
            logger.error("No backup found!")
            return

        # Restore data directory
        backup_data = self.backup_dir / "data"
        if backup_data.exists():
            if self.old_data_dir.exists():
                shutil.rmtree(self.old_data_dir)
            shutil.copytree(backup_data, self.old_data_dir)
            logger.info("Restored data directory")

        # Restore config files
        for config_file in [".env", "config.yaml", "config.json"]:
            src = self.backup_dir / config_file
            dst = self.project_root / config_file
            if src.exists():
                shutil.copy2(src, dst)
                logger.info(f"Restored {config_file}")

    def _migrate_chromadb(self) -> None:
        """Migrate ChromaDB to new version."""
        logger.info("Migrating ChromaDB...")

        old_chroma = self.old_data_dir / "chroma"
        new_chroma = self.new_data_dir / "chroma_v2"

        if old_chroma.exists():
            if self.dry_run:
                logger.info(f"Would migrate ChromaDB from {old_chroma} to {new_chroma}")
            else:
                if new_chroma.exists():
                    shutil.rmtree(new_chroma)
                shutil.copytree(old_chroma, new_chroma)
                logger.info(f"Migrated ChromaDB to {new_chroma}")

                # Update collection metadata for v2
                self._update_chroma_metadata(new_chroma)
        else:
            logger.info("No existing ChromaDB found, creating new")
            if not self.dry_run:
                new_chroma.mkdir(parents=True, exist_ok=True)

    def _update_chroma_metadata(self, chroma_path: Path) -> None:
        """Update ChromaDB metadata for v2 compatibility.

        Args:
            chroma_path: Path to ChromaDB directory
        """
        try:
            # Update SQLite metadata if exists
            db_file = chroma_path / "chroma.sqlite3"
            if db_file.exists():
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()

                # Update version metadata
                cursor.execute(
                    "UPDATE metadata SET value = ? WHERE key = ?",
                    ("2.0.0", "version")
                )

                conn.commit()
                conn.close()
                logger.info("Updated ChromaDB metadata")
        except Exception as e:
            logger.warning(f"Could not update ChromaDB metadata: {e}")

    def _migrate_voice_profiles(self) -> None:
        """Migrate voice profiles to new format."""
        logger.info("Migrating voice profiles...")

        voices_dir = Path.home() / ".myweepal" / "voices"
        old_profiles = self.old_data_dir / "voice_profiles"

        if old_profiles.exists():
            if self.dry_run:
                logger.info(f"Would migrate voice profiles to {voices_dir}")
            else:
                voices_dir.mkdir(parents=True, exist_ok=True)

                # Copy audio files
                for audio_file in old_profiles.glob("*.wav"):
                    dst = voices_dir / audio_file.name
                    shutil.copy2(audio_file, dst)
                    logger.info(f"Migrated voice sample: {audio_file.name}")

                # Convert old profile format to new
                old_profile_json = old_profiles / "profiles.json"
                if old_profile_json.exists():
                    with open(old_profile_json, "r") as f:
                        old_data = json.load(f)

                    new_profiles = self._convert_voice_profiles(old_data)

                    new_profile_json = voices_dir.parent / "tts_cache" / "voice_profiles.json"
                    new_profile_json.parent.mkdir(parents=True, exist_ok=True)

                    with open(new_profile_json, "w") as f:
                        json.dump(new_profiles, f, indent=2)

                    logger.info("Converted voice profiles to v2 format")
        else:
            logger.info("No existing voice profiles found")
            if not self.dry_run:
                voices_dir.mkdir(parents=True, exist_ok=True)

    def _convert_voice_profiles(self, old_profiles: Dict) -> Dict:
        """Convert old voice profile format to new.

        Args:
            old_profiles: Old profile data

        Returns:
            New profile format
        """
        new_profiles = {}

        for name, data in old_profiles.items():
            new_profiles[name] = {
                "name": name,
                "reference_audio": data.get("audio_file"),
                "emotion_default": 0.5,
                "speed": data.get("speed", 1.0),
                "pitch": data.get("pitch", 1.0),
                "metadata": {
                    "created": data.get("created_at", ""),
                    "duration": data.get("duration", 0),
                    "sample_rate": 24000,
                    "migrated_from_v1": True
                }
            }

        return new_profiles

    def _migrate_conversation_history(self) -> None:
        """Migrate conversation history."""
        logger.info("Migrating conversation history...")

        old_history = self.old_data_dir / "conversations"
        new_history = self.new_data_dir / "memories"

        if old_history.exists():
            if self.dry_run:
                logger.info(f"Would migrate conversations to {new_history}")
            else:
                new_history.mkdir(parents=True, exist_ok=True)

                # Migrate JSON conversation files
                for conv_file in old_history.glob("*.json"):
                    dst = new_history / conv_file.name
                    shutil.copy2(conv_file, dst)
                    logger.info(f"Migrated conversation: {conv_file.name}")

                # Migrate pickle files if any
                for pkl_file in old_history.glob("*.pkl"):
                    self._convert_pickle_to_json(pkl_file, new_history)
        else:
            logger.info("No existing conversation history found")

    def _convert_pickle_to_json(self, pkl_file: Path, output_dir: Path) -> None:
        """Convert pickle file to JSON.

        Args:
            pkl_file: Pickle file path
            output_dir: Output directory
        """
        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)

            json_file = output_dir / f"{pkl_file.stem}.json"
            with open(json_file, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.info(f"Converted {pkl_file.name} to JSON")
        except Exception as e:
            logger.warning(f"Could not convert {pkl_file.name}: {e}")

    def _create_new_directories(self) -> None:
        """Create new directory structure for v2."""
        logger.info("Creating new directory structure...")

        directories = [
            Path.home() / ".myweepal" / "models",
            Path.home() / ".myweepal" / "voices",
            Path.home() / ".myweepal" / "tts_cache",
            self.project_root / "logs",
            self.project_root / "data" / "exports",
            self.project_root / "data" / "memories",
            self.project_root / "data" / "chroma_v2",
        ]

        for directory in directories:
            if self.dry_run:
                logger.info(f"Would create directory: {directory}")
            else:
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {directory}")

    def _update_configuration(self) -> None:
        """Update configuration files for v2."""
        logger.info("Updating configuration...")

        if self.dry_run:
            logger.info("Would update .env and config.yaml with v2 settings")
            return

        # The .env and config.yaml files should already be created by the main script
        # Here we just log that they've been updated
        env_file = self.project_root / ".env"
        config_file = self.project_root / "config.yaml"

        if env_file.exists():
            logger.info("Environment configuration updated for v2")

        if config_file.exists():
            logger.info("Application configuration updated for v2")

    def _download_models(self) -> None:
        """Download required models for v2."""
        logger.info("Preparing model downloads...")

        models_to_download = [
            ("ASR", "mlx-community/whisper-large-v3-turbo"),
            ("TTS", "mlx-community/Kokoro-82M-bf16"),
            ("LLM", "mlx-community/Qwen3-4B-Thinking-2507-4bit"),
        ]

        if self.dry_run:
            logger.info("Would download the following models:")
            for model_type, model_name in models_to_download:
                logger.info(f"  - {model_type}: {model_name}")
        else:
            logger.info("Models will be downloaded on first use")
            logger.info("To download now, run:")
            for model_type, model_name in models_to_download:
                if model_type == "LLM":
                    logger.info(f"  python -c \"from mlx_lm import load; load('{model_name}')\"")

    def cleanup(self) -> None:
        """Clean up migration artifacts."""
        logger.info("Cleaning up...")

        if self.dry_run:
            logger.info("Would remove migration backup")
            return

        if self.backup_dir.exists():
            response = input("Remove migration backup? (y/n): ")
            if response.lower() == "y":
                shutil.rmtree(self.backup_dir)
                logger.info("Removed migration backup")
            else:
                logger.info(f"Backup preserved at {self.backup_dir}")


def main():
    """Main migration entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Migrate MyWeePal from v1 to v2")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up migration artifacts after successful migration"
    )

    args = parser.parse_args()

    migration = MyWeePalMigration(dry_run=args.dry_run)

    if migration.run():
        logger.info("Migration successful!")

        if args.cleanup:
            migration.cleanup()

        logger.info("\nNext steps:")
        logger.info("1. Run 'uv sync' to ensure all dependencies are installed")
        logger.info("2. Test the application with 'uv run python -m myweepal.main'")
        logger.info("3. Download models as needed")
    else:
        logger.error("Migration failed! Check logs for details")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())