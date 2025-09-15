#!/usr/bin/env python
"""Basic integration test for MyWeePal 2.0."""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        # Test audio modules
        from myweepal.audio.streaming import StreamingASR
        print("✓ StreamingASR imported")

        from myweepal.audio.emotion_tts import EmotionAwareTTS
        print("✓ EmotionAwareTTS imported")

        # Test core modules
        from myweepal.core.llm import LLMInference, LLMConfig
        print("✓ LLM modules imported")

        # Test configuration
        import yaml
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        print("✓ Configuration loaded")

        print("\nAll imports successful!")
        return True

    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_llm_config():
    """Test LLM configuration."""
    print("\nTesting LLM configuration...")

    try:
        from myweepal.core.llm import LLMConfig

        config = LLMConfig()
        assert config.model_name == "mlx-community/Qwen3-4B-Thinking-2507-4bit"
        assert config.max_tokens == 512
        assert config.context_window == 8192

        print("✓ LLM config uses Qwen model")
        print(f"  Model: {config.model_name}")
        print(f"  Max tokens: {config.max_tokens}")
        print(f"  Context window: {config.context_window}")

        return True

    except Exception as e:
        print(f"✗ LLM config test failed: {e}")
        return False


def test_streaming_asr():
    """Test StreamingASR initialization."""
    print("\nTesting StreamingASR...")

    try:
        from myweepal.audio.streaming import StreamingASR

        # Create instance (won't load actual model)
        asr = StreamingASR(
            model_id="test-model",
            sample_rate=16000,
            use_mlx=False
        )

        print("✓ StreamingASR initialized")
        print(f"  Sample rate: {asr.sample_rate}")
        print(f"  Chunk duration: {asr.chunk_duration}")
        print(f"  Energy threshold: {asr.energy_threshold}")

        return True

    except Exception as e:
        print(f"✗ StreamingASR test failed: {e}")
        return False


def test_emotion_tts():
    """Test EmotionAwareTTS initialization."""
    print("\nTesting EmotionAwareTTS...")

    try:
        from myweepal.audio.emotion_tts import EmotionAwareTTS

        # Create instance (won't load actual model)
        tts = EmotionAwareTTS(
            model_name="test-model",
            sample_rate=24000,
            device="cpu"
        )

        print("✓ EmotionAwareTTS initialized")
        print(f"  Sample rate: {tts.sample_rate}")
        print(f"  Available emotions: {', '.join(tts.get_emotion_list())}")

        return True

    except Exception as e:
        print(f"✗ EmotionAwareTTS test failed: {e}")
        return False


def test_migration_script():
    """Test migration script exists and is valid."""
    print("\nTesting migration script...")

    try:
        from migrate import MyWeePalMigration

        migration = MyWeePalMigration(dry_run=True)

        print("✓ Migration script loaded")
        print(f"  Project root: {migration.project_root}")
        print(f"  Dry run mode: {migration.dry_run}")

        return True

    except Exception as e:
        print(f"✗ Migration script test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("MyWeePal 2.0 Integration Tests")
    print("=" * 60)

    tests = [
        test_imports,
        test_llm_config,
        test_streaming_asr,
        test_emotion_tts,
        test_migration_script,
    ]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "=" * 60)
    print("Test Summary:")
    print(f"  Passed: {sum(results)}/{len(results)}")
    print(f"  Failed: {len(results) - sum(results)}/{len(results)}")

    if all(results):
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())