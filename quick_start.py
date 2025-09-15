#!/usr/bin/env python
"""Quick start script for MyWeePal 2.0 - Interactive setup and demo."""

import sys
import os
from pathlib import Path
import time

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


def print_banner():
    """Print welcome banner."""
    print("\n" + "=" * 60)
    print("   MyWeePal 2.0 - Quick Start Guide")
    print("   Local Life-Story AI with Speaker Recognition")
    print("=" * 60 + "\n")


def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")

    issues = []

    # Check core dependencies
    try:
        import mlx
        print("  ✅ MLX framework")
    except ImportError:
        issues.append("MLX not found - run: uv pip install mlx")

    try:
        import chromadb
        print("  ✅ ChromaDB")
    except ImportError:
        issues.append("ChromaDB not found - run: uv pip install chromadb")

    try:
        import sounddevice
        print("  ✅ Audio support")
    except ImportError:
        issues.append("sounddevice not found - run: uv pip install sounddevice")

    # Check optional dependencies
    try:
        import whisper
        print("  ✅ Whisper ASR (optional)")
    except ImportError:
        print("  ⚠️  Whisper not installed (optional)")

    try:
        import speechbrain
        print("  ✅ SpeechBrain (optional)")
    except ImportError:
        print("  ⚠️  SpeechBrain not installed (optional)")

    if issues:
        print("\n❌ Missing dependencies:")
        for issue in issues:
            print(f"   - {issue}")
        return False

    print("\n✅ Core dependencies OK!\n")
    return True


def setup_environment():
    """Set up environment file if not exists."""
    env_file = Path(".env")

    if not env_file.exists():
        print("📝 Creating .env file...")

        env_content = """# MyWeePal 2.0 Configuration
MODEL_NAME=mlx-community/Qwen3-4B-Thinking-2507-4bit
MAX_TOKENS=512
ENABLE_SPEAKER_RECOGNITION=true
IDENTIFICATION_THRESHOLD=0.85
DEFAULT_PRIVACY_LEVEL=private
"""

        with open(env_file, "w") as f:
            f.write(env_content)

        print("  ✅ Created .env file with defaults\n")
    else:
        print("  ✅ Using existing .env file\n")


def test_audio():
    """Test audio input/output."""
    print("🎤 Testing audio...")

    try:
        import sounddevice as sd
        import numpy as np

        # List audio devices
        devices = sd.query_devices()
        print(f"  Found {len(devices)} audio devices")

        # Test recording
        print("  Testing 1-second recording...")
        duration = 1
        fs = 16000
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
        sd.wait()

        # Test playback
        print("  Testing playback...")
        tone = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, int(fs * 0.5)))
        sd.play(tone, fs)
        sd.wait()

        print("  ✅ Audio working!\n")
        return True

    except Exception as e:
        print(f"  ❌ Audio error: {e}\n")
        return False


def demo_speaker_enrollment():
    """Demo speaker enrollment process."""
    print("👤 Speaker Enrollment Demo")
    print("-" * 40)

    print("""
To enroll a speaker, you need:
1. A 10-30 second audio recording
2. The person's name
3. Their role (prime, family, or guest)

Example command:
    uv run python enroll_speaker.py \\
        --audio voice_sample.wav \\
        --name "David" \\
        --role prime

Would you like to record a sample now? (y/n): """, end="")

    response = input().lower()

    if response == 'y':
        print("\n📝 To record a voice sample:")
        print("1. Run: uv run python record_voice.py --duration 15")
        print("2. Speak naturally for 15 seconds")
        print("3. Use the recorded file for enrollment\n")


def demo_basic_usage():
    """Show basic usage examples."""
    print("🚀 Basic Usage Examples")
    print("-" * 40)

    examples = [
        ("Test installation", "uv run python test_basic.py"),
        ("List enrolled speakers", "uv run python speaker_admin.py list"),
        ("Enroll a speaker", "uv run python enroll_speaker.py --audio sample.wav --name 'Name' --role family"),
        ("Test speaker recognition", "uv run python test_speaker.py --record 5"),
        ("Start with speaker recognition", "uv run python -m myweepal.main start --speaker-recognition"),
        ("Start for specific user", "uv run python -m myweepal.main start --user 'David'"),
    ]

    for desc, cmd in examples:
        print(f"\n{desc}:")
        print(f"  $ {cmd}")


def interactive_menu():
    """Interactive menu for quick actions."""
    print("\n📋 Quick Actions Menu")
    print("-" * 40)

    actions = [
        "Run basic tests",
        "Test speaker recognition",
        "View documentation",
        "Start MyWeePal",
        "Exit"
    ]

    for i, action in enumerate(actions, 1):
        print(f"{i}. {action}")

    print("\nSelect an option (1-5): ", end="")
    choice = input()

    if choice == "1":
        print("\n🧪 Running basic tests...")
        os.system("uv run python test_basic.py")
    elif choice == "2":
        print("\n🎤 Testing speaker recognition...")
        os.system("uv run python tests/test_speaker_recognition.py")
    elif choice == "3":
        print("\n📖 Opening documentation...")
        print("  README: ./README.md")
        print("  Speaker Setup: ./docs/SPEAKER_SETUP.md")
        print("  Environment Config: ./docs/ENV_CONFIG.md")
    elif choice == "4":
        print("\n🚀 Starting MyWeePal...")
        os.system("uv run python -m myweepal.main start --speaker-recognition")
    else:
        print("\n👋 Goodbye!")
        return False

    return True


def main():
    """Main quick start flow."""
    print_banner()

    # Check dependencies
    if not check_dependencies():
        print("\n⚠️  Please install missing dependencies first!")
        print("Run: uv sync")
        return 1

    # Set up environment
    setup_environment()

    # Test audio
    audio_ok = test_audio()
    if not audio_ok:
        print("⚠️  Audio issues detected. Check your microphone/speakers.")

    # Show demos
    demo_speaker_enrollment()
    demo_basic_usage()

    # Interactive menu
    print("\n" + "=" * 60)
    while interactive_menu():
        time.sleep(1)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        sys.exit(0)