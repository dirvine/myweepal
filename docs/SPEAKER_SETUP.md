# Speaker Recognition Setup Guide

This guide will help you set up speaker recognition and enroll users in MyWeePal 2.0.

## Prerequisites

1. **Audio Recording**: You need a 10-30 second audio sample for each user
2. **Quiet Environment**: Record in a quiet space for best results
3. **Natural Speech**: Speak naturally, as you would during normal conversation

## Step 1: Record Voice Samples

### Option A: Using Built-in Recording Script

```bash
# Record a voice sample directly
uv run python record_voice.py --duration 15 --output david_voice.wav

# The script will guide you through the recording process
```

### Option B: Using External Recording

You can use any audio recording app:
- QuickTime Player (Mac)
- Voice Memos (iPhone/Mac)
- Audacity (Free, cross-platform)

**Recording Guidelines:**
- Duration: 10-30 seconds
- Format: WAV or MP3
- Sample Rate: 16kHz or higher
- Content: Natural speech (introduce yourself, read a paragraph, or speak freely)

## Step 2: Enroll Speakers

### Using the Enrollment Script

```bash
# Create the enrollment script
cat > enroll_speaker.py << 'EOF'
#!/usr/bin/env python
"""Speaker enrollment script for MyWeePal."""

import sys
import argparse
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from myweepal.audio.speaker_recognition import SpeakerRecognitionSystem, UserRole
from myweepal.core.personalized_memory import PersonalizedMemorySystem


def main():
    parser = argparse.ArgumentParser(description="Enroll a speaker for MyWeePal")
    parser.add_argument("--audio", required=True, help="Path to audio file (10-30 seconds)")
    parser.add_argument("--name", required=True, help="Speaker's name")
    parser.add_argument("--role", choices=["prime", "family", "guest"],
                       default="family", help="User role")

    args = parser.parse_args()

    # Check audio file exists
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return 1

    # Initialize systems
    print("Initializing speaker recognition system...")
    speaker_system = SpeakerRecognitionSystem()
    memory_system = PersonalizedMemorySystem()

    # Map role to enum
    role_map = {
        "prime": UserRole.PRIME,
        "family": UserRole.FAMILY,
        "guest": UserRole.GUEST
    }
    user_role = role_map[args.role]

    # Enroll speaker
    print(f"Enrolling {args.name} as {args.role}...")
    success, result = speaker_system.enroll_speaker(
        str(audio_path),
        args.name,
        user_role
    )

    if success:
        print(f"✅ Successfully enrolled {args.name}!")
        print(f"   User ID: {result}")

        # Create memory profile
        memory_system.create_user_profile(
            user_id=result,
            name=args.name,
            preferences={"role": args.role}
        )
        print(f"✅ Created personalized memory profile")

        # Save the user ID for reference
        users_file = Path("enrolled_users.txt")
        with open(users_file, "a") as f:
            f.write(f"{result},{args.name},{args.role}\n")

        print(f"\nNext steps:")
        print(f"1. Test recognition: uv run python test_speaker.py --user-id {result}")
        print(f"2. Start session: uv run python -m myweepal.main start --user {args.name}")
    else:
        print(f"❌ Enrollment failed: {result}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
EOF

# Make it executable
chmod +x enroll_speaker.py
```

### Enroll Users

```bash
# Enroll the primary user (David)
uv run python enroll_speaker.py --audio david_voice.wav --name "David" --role prime

# Enroll family members
uv run python enroll_speaker.py --audio alice_voice.wav --name "Alice" --role family
uv run python enroll_speaker.py --audio bob_voice.wav --name "Bob" --role family

# Enroll a guest
uv run python enroll_speaker.py --audio guest_voice.wav --name "Guest User" --role guest
```

## Step 3: Test Speaker Recognition

### Create Test Script

```bash
# Create the test script
cat > test_speaker.py << 'EOF'
#!/usr/bin/env python
"""Test speaker recognition."""

import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from myweepal.audio.speaker_recognition import SpeakerRecognitionSystem
import sounddevice as sd


def main():
    parser = argparse.ArgumentParser(description="Test speaker recognition")
    parser.add_argument("--audio", help="Audio file to test")
    parser.add_argument("--record", type=int, help="Record N seconds of audio")
    parser.add_argument("--list", action="store_true", help="List enrolled speakers")

    args = parser.parse_args()

    # Initialize system
    speaker_system = SpeakerRecognitionSystem()

    if args.list:
        # List all enrolled speakers
        print("\nEnrolled Speakers:")
        print("-" * 50)
        for user_id, profile in speaker_system.profiles.items():
            print(f"Name: {profile.name}")
            print(f"  ID: {user_id}")
            print(f"  Role: {profile.role.value}")
            print(f"  Enrolled: {profile.created_at}")
            print()
        return 0

    # Get audio for testing
    if args.audio:
        # Load from file
        import librosa
        audio, sr = librosa.load(args.audio, sr=16000)
    elif args.record:
        # Record new audio
        print(f"Recording for {args.record} seconds...")
        audio = sd.rec(int(args.record * 16000), samplerate=16000, channels=1, dtype=np.float32)
        sd.wait()
        audio = audio.flatten()
        print("Recording complete!")
    else:
        print("Please specify --audio or --record")
        return 1

    # Test identification
    print("\nAnalyzing speaker...")
    user_id, confidence, scores = speaker_system.identify_speaker(audio, return_all_scores=True)

    if user_id:
        speaker_info = speaker_system.get_speaker_info(user_id)
        print(f"✅ Identified: {speaker_info['name']}")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   Role: {speaker_info['role']}")
    else:
        print("❌ Speaker not recognized")
        if scores:
            print("\nClosest matches:")
            for uid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]:
                info = speaker_system.get_speaker_info(uid)
                print(f"  - {info['name']}: {score:.2%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x test_speaker.py
```

### Test Recognition

```bash
# List enrolled speakers
uv run python test_speaker.py --list

# Test with an audio file
uv run python test_speaker.py --audio test_voice.wav

# Test with live recording
uv run python test_speaker.py --record 5
```

## Step 4: Manage Speakers

### Create Management Script

```bash
# Create speaker management script
cat > speaker_admin.py << 'EOF'
#!/usr/bin/env python
"""Speaker administration tool."""

import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from myweepal.core.dialogue import DialogueManager


def list_speakers(dm: DialogueManager):
    """List all enrolled speakers."""
    speakers = dm.list_enrolled_speakers()

    if not speakers:
        print("No speakers enrolled yet.")
        return

    print("\n" + "=" * 70)
    print("ENROLLED SPEAKERS")
    print("=" * 70)

    for speaker in speakers:
        print(f"\n👤 {speaker['name']}")
        print(f"   ID: {speaker['user_id']}")
        print(f"   Role: {speaker['role'].upper()}")
        print(f"   Enrolled: {speaker['enrolled_at']}")
        if speaker['last_seen']:
            print(f"   Last Seen: {speaker['last_seen']}")


def remove_speaker(dm: DialogueManager, user_id: str):
    """Remove a speaker."""
    print(f"⚠️  This will remove the speaker and ALL their memories!")
    confirm = input("Are you sure? (yes/no): ")

    if confirm.lower() == "yes":
        if dm.remove_speaker(user_id):
            print("✅ Speaker removed successfully")
        else:
            print("❌ Failed to remove speaker")
    else:
        print("Cancelled")


def main():
    parser = argparse.ArgumentParser(description="Speaker administration")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List command
    subparsers.add_parser("list", help="List all enrolled speakers")

    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove a speaker")
    remove_parser.add_argument("user_id", help="User ID to remove")

    # Stats command
    subparsers.add_parser("stats", help="Show speaker statistics")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Initialize dialogue manager
    dm = DialogueManager(enable_speaker_recognition=True)

    if args.command == "list":
        list_speakers(dm)
    elif args.command == "remove":
        remove_speaker(dm, args.user_id)
    elif args.command == "stats":
        speakers = dm.list_enrolled_speakers()
        print(f"\nTotal enrolled speakers: {len(speakers)}")
        roles = {}
        for s in speakers:
            role = s['role']
            roles[role] = roles.get(role, 0) + 1
        for role, count in roles.items():
            print(f"  {role.capitalize()}: {count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x speaker_admin.py
```

### Use Management Commands

```bash
# List all speakers
uv run python speaker_admin.py list

# Show statistics
uv run python speaker_admin.py stats

# Remove a speaker (use with caution!)
uv run python speaker_admin.py remove USER_ID
```

## Step 5: Start Using MyWeePal

### With Speaker Recognition

```bash
# Start with automatic speaker recognition
uv run python -m myweepal.main start --speaker-recognition

# The system will automatically identify who's speaking
# and store memories in their personal collection
```

### With Specific User

```bash
# Start session for a specific user
uv run python -m myweepal.main start --user "David"
```

## Troubleshooting

### Poor Recognition Accuracy

1. **Re-enroll with better audio**: Record in a quieter environment
2. **Use longer samples**: 20-30 seconds works better than 10 seconds
3. **Multiple enrollments**: Enroll the same person multiple times to improve accuracy

### Speaker Not Recognized

1. Check if the speaker is enrolled: `uv run python speaker_admin.py list`
2. Test with the enrollment audio: `uv run python test_speaker.py --audio original_enrollment.wav`
3. Adjust the identification threshold in `.env`:
   ```env
   IDENTIFICATION_THRESHOLD=0.7  # Lower = more permissive
   ```

### Audio Issues

1. **Check microphone**: `uv run python -c "import sounddevice; print(sounddevice.query_devices())"`
2. **Test recording**: `uv run python test_speaker.py --record 5`
3. **Verify audio format**: WAV files work best, 16kHz sample rate

## Privacy Notes

- All voice profiles are stored locally in `~/.myweepal/speaker_profiles/`
- Each user's memories are stored in separate ChromaDB collections
- Voice samples are not stored after enrollment (only embeddings)
- Users can be removed completely with `speaker_admin.py remove`

## Advanced Configuration

### Adjust Recognition Parameters

Edit `.env` file:

```env
# Speaker Recognition Tuning
MIN_ENROLLMENT_DURATION=10.0    # Minimum seconds for enrollment
MAX_ENROLLMENT_DURATION=30.0    # Maximum seconds for enrollment
IDENTIFICATION_THRESHOLD=0.85   # Confidence threshold (0-1)
EMBEDDING_SIZE=192             # Size of voice embeddings
```

### Enable Advanced Models

For better accuracy (requires additional dependencies):

```bash
# Install advanced speaker recognition
uv pip install speechbrain pyannote-audio

# These will be used automatically if available
```

## Next Steps

1. **Enroll all family members** who will use the system
2. **Test recognition** with each user
3. **Configure privacy settings** for memory sharing
4. **Start capturing life stories** with personalized memory!

For more help, see the main [README](../README.md) or open an issue on GitHub.