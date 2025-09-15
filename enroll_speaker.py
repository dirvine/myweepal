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