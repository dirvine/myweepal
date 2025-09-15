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