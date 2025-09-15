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