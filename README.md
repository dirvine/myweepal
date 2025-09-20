# MyWeePal 2.0 🤖

## CURRENTLY EXPERIMENTAL - NOT USABLE FOR PRODUCTION YET !!!

**Local Conversational Life-Story AI with Multi-User Support for Apple Silicon**

MyWeePal is a privacy-focused, locally-running AI assistant that conducts empathetic interviews to capture and preserve life stories. Now with **speaker recognition** and **personalized memories** for multiple users. Built specifically for Apple Silicon Macs using MLX framework for optimal performance.

## 🆕 Version 2.0 Features

### Speaker Recognition & Multi-User Support
- **🎤 Voice Enrollment**: 10-30 second voice samples for user registration
- **👥 Multi-User Profiles**: Support for Prime user (David), Family members, and Guests
- **🔊 Real-Time Speaker Identification**: Identifies who's speaking in < 2 seconds
- **🔐 Voice Authentication**: Biometric voice verification
- **💾 Personalized Memory**: Individual memory banks for each user
- **🔒 Privacy Controls**: Private, Family-shared, or Public memories

## 🌟 Core Features

- **🧠 Local LLM Inference**: Qwen3-4B-Thinking model optimized with MLX (no cloud dependencies)
- **🎙️ Streaming Speech Recognition**: Real-time ASR with Parakeet/Whisper fallback
- **🗣️ Emotion-Aware TTS**: Kokoro/Chatterbox with 10 emotion types & voice cloning
- **😊 Emotion Detection**: Real-time facial emotion analysis via webcam
- **💾 ChromaDB Storage**: Per-user vector databases for intelligent memory retrieval
- **🔒 Privacy-First**: All data stays on your device
- **⚡ Apple Silicon Optimized**: Leverages Metal/Neural Engine acceleration

## 📋 Requirements

### Hardware
- **Mac**: Apple Silicon (M1/M2/M3/M4) with minimum 16GB RAM (32GB+ recommended)
- **Storage**: 20GB+ free space for models and data
- **Camera**: Built-in or external webcam for emotion detection
- **Microphone**: For speech input and speaker recognition

### Software
- macOS 13.0+ (Ventura or newer)
- Python 3.11+
- Xcode Command Line Tools

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/dirvine/myweepal.git
cd myweepal
```

### 2. Install UV Package Manager
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Install Dependencies
```bash
# Install core dependencies
uv sync

# Optional: Install speaker recognition dependencies
uv pip install speechbrain pyannote-audio

# Optional: Install other features
uv pip install fer openai-whisper TTS
```

### 4. Set Up Environment
```bash
cp .env.example .env
# Edit .env with your preferred settings
```

### 5. Download Models (Optional - will auto-download on first use)
```bash
# LLM Model
uv run python -c "from mlx_lm import load; load('mlx-community/Qwen3-4B-Thinking-2507-4bit')"
```

## 🎯 Quick Start

### Basic Usage

```bash
# Run the main application
uv run python -m myweepal.main

# Run basic tests to verify installation
uv run python test_basic.py
```

### Speaker Enrollment (First Time Users)

```bash
# Enroll a new speaker (10-30 second audio sample)
uv run python enroll_speaker.py --name "David" --role prime

# Enroll family members
uv run python enroll_speaker.py --name "Alice" --role family
```

### Start an Interactive Session

```bash
# Start with speaker recognition enabled
uv run python -m myweepal.main start --speaker-recognition

# Start with specific user
uv run python -m myweepal.main start --user "David"

# Start with video emotion detection
uv run python -m myweepal.main start --video --speaker-recognition
```

## 👥 Multi-User Features

### User Roles

1. **Prime User** (David)
   - Full access to all features
   - Can manage other users
   - Primary memory bank owner

2. **Family Members**
   - Personal memory banks
   - Can share memories with family
   - Access to family-shared memories

3. **Guests**
   - Temporary memory storage
   - Limited access period
   - Privacy-protected interactions

### Privacy Levels

- **Private**: Only accessible by the user
- **Family**: Shared with family members
- **Public**: Accessible by all users
- **Ephemeral**: Auto-delete after specified time

## 📖 Usage Guide

### Speaker Enrollment Process

```python
# Example enrollment script
from myweepal.audio.speaker_recognition import SpeakerRecognitionSystem, UserRole

# Initialize system
speaker_system = SpeakerRecognitionSystem()

# Enroll new speaker
success, user_id = speaker_system.enroll_speaker(
    audio_path="voice_samples/david.wav",
    name="David",
    role=UserRole.PRIME
)

if success:
    print(f"Enrolled successfully! User ID: {user_id}")
```

### Interactive Interview Session

MyWeePal conducts structured interviews through different life phases:

1. **Introduction** - Getting to know you
2. **Childhood** - Early memories and experiences
3. **Education** - Learning and growth
4. **Career** - Professional journey
5. **Relationships** - Important connections
6. **Achievements** - Accomplishments and milestones
7. **Challenges** - Overcoming difficulties
8. **Aspirations** - Dreams and goals
9. **Reflection** - Looking back and forward

Each user's responses are stored in their personal memory bank.

### CLI Commands

```bash
# Core Commands
uv run python -m myweepal.main start [OPTIONS]
  --speaker-recognition    Enable speaker recognition
  --user TEXT             Start with specific user
  --video                 Enable emotion detection
  --model TEXT            MLX model name
  --memory-dir TEXT       Memory storage directory

# Speaker Management
uv run python speaker_admin.py
  list                    List all enrolled speakers
  enroll NAME             Enroll new speaker
  remove USER_ID          Remove speaker profile
  test USER_ID            Test speaker recognition

# Memory Management
uv run python memory_admin.py
  export USER_ID          Export user memories
  search USER_ID QUERY    Search user memories
  stats USER_ID           Show memory statistics
  clear USER_ID           Clear user memories (requires confirmation)

# Testing
uv run python test_basic.py              # Basic integration tests
uv run python tests/test_speaker_recognition.py  # Speaker tests
```

## 🏗️ Architecture

```
MyWeePal 2.0/
├── Core Modules
│   ├── LLM Inference (Qwen3-4B via MLX)
│   ├── Dialogue Manager (Multi-user aware)
│   ├── Memory Store (Legacy ChromaDB)
│   └── Personalized Memory (Per-user ChromaDB)
├── Vision Module
│   └── Emotion Detection (FER/OpenCV)
├── Audio Modules
│   ├── Speech Recognition (Whisper/Parakeet)
│   ├── Streaming ASR (Real-time)
│   ├── Text-to-Speech (Kokoro/Chatterbox)
│   ├── Emotion-aware TTS
│   └── Speaker Recognition (ECAPA-TDNN)
└── Main Application
    └── CLI Interface (Typer/Rich)
```

## 🔧 Configuration

### Environment Variables (.env)

```env
# Model Settings
MODEL_NAME=mlx-community/Qwen3-4B-Thinking-2507-4bit
MAX_TOKENS=512
CONTEXT_WINDOW=8192
TEMPERATURE=0.7

# Speaker Recognition
ENABLE_SPEAKER_RECOGNITION=true
MIN_ENROLLMENT_DURATION=10.0
MAX_ENROLLMENT_DURATION=30.0
IDENTIFICATION_THRESHOLD=0.85

# Memory Database
CHROMA_PERSIST_DIR=./data/chroma
PERSONALIZED_MEMORY_DIR=./data/personalized_memory
MAX_MEMORIES_PER_USER=10000
RETENTION_DAYS=365

# Audio Settings
AUDIO_SAMPLE_RATE=16000
AUDIO_CHANNELS=1
CHUNK_DURATION=2.0
OVERLAP_DURATION=0.5

# TTS Settings
TTS_MODEL=kokoro-82m-bf16
TTS_SAMPLE_RATE=24000
ENABLE_VOICE_CLONING=true

# Vision Settings
CAMERA_INDEX=0
EMOTION_DETECTION_INTERVAL=5
```

### Configuration File (config.yaml)

```yaml
app:
  name: MyWeePal
  version: 2.0.0
  debug: false

models:
  llm:
    name: mlx-community/Qwen3-4B-Thinking-2507-4bit
    max_tokens: 512
    context_window: 8192

  asr:
    primary: parakeet-tdt-0.6b-v2
    fallback: whisper-large-v3-turbo

  tts:
    primary: kokoro-82m-bf16
    fallback: tacotron2-ddc

speaker_recognition:
  enabled: true
  model: speechbrain/spkrec-ecapa-voxceleb
  embedding_size: 192

users:
  prime:
    name: David
    role: prime
    preferences:
      language: en
      accent: scottish
```

## 🧪 Testing

### Run All Tests
```bash
# Run basic integration tests
uv run python test_basic.py

# Run speaker recognition tests
uv run python tests/test_speaker_recognition.py

# Run with pytest (if installed)
uv run pytest tests/

# Run with coverage
uv run pytest --cov=myweepal --cov-report=html
```

### Test Individual Components
```bash
# Test speaker enrollment
uv run python -c "from myweepal.audio.speaker_recognition import SpeakerRecognitionSystem; s = SpeakerRecognitionSystem(); print('Speaker system initialized')"

# Test personalized memory
uv run python -c "from myweepal.core.personalized_memory import PersonalizedMemorySystem; m = PersonalizedMemorySystem(); print('Memory system initialized')"
```

## 📊 Data Management

### Export User Memories
```bash
# Export specific user's memories
uv run python export_memories.py --user-id david_001 --output david_memories.json

# Export all users
uv run python export_memories.py --all --output all_memories.json
```

### Backup and Restore
```bash
# Backup entire system
tar -czf myweepal_backup.tar.gz data/

# Restore from backup
tar -xzf myweepal_backup.tar.gz
```

## 🔐 Privacy & Security

- **100% Local**: No data leaves your device
- **Per-User Isolation**: Each user has separate memory storage
- **Voice Biometrics**: Speaker verification for authentication
- **Encrypted Storage**: ChromaDB with local persistence
- **Privacy Controls**: Choose who can access your memories
- **No Cloud Dependencies**: Works completely offline
- **Data Ownership**: Export or delete your data anytime

## 🚀 Advanced Features

### Voice Cloning
```python
# Clone a voice for personalized TTS
from myweepal.audio.emotion_tts import EmotionAwareTTS

tts = EmotionAwareTTS()
tts.clone_voice(
    audio_path="voice_samples/david.wav",
    profile_name="david_voice"
)
```

### Emotion-Aware Responses
```python
# Generate responses with emotional context
tts.synthesize_with_emotion(
    text="I understand how you feel",
    emotion="empathetic",
    intensity=0.8
)
```

### Multi-Speaker Conversations
```python
# Process conversations with multiple speakers
segments = speaker_system.process_conversation(audio_segments)
for segment in segments:
    print(f"{segment.name}: {segment.text}")
```

## 🚧 Roadmap

- [x] Speaker recognition and enrollment
- [x] Multi-user personalized memory
- [x] Privacy controls for memories
- [x] Voice authentication
- [ ] Web interface with user dashboard
- [ ] Mobile companion app
- [ ] Family tree visualization
- [ ] Memory timeline view
- [ ] Voice cloning improvements
- [ ] Multi-language support
- [ ] Cloud backup (encrypted, optional)

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit PRs.

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Apple MLX team for the framework
- Qwen team for the thinking model
- SpeechBrain for speaker recognition
- ChromaDB for vector storage
- The open-source community

## 📞 Support

- **Author**: David Irvine
- **Location**: Barr, Scotland
- **Issues**: [GitHub Issues](https://github.com/dirvine/myweepal/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dirvine/myweepal/discussions)

---

**Built with ❤️ in Scotland by David Irvine**

## Version History

### v2.0.0 (Current)
- Added speaker recognition with voice enrollment
- Implemented multi-user support with personalized memory
- Upgraded to Qwen3-4B-Thinking model
- Added streaming ASR and emotion-aware TTS
- Migrated from Poetry to UV package manager
- Added privacy controls and voice authentication

### v1.0.0
- Initial release with basic interview functionality
- Single-user memory storage
- Basic emotion detection
