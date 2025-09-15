# MyWeePal 🤖

**Local Conversational Life-Story AI for Apple Silicon**

MyWeePal is a privacy-focused, locally-running AI assistant that conducts empathetic interviews to capture and preserve your life story. Built specifically for Apple Silicon Macs using MLX framework for optimal performance.

## 🌟 Features

- **🧠 Local LLM Inference**: Runs entirely on your Mac using MLX framework (no cloud dependencies)
- **🎙️ Speech Recognition**: Real-time speech-to-text using Whisper
- **🗣️ Text-to-Speech**: Natural voice synthesis with voice cloning capabilities
- **😊 Emotion Detection**: Real-time facial emotion analysis via webcam
- **💾 Memory Storage**: Vector database for intelligent memory retrieval
- **🔒 Privacy-First**: All data stays on your device
- **⚡ Apple Silicon Optimized**: Leverages Metal/Neural Engine acceleration

## 📋 Requirements

### Hardware
- **Mac**: Apple Silicon (M1/M2/M3) with minimum 32GB RAM (64GB recommended)
- **Storage**: 10GB+ free space for models and data
- **Camera**: Built-in or external webcam for emotion detection
- **Microphone**: For speech input

### Software
- macOS 13.0+ (Ventura or newer)
- Python 3.10+
- Xcode Command Line Tools

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/dirvine/myweepal.git
cd myweepal
```

### 2. Install Poetry (if not already installed)
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 3. Install Dependencies
```bash
poetry install
```

### 4. Set Up Environment
```bash
cp .env.example .env
# Edit .env with your preferred settings
```

### 5. Download MLX Model
```bash
poetry run python -c "from mlx_lm import load; load('mlx-community/Llama-3.2-3B-Instruct-4bit')"
```

## 🎯 Quick Start

### Start an Interactive Session
```bash
poetry run python -m myweepal.main start
```

### With Video Display
```bash
poetry run python -m myweepal.main start --video
```

### Test Individual Components
```bash
# Test LLM
poetry run python -m myweepal.main test llm

# Test Speech Recognition
poetry run python -m myweepal.main test asr

# Test Text-to-Speech
poetry run python -m myweepal.main test tts

# Test Emotion Detection
poetry run python -m myweepal.main test emotion

# Test Memory Storage
poetry run python -m myweepal.main test memory
```

## 📖 Usage Guide

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

### CLI Commands

```bash
# Start interactive session
poetry run python -m myweepal.main start [OPTIONS]

Options:
  --model, -m TEXT        MLX model name
  --memory-dir, -d TEXT   Memory storage directory
  --video, -v             Enable video display

# Export memories
poetry run python -m myweepal.main export OUTPUT_FILE [OPTIONS]

# Analyze emotions
poetry run python -m myweepal.main analyze [OPTIONS]

# Calibrate emotion detection
poetry run python -m myweepal.main calibrate [OPTIONS]
```

## 🏗️ Architecture

```
MyWeePal/
├── Core Modules
│   ├── LLM Inference (MLX-based)
│   ├── Dialogue Manager
│   └── Memory Store (ChromaDB)
├── Vision Module
│   └── Emotion Detection (FER/OpenCV)
├── Audio Modules
│   ├── Speech Recognition (Whisper)
│   └── Text-to-Speech (TTS)
└── Main Application
    └── CLI Interface (Typer/Rich)
```

## 🔧 Configuration

Edit `.env` file to customize:

```env
# MLX Model Settings
MLX_MODEL_NAME=mlx-community/Llama-3.2-3B-Instruct-4bit
MLX_MAX_TOKENS=256
MLX_TEMPERATURE=0.7

# Memory Database
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_COLLECTION_NAME=myweepal_memories

# Audio Settings
AUDIO_SAMPLE_RATE=16000
AUDIO_CHANNELS=1

# Vision Settings
CAMERA_INDEX=0
EMOTION_DETECTION_INTERVAL=5
```

## 🧪 Development

### Run Tests
```bash
poetry run pytest
```

### Run with Coverage
```bash
poetry run pytest --cov=myweepal --cov-report=html
```

### Code Quality
```bash
# Format code
poetry run black myweepal tests

# Lint
poetry run ruff check myweepal tests

# Type check
poetry run mypy myweepal
```

## 📊 Data Export

Export your memories for backup or analysis:

```bash
poetry run python -m myweepal.main export memories.json
```

The exported JSON includes:
- All Q&A pairs
- Timestamps
- Emotional states
- Categories
- Metadata

## 🔐 Privacy & Security

- **100% Local**: No data leaves your device
- **Encrypted Storage**: ChromaDB with local persistence
- **No Cloud Dependencies**: Works offline
- **User Control**: Export or delete your data anytime

## 🚧 Roadmap

- [ ] Multi-language support
- [ ] Advanced voice cloning
- [ ] Web interface
- [ ] Mobile companion app
- [ ] Edge deployment (Jetson)
- [ ] Family sharing features
- [ ] Backup & sync options

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit PRs.

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Apple MLX team for the framework
- OpenAI Whisper for speech recognition
- ChromaDB for vector storage
- The open-source community

## 📞 Support

- Issues: [GitHub Issues](https://github.com/dirvine/myweepal/issues)
- Discussions: [GitHub Discussions](https://github.com/dirvine/myweepal/discussions)

---

**Built with ❤️ in Scotland by David Irvine**