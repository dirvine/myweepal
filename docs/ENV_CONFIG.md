# Environment Configuration Guide

This guide explains all environment variables available in MyWeePal 2.0.

## Quick Setup

1. Create a `.env` file in the project root:
```bash
touch .env
```

2. Copy the example configuration below and customize as needed.

## Complete .env Configuration

```env
# =====================================
# Model Settings
# =====================================
MODEL_NAME=mlx-community/Qwen3-4B-Thinking-2507-4bit
MAX_TOKENS=512
CONTEXT_WINDOW=8192
TEMPERATURE=0.7
TOP_P=0.9

# =====================================
# Speaker Recognition
# =====================================
ENABLE_SPEAKER_RECOGNITION=true
MIN_ENROLLMENT_DURATION=10.0       # Minimum seconds for voice enrollment
MAX_ENROLLMENT_DURATION=30.0       # Maximum seconds for voice enrollment
IDENTIFICATION_THRESHOLD=0.85      # Confidence threshold (0-1, lower = more permissive)
SPEAKER_EMBEDDING_SIZE=192         # Size of voice embeddings
SPEAKER_MODEL=speechbrain/spkrec-ecapa-voxceleb

# =====================================
# Audio Settings
# =====================================
AUDIO_SAMPLE_RATE=16000
AUDIO_CHANNELS=1
CHUNK_DURATION=2.0                 # Audio chunk size for processing
OVERLAP_DURATION=0.5               # Overlap between chunks for context
ENERGY_THRESHOLD=0.01              # Voice activity detection threshold

# =====================================
# Text-to-Speech (TTS)
# =====================================
TTS_MODEL=kokoro-82m-bf16
TTS_FALLBACK_MODEL=tacotron2-ddc
TTS_SAMPLE_RATE=24000
ENABLE_VOICE_CLONING=true
ENABLE_EMOTION_TTS=true

# =====================================
# Speech Recognition (ASR)
# =====================================
ASR_MODEL=parakeet-tdt-0.6b-v2
ASR_FALLBACK_MODEL=whisper-large-v3-turbo
ENABLE_STREAMING_ASR=true

# =====================================
# Memory Database
# =====================================
CHROMA_PERSIST_DIR=./data/chroma
PERSONALIZED_MEMORY_DIR=./data/personalized_memory
MAX_MEMORIES_PER_USER=10000
RETENTION_DAYS=365
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# =====================================
# Privacy Settings
# =====================================
DEFAULT_PRIVACY_LEVEL=private      # Options: private, family, public, ephemeral
EPHEMERAL_TTL_HOURS=24            # Auto-delete time for ephemeral memories
FAMILY_SHARING_ENABLED=true        # Allow family members to share memories

# =====================================
# Vision Settings
# =====================================
CAMERA_INDEX=0
EMOTION_DETECTION_INTERVAL=5.0
EMOTION_SMOOTHING_WINDOW=3
ENABLE_EMOTION_DETECTION=true

# =====================================
# System Paths
# =====================================
DATA_DIR=./data
LOGS_DIR=./logs
MODELS_DIR=./models
SESSIONS_DIR=./data/sessions
SPEAKER_PROFILES_DIR=./data/speaker_profiles

# =====================================
# Debug Settings
# =====================================
ENABLE_DEBUG_MODE=false
LOG_LEVEL=INFO                     # DEBUG, INFO, WARNING, ERROR
LOG_TO_FILE=true
```

## Configuration Details

### Model Settings

- **MODEL_NAME**: The MLX model to use for language generation
  - Default: `mlx-community/Qwen3-4B-Thinking-2507-4bit`
  - Options: Any MLX-compatible model

- **MAX_TOKENS**: Maximum tokens to generate per response
  - Default: `512`
  - Range: 128-2048

- **CONTEXT_WINDOW**: Size of the context window
  - Default: `8192`
  - Adjust based on model capabilities

- **TEMPERATURE**: Controls randomness in generation
  - Default: `0.7`
  - Range: 0.0 (deterministic) to 1.0 (creative)

### Speaker Recognition

- **ENABLE_SPEAKER_RECOGNITION**: Enable/disable speaker identification
  - Default: `true`
  - Set to `false` for single-user mode

- **MIN_ENROLLMENT_DURATION**: Minimum audio length for enrollment
  - Default: `10.0` seconds
  - Lower values may reduce accuracy

- **IDENTIFICATION_THRESHOLD**: Confidence threshold for speaker identification
  - Default: `0.85`
  - Lower values = more permissive (more false positives)
  - Higher values = more strict (more false negatives)

### Audio Settings

- **CHUNK_DURATION**: Length of audio chunks for processing
  - Default: `2.0` seconds
  - Shorter = more responsive but less context
  - Longer = better context but higher latency

- **ENERGY_THRESHOLD**: Voice activity detection sensitivity
  - Default: `0.01`
  - Lower = more sensitive (picks up quieter sounds)
  - Higher = less sensitive (requires louder speech)

### Privacy Settings

- **DEFAULT_PRIVACY_LEVEL**: Default privacy for new memories
  - `private`: Only accessible by the user
  - `family`: Shared with family members
  - `public`: Accessible by all users
  - `ephemeral`: Auto-deleted after TTL

- **EPHEMERAL_TTL_HOURS**: Hours before ephemeral memories are deleted
  - Default: `24`
  - Range: 1-168 (1 hour to 1 week)

### Memory Settings

- **MAX_MEMORIES_PER_USER**: Maximum memories per user
  - Default: `10000`
  - Older memories are archived when limit is reached

- **RETENTION_DAYS**: Days to keep memories
  - Default: `365`
  - Set to `-1` for indefinite retention

## Environment-Specific Configurations

### Development
```env
ENABLE_DEBUG_MODE=true
LOG_LEVEL=DEBUG
IDENTIFICATION_THRESHOLD=0.7  # More permissive for testing
```

### Production
```env
ENABLE_DEBUG_MODE=false
LOG_LEVEL=ERROR
IDENTIFICATION_THRESHOLD=0.9  # Stricter for accuracy
LOG_TO_FILE=true
```

### Testing
```env
MIN_ENROLLMENT_DURATION=5.0   # Shorter for quick tests
MAX_MEMORIES_PER_USER=100     # Smaller for testing
RETENTION_DAYS=7               # Shorter retention
```

## Model Selection Guide

### LLM Models (MLX)
- **Qwen3-4B-Thinking**: Best for reasoning and conversation
- **Llama-3.2-3B**: Good general purpose
- **Phi-3-mini**: Lightweight, fast responses

### ASR Models
- **parakeet-tdt-0.6b-v2**: Optimized for real-time, good with accents
- **whisper-large-v3-turbo**: Most accurate, higher latency
- **whisper-base**: Lightweight, good for testing

### TTS Models
- **kokoro-82m-bf16**: High quality, emotion support
- **tacotron2-ddc**: Good quality, stable
- **chatterbox**: Voice cloning support

## Troubleshooting

### Poor Speaker Recognition
```env
# Adjust these settings:
IDENTIFICATION_THRESHOLD=0.7      # Lower threshold
MIN_ENROLLMENT_DURATION=15.0      # Longer enrollment
SPEAKER_EMBEDDING_SIZE=256        # Larger embeddings
```

### High Memory Usage
```env
# Reduce these settings:
MAX_MEMORIES_PER_USER=5000
CONTEXT_WINDOW=4096
MAX_TOKENS=256
```

### Slow Response Times
```env
# Optimize for speed:
CHUNK_DURATION=1.0
OVERLAP_DURATION=0.25
MODEL_NAME=mlx-community/Phi-3-mini-4bit
```

## Security Notes

- Never commit `.env` files to version control
- Keep sensitive settings (API keys if added) in `.env.local`
- Use environment-specific files: `.env.development`, `.env.production`
- Regularly rotate any authentication tokens

## Default Values

If a variable is not set in `.env`, these defaults are used:

| Variable | Default Value |
|----------|--------------|
| MODEL_NAME | mlx-community/Qwen3-4B-Thinking-2507-4bit |
| MAX_TOKENS | 512 |
| TEMPERATURE | 0.7 |
| ENABLE_SPEAKER_RECOGNITION | true |
| IDENTIFICATION_THRESHOLD | 0.85 |
| AUDIO_SAMPLE_RATE | 16000 |
| DEFAULT_PRIVACY_LEVEL | private |
| MAX_MEMORIES_PER_USER | 10000 |
| LOG_LEVEL | INFO |