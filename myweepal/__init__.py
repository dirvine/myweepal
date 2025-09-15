"""MyWeePal - Local Conversational Life-Story AI for Apple Silicon."""

__version__ = "0.1.0"
__author__ = "David Irvine"
__email__ = "david@saorsalabs.com"

from myweepal.core.llm import LLMInference
from myweepal.core.dialogue import DialogueManager
from myweepal.core.memory import MemoryStore
from myweepal.vision.emotion import EmotionDetector
from myweepal.audio.asr import SpeechRecognizer
from myweepal.audio.tts import TextToSpeech

__all__ = [
    "LLMInference",
    "DialogueManager",
    "MemoryStore",
    "EmotionDetector",
    "SpeechRecognizer",
    "TextToSpeech",
]