"""Main application entry point for MyWeePal."""

import logging
import sys
import signal
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.logging import RichHandler
import asyncio
from dotenv import load_dotenv
import os

from myweepal.core.dialogue import DialogueManager
from myweepal.core.llm import LLMConfig
from myweepal.audio.tts import TTSConfig

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)

# CLI app
app = typer.Typer(help="MyWeePal - Local Conversational Life-Story AI")
console = Console()


class MyWeePalApp:
    """Main application class."""

    def __init__(
        self,
        mlx_model: Optional[str] = None,
        memory_dir: Optional[str] = None,
        enable_video: bool = False,
    ):
        """Initialize MyWeePal application.

        Args:
            mlx_model: MLX model name
            memory_dir: Directory for memory storage
            enable_video: Enable video display
        """
        self.mlx_model = mlx_model or os.getenv("MLX_MODEL_NAME", "mlx-community/Llama-3.2-3B-Instruct-4bit")
        self.memory_dir = memory_dir or os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
        self.enable_video = enable_video

        self.dialogue_manager = None
        self.is_running = False

    def initialize(self) -> None:
        """Initialize all components."""
        console.print(Panel.fit("🤖 MyWeePal Initialization", style="bold blue"))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Initialize LLM
            task = progress.add_task("Loading MLX model...", total=1)
            llm_config = LLMConfig(
                model_name=self.mlx_model,
                max_tokens=int(os.getenv("MLX_MAX_TOKENS", "256")),
                temperature=float(os.getenv("MLX_TEMPERATURE", "0.7")),
            )
            progress.update(task, completed=1)

            # Initialize TTS
            task = progress.add_task("Setting up text-to-speech...", total=1)
            tts_config = TTSConfig()
            progress.update(task, completed=1)

            # Initialize Dialogue Manager
            task = progress.add_task("Initializing dialogue system...", total=1)
            self.dialogue_manager = DialogueManager(
                llm_config=llm_config,
                tts_config=tts_config,
                memory_persist_dir=self.memory_dir,
            )
            progress.update(task, completed=1)

            # Configure video if enabled
            if self.enable_video:
                task = progress.add_task("Starting video capture...", total=1)
                self.dialogue_manager.emotion_detector.enable_display = True
                progress.update(task, completed=1)

        console.print("✅ [green]Initialization complete![/green]")

    def run_interactive_session(self) -> None:
        """Run interactive conversation session."""
        self.is_running = True

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_interrupt)

        console.print("\n" + "=" * 50)
        console.print("[bold cyan]MyWeePal Life Story Interview[/bold cyan]")
        console.print("=" * 50)
        console.print("\n[yellow]Press Ctrl+C to end the session[/yellow]\n")

        try:
            # Start the dialogue session
            self.dialogue_manager.start_session()

            # Keep running until interrupted
            while self.is_running:
                # Display session stats periodically
                asyncio.run(self._monitor_session())

        except KeyboardInterrupt:
            console.print("\n[yellow]Session interrupted by user[/yellow]")
        finally:
            self._cleanup()

    async def _monitor_session(self) -> None:
        """Monitor and display session statistics."""
        while self.is_running:
            await asyncio.sleep(30)  # Update every 30 seconds

            if self.dialogue_manager and self.dialogue_manager.is_active:
                stats = self.dialogue_manager.get_session_stats()

                console.print("\n[dim]--- Session Status ---[/dim]")
                console.print(f"Phase: {stats['phase']}")
                console.print(f"Questions: {stats['questions_asked']}")
                console.print(f"Memories: {stats['memories_created']}")
                console.print(f"Mood: {stats['user_mood']}")
                console.print(f"Engagement: {stats['engagement_level']:.2f}")
                console.print("[dim]-------------------[/dim]\n")

    def _handle_interrupt(self, signum, frame) -> None:
        """Handle interrupt signal."""
        console.print("\n[yellow]Shutting down gracefully...[/yellow]")
        self.is_running = False

    def _cleanup(self) -> None:
        """Clean up resources."""
        if self.dialogue_manager:
            console.print("[yellow]Saving session data...[/yellow]")
            self.dialogue_manager.stop_session()

        console.print("[green]Session ended. Thank you![/green]")

    def export_memories(self, output_file: str) -> None:
        """Export memories to file.

        Args:
            output_file: Output file path
        """
        if not self.dialogue_manager:
            self.initialize()

        console.print(f"[yellow]Exporting memories to {output_file}...[/yellow]")

        memories = self.dialogue_manager.memory_store.get_all_memories()

        import json

        export_data = {
            "total_memories": len(memories),
            "memories": [
                {
                    "id": m.id,
                    "question": m.question,
                    "answer": m.answer,
                    "timestamp": m.timestamp,
                    "category": m.category,
                    "emotion": m.emotion,
                }
                for m in memories
            ],
        }

        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2)

        console.print(f"[green]✅ Exported {len(memories)} memories to {output_file}[/green]")

    def analyze_emotions(self) -> None:
        """Analyze emotion patterns from stored memories."""
        if not self.dialogue_manager:
            self.initialize()

        stats = self.dialogue_manager.memory_store.get_memory_stats()

        console.print("\n[bold]Emotion Analysis[/bold]")
        console.print("=" * 30)

        if stats["emotions"]:
            for emotion, count in sorted(stats["emotions"].items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int((count / max(stats["emotions"].values())) * 20)
                console.print(f"{emotion:12} {bar} {count}")
        else:
            console.print("[dim]No emotion data available[/dim]")

        console.print("\n[bold]Categories[/bold]")
        console.print("=" * 30)

        if stats["categories"]:
            for category, count in sorted(stats["categories"].items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int((count / max(stats["categories"].values())) * 20)
                console.print(f"{category:12} {bar} {count}")
        else:
            console.print("[dim]No category data available[/dim]")


@app.command()
def start(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="MLX model name"),
    memory_dir: Optional[str] = typer.Option(None, "--memory-dir", "-d", help="Memory storage directory"),
    enable_video: bool = typer.Option(False, "--video", "-v", help="Enable video display"),
):
    """Start an interactive MyWeePal session."""
    app_instance = MyWeePalApp(
        mlx_model=model,
        memory_dir=memory_dir,
        enable_video=enable_video,
    )

    try:
        app_instance.initialize()
        app_instance.run_interactive_session()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@app.command()
def export(
    output: str = typer.Argument(..., help="Output file path"),
    memory_dir: Optional[str] = typer.Option(None, "--memory-dir", "-d", help="Memory storage directory"),
):
    """Export memories to a JSON file."""
    app_instance = MyWeePalApp(memory_dir=memory_dir)

    try:
        app_instance.export_memories(output)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@app.command()
def analyze(
    memory_dir: Optional[str] = typer.Option(None, "--memory-dir", "-d", help="Memory storage directory"),
):
    """Analyze emotion patterns from memories."""
    app_instance = MyWeePalApp(memory_dir=memory_dir)

    try:
        app_instance.analyze_emotions()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@app.command()
def calibrate(
    duration: float = typer.Option(10.0, "--duration", "-t", help="Calibration duration in seconds"),
    enable_video: bool = typer.Option(True, "--video", "-v", help="Enable video display"),
):
    """Calibrate emotion detection baseline."""
    app_instance = MyWeePalApp(enable_video=enable_video)

    try:
        app_instance.initialize()

        console.print("[yellow]Starting emotion calibration...[/yellow]")
        console.print("[dim]Please look at the camera and maintain a neutral expression[/dim]")

        baseline = app_instance.dialogue_manager.emotion_detector.calibrate(duration)

        console.print("\n[green]Calibration complete![/green]")
        console.print("\nBaseline emotions:")
        for emotion, score in baseline.items():
            console.print(f"  {emotion}: {score:.3f}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@app.command()
def test(
    component: str = typer.Argument(..., help="Component to test (llm, asr, tts, emotion, memory)"),
):
    """Test individual components."""
    console.print(f"[yellow]Testing {component}...[/yellow]")

    try:
        if component == "llm":
            from myweepal.core.llm import LLMInference

            llm = LLMInference()
            response = llm.generate_response("Hello, how are you?")
            console.print(f"[green]LLM Response: {response}[/green]")

        elif component == "asr":
            from myweepal.audio.asr import SpeechRecognizer

            asr = SpeechRecognizer()
            console.print("Speak something (5 seconds)...")
            result = asr.record_and_transcribe(5.0)
            if result:
                console.print(f"[green]Transcription: {result.text}[/green]")
            else:
                console.print("[red]No transcription[/red]")

        elif component == "tts":
            from myweepal.audio.tts import TextToSpeech

            tts = TextToSpeech()
            tts.speak("Hello! This is a test of the text to speech system.", blocking=True)
            console.print("[green]TTS test complete[/green]")

        elif component == "emotion":
            from myweepal.vision.emotion import EmotionDetector

            detector = EmotionDetector(enable_display=True)
            detector.start()
            console.print("Detecting emotions for 10 seconds...")
            import time

            time.sleep(10)
            emotion = detector.get_current_emotion()
            detector.stop()

            if emotion:
                console.print(f"[green]Detected emotion: {emotion.dominant_emotion}[/green]")
                console.print(f"Confidence: {emotion.confidence:.2f}")
            else:
                console.print("[yellow]No emotion detected[/yellow]")

        elif component == "memory":
            from myweepal.core.memory import MemoryStore
            import numpy as np

            store = MemoryStore()
            embedding = np.random.randn(768)
            memory_id = store.add_memory(
                "Test question?",
                "Test answer.",
                embedding,
                category="test",
                emotion="happy",
            )
            console.print(f"[green]Memory stored with ID: {memory_id}[/green]")

            stats = store.get_memory_stats()
            console.print(f"Total memories: {stats['total_memories']}")

        else:
            console.print(f"[red]Unknown component: {component}[/red]")

    except Exception as e:
        console.print(f"[red]Test failed: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    app()