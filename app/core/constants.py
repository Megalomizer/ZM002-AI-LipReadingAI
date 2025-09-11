from pathlib import Path

OLLAMA_MODEL="llama3.2"
VISUAL_MODEL="SilentSpeak/LipCoordNet"
FRAMES_COLLECTION=64

BASE_DIR= Path(__file__).resolve().parent.parent
DATABASE_URL= BASE_DIR / "data" / "data.db"

OLLAMA_LIPREAD_PROMPT="""
Interpret the lipreading given in the context.
"""