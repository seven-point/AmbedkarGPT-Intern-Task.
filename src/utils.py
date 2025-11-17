import os
from dotenv import load_dotenv

load_dotenv()

def get_env(key: str, default=None):
    return os.environ.get(key, default)

def read_speech_file(path: str = "speech.txt") -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"speech file not found at {path}. Please put 'speech.txt' in the project root.")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
