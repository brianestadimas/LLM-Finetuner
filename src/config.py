# src/config.py
import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env file
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

class Config:
    RUNPOD_KEY = os.getenv("RUNPOD_KEY")
