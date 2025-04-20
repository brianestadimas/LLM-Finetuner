import os
from typing import List, Dict
from src.db import SessionLocal, Conversation

def find_highest_checkpoint(checkpoint_dir: str) -> str:
    checkpoints = [
        d for d in os.listdir(checkpoint_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoint_dir, d))
    ]
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")

    # Sort by the numeric portion after "checkpoint-"
    checkpoints_sorted = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    highest_checkpoint = checkpoints_sorted[-1]
    return os.path.join(checkpoint_dir, highest_checkpoint)

def folder_has_files(path: str) -> bool:
    return os.path.exists(path) and any(os.scandir(path))

def load_history(session_id: str) -> List[Dict]:
    with SessionLocal() as db:
        convo = db.get(Conversation, session_id)
        return convo.history if convo else []

def save_history(session_id: str, history: List[Dict]) -> None:
    with SessionLocal() as db:
        convo = db.get(Conversation, session_id)
        if convo is None:                         # brand‑new session row
            convo = Conversation(session_id=session_id, history=history)
            db.add(convo)
        else:
            convo.history = history
        db.commit()
