from sqlalchemy import create_engine, Column, String, DateTime, JSON, Index
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

DATABASE_URL = "sqlite:///./llmfinetune.db"

engine = create_engine(
    DATABASE_URL,
    future=True,
    echo=False, 
    connect_args={"check_same_thread": False}, 
)
SessionLocal = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False))
Base = declarative_base()

class Conversation(Base):
    __tablename__ = "conversations"

    session_id = Column(String, primary_key=True, index=True)
    history = Column(JSON, default=list) 

def init_db() -> None:
    """Call once at startup."""
    Base.metadata.create_all(bind=engine)


DEFAULT_SESSION_ID = "default"          # <= the implicit session
def ensure_default_session() -> None:
    """Create the default row once at start‑up."""
    with SessionLocal() as db:
        if db.get(Conversation, DEFAULT_SESSION_ID) is None:
            db.add(Conversation(session_id=DEFAULT_SESSION_ID, history=[]))
            db.commit()