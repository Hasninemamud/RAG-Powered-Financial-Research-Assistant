from pydantic import BaseModel
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

class Settings(BaseModel):
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))
    TOP_K: int = int(os.getenv("TOP_K", 3))
    INDEX_DIR: str = os.getenv("INDEX_DIR", "models")
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")

    @property
    def index_path(self) -> Path:
        return Path(self.INDEX_DIR) / "index.faiss"

    @property
    def store_path(self) -> Path:
        return Path(self.INDEX_DIR) / "store.json"

settings = Settings()