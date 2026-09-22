from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    chroma_db_path: str = "./chroma_db"
    upload_dir: str = "./uploads"
    max_file_size_mb: int = 50
    top_k_results: int = 5
    chunk_size: int = 500
    chunk_overlap: int = 50
    cors_origins: str = "http://localhost:3000"

    class Config:
        env_file = ".env"
        extra = "ignore"

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

settings = Settings()
