"""Configuration utilities for environment-driven settings."""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    supabase_db_url: Optional[str]
    supabase_schema: str = "public"

    @classmethod
    def load(cls) -> "Settings":
        """Load settings from environment variables with sensible defaults."""
        supabase_db_url = os.getenv("SUPABASE_DB_URL")
        supabase_schema = os.getenv("SUPABASE_SCHEMA", "public")
        return cls(supabase_db_url=supabase_db_url, supabase_schema=supabase_schema)


settings = Settings.load()
