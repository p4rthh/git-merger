from enum import Enum

from pydantic_settings import BaseSettings, SettingsConfigDict


class SandboxProvider(str, Enum):
    contree = "contree"
    e2b = "e2b"
    local_docker = "local_docker"


class Settings(BaseSettings):

    # API credentials
    NEBIUS_API_KEY: str
    CONTREE_AUTH_TOKEN: str
    CONTREE_PROJECT_ID: str

    # Sandbox
    SANDBOX_PROVIDER: SandboxProvider = SandboxProvider.contree

    # Repair settings
    MAX_REPAIR_ITERATIONS: int = 5
    TOKEN_BUDGET: int = 500000
    SANDBOX_TIMEOUT_SECONDS: int = 60

    # Models
    ULTRA_MODEL_ID: str = "nvidia/nemotron-3-ultra-550b-a55b"
    SUPER_MODEL_ID: str = "nvidia/nemotron-3-super-120b-a12b"
    NANO_MODEL_ID: str = "nvidia/nemotron-3-nano-30b-a3b"

    # Read from environment, with .env as fallback
    model_config = SettingsConfigDict(
        env_file=".env.example",
        env_file_encoding="utf-8",
    )


settings = Settings()