import yaml
import os
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv

load_dotenv()  # Load constraints from .env file


class AppConfig:
    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppConfig, cls).__new__(cls)
            cls._instance.load()
        return cls._instance

    def load(self, config_path: str = "config.yaml"):
        """Load configuration from yaml file."""
        # Look for config in project root
        root_dir = Path(__file__).parent.parent.parent
        path = root_dir / config_path
        
        if not path.exists():
            # Fallback to looking in current directory (e.g. for tests)
            path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found at {path} or {config_path}")

        with open(path, "r") as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    @property
    def duration_target_s(self) -> int:
        return self.get("duration_target_s", 4620)

    @property
    def duration_cap_s(self) -> int:
        return self.get("duration_cap_s", 7200)

    @property
    def max_tracks_per_artist(self) -> int:
        return self.get("max_tracks_per_artist", 2)
    
    @property
    def weights(self) -> Dict[str, float]:
        return {
            "fit": self.get("w_fit", 0.35),
            "flow": self.get("w_flow", 0.25),
            "variety": self.get("w_variety", 0.15),
            "access": self.get("w_access", 0.15),
            "quality": self.get("w_quality", 0.10),
        }

    # Secrets (Environment Variables)
    @property
    def spotify_client_id(self) -> str:
        return os.getenv("SPOTIFY_CLIENT_ID", "")

    @property
    def spotify_client_secret(self) -> str:
        return os.getenv("SPOTIFY_CLIENT_SECRET", "")

    @property
    def openai_api_key(self) -> str:
        return os.getenv("OPENAI_API_KEY", "")

    @property
    def anthropic_api_key(self) -> str:
        return os.getenv("ANTHROPIC_API_KEY", "")
        
    @property
    def spotify_redirect_uri(self) -> str:
        return os.getenv("SPOTIFY_REDIRECT_URI", "http://localhost:8888/callback")

# Global instance
config = AppConfig()
