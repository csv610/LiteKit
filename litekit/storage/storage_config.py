"""Storage Configuration - Centralized LMDB Storage Settings."""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class StorageConfig:
    """
    Centralized configuration for LMDB database storage.

    Attributes:
        db_path: Path to LMDB database file. Auto-generated per module in the
            user's cache directory. Can be overridden.
        db_capacity_mb: Maximum database capacity in MB. Default: 500MB
        db_store: Whether to cache results in database. Default: True (enabled)
        db_overwrite: If True, overwrite existing cached entries.
            If False, use cached entry if exists. Default: False
    """

    db_path: Optional[str] = None
    db_capacity_mb: int = 500
    db_store: bool = True
    db_overwrite: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.db_capacity_mb <= 0:
            raise ValueError("db_capacity_mb must be greater than 0")
        if not isinstance(self.db_store, bool):
            raise ValueError("db_store must be a boolean")
        if not isinstance(self.db_overwrite, bool):
            raise ValueError("db_overwrite must be a boolean")

    @classmethod
    def for_module(cls, module_name: str) -> "StorageConfig":
        """
        Create StorageConfig with auto-generated db_path for a module.

        Args:
            module_name: Name of the module (e.g., "medicine_info", "disease_info")

        Returns:
            StorageConfig with db_path set to: ~/.cache/litekit/{module_name}.lmdb
        """
        cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "litekit"
        cache_dir.mkdir(parents=True, exist_ok=True)
        db_path = str(cache_dir / f"{module_name}.lmdb")
        return cls(db_path=db_path)
