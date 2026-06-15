"""Tests for storage configuration module."""

import pytest
from litekit.storage.storage_config import StorageConfig


class TestStorageConfig:
    def test_defaults(self):
        cfg = StorageConfig()
        assert cfg.db_path is None
        assert cfg.db_capacity_mb == 500
        assert cfg.db_store is True
        assert cfg.db_overwrite is False

    def test_custom_values(self):
        cfg = StorageConfig(
            db_path="/tmp/test.lmdb",
            db_capacity_mb=100,
            db_store=False,
            db_overwrite=True,
        )
        assert cfg.db_path == "/tmp/test.lmdb"
        assert cfg.db_capacity_mb == 100
        assert cfg.db_store is False
        assert cfg.db_overwrite is True

    def test_negative_capacity_raises(self):
        with pytest.raises(ValueError, match="db_capacity_mb must be greater than 0"):
            StorageConfig(db_capacity_mb=-1)

    def test_zero_capacity_raises(self):
        with pytest.raises(ValueError, match="db_capacity_mb must be greater than 0"):
            StorageConfig(db_capacity_mb=0)

    def test_non_bool_db_store_raises(self):
        with pytest.raises(ValueError, match="db_store must be a boolean"):
            StorageConfig(db_store="yes")

    def test_non_bool_db_overwrite_raises(self):
        with pytest.raises(ValueError, match="db_overwrite must be a boolean"):
            StorageConfig(db_overwrite=1)

    def test_for_module(self):
        cfg = StorageConfig.for_module("test_module")
        assert cfg.db_path is not None
        assert "test_module" in cfg.db_path
        assert cfg.db_path.endswith(".lmdb")
        assert cfg.db_capacity_mb == 500
