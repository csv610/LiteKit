"""Tests for LMDB storage module."""

import json
import os

import pytest

from litekit.lmdb_storage import LMDBConfig, LMDBStorage


class TestLMDBConfig:
    def test_defaults(self):
        cfg = LMDBConfig()
        assert cfg.db_path == "storage.lmdb"
        assert cfg.capacity_mb == 100
        assert cfg.enable_logging is True
        assert cfg.compression_threshold == 100
        assert cfg.max_key_size == 511

    def test_custom_values(self):
        cfg = LMDBConfig(
            db_path="/tmp/test.lmdb",
            capacity_mb=200,
            enable_logging=False,
            compression_threshold=500,
            max_key_size=100,
        )
        assert cfg.db_path == "/tmp/test.lmdb"
        assert cfg.capacity_mb == 200
        assert cfg.enable_logging is False
        assert cfg.compression_threshold == 500
        assert cfg.max_key_size == 100

    def test_zero_capacity_raises(self):
        with pytest.raises(ValueError, match="capacity_mb must be greater than 0"):
            LMDBConfig(capacity_mb=0)

    def test_negative_capacity_raises(self):
        with pytest.raises(ValueError, match="capacity_mb must be greater than 0"):
            LMDBConfig(capacity_mb=-1)

    def test_negative_compression_threshold_raises(self):
        with pytest.raises(ValueError, match="compression_threshold must be non-negative"):
            LMDBConfig(compression_threshold=-1)

    def test_zero_max_key_size_raises(self):
        with pytest.raises(ValueError, match="max_key_size must be greater than 0"):
            LMDBConfig(max_key_size=0)

    def test_empty_db_path_raises(self):
        with pytest.raises(ValueError, match="db_path must be a non-empty string"):
            LMDBConfig(db_path="")

    def test_whitespace_db_path_raises(self):
        with pytest.raises(ValueError, match="db_path must be a non-empty string"):
            LMDBConfig(db_path="   ")


class TestLMDBStorage:
    DB_PATH = "/tmp/test_lmdb_storage.lmdb"

    def setup_method(self):
        self._cleanup()

    def teardown_method(self):
        self._cleanup()

    def _cleanup(self):
        import shutil
        if os.path.exists(self.DB_PATH):
            shutil.rmtree(self.DB_PATH, ignore_errors=True)

    def test_put_and_get(self):
        storage = LMDBStorage(
            db_path=self.DB_PATH,
            capacity_mb=10,
            enable_logging=False,
        )
        try:
            assert storage.put("key1", "value1") is True
            assert storage.get("key1") == "value1"
        finally:
            storage.close()

    def test_get_nonexistent_key(self):
        storage = LMDBStorage(
            db_path=self.DB_PATH,
            capacity_mb=10,
            enable_logging=False,
        )
        try:
            assert storage.get("nonexistent") is None
        finally:
            storage.close()

    def test_put_empty_key_returns_false(self):
        storage = LMDBStorage(
            db_path=self.DB_PATH,
            capacity_mb=10,
            enable_logging=False,
        )
        try:
            assert storage.put("", "value") is False
        finally:
            storage.close()

    def test_put_none_value_returns_false(self):
        storage = LMDBStorage(
            db_path=self.DB_PATH,
            capacity_mb=10,
            enable_logging=False,
        )
        try:
            assert storage.put("key", None) is False
        finally:
            storage.close()

    def test_exists(self):
        storage = LMDBStorage(
            db_path=self.DB_PATH,
            capacity_mb=10,
            enable_logging=False,
        )
        try:
            storage.put("key1", "value1")
            assert storage.exists("key1") is True
            assert storage.exists("nonexistent") is False
        finally:
            storage.close()

    def test_delete(self):
        storage = LMDBStorage(
            db_path=self.DB_PATH,
            capacity_mb=10,
            enable_logging=False,
        )
        try:
            storage.put("key1", "value1")
            assert storage.get("key1") == "value1"
            assert storage.delete("key1") is True
            assert storage.get("key1") is None
        finally:
            storage.close()

    def test_delete_nonexistent_returns_false(self):
        storage = LMDBStorage(
            db_path=self.DB_PATH,
            capacity_mb=10,
            enable_logging=False,
        )
        try:
            assert storage.delete("nonexistent") is False
        finally:
            storage.close()

    def test_num_keys(self):
        storage = LMDBStorage(
            db_path=self.DB_PATH,
            capacity_mb=10,
            enable_logging=False,
        )
        try:
            assert storage.num_keys() == 0
            storage.put("a", "1")
            storage.put("b", "2")
            assert storage.num_keys() == 2
        finally:
            storage.close()

    def test_clear(self):
        storage = LMDBStorage(
            db_path=self.DB_PATH,
            capacity_mb=10,
            enable_logging=False,
        )
        try:
            storage.put("a", "1")
            storage.put("b", "2")
            assert storage.num_keys() == 2
            cleared = storage.clear()
            assert cleared == 2
            assert storage.num_keys() == 0
        finally:
            storage.close()

    def test_get_keys(self):
        storage = LMDBStorage(
            db_path=self.DB_PATH,
            capacity_mb=10,
            enable_logging=False,
        )
        try:
            storage.put("a", "1")
            storage.put("b", "2")
            keys = storage.get_keys()
            assert sorted(keys) == ["a", "b"]
        finally:
            storage.close()

    def test_get_keys_as_generator(self):
        storage = LMDBStorage(
            db_path=self.DB_PATH,
            capacity_mb=10,
            enable_logging=False,
        )
        try:
            storage.put("a", "1")
            storage.put("b", "2")
            gen = storage.get_keys(as_generator=True)
            keys = sorted(list(gen))
            assert keys == ["a", "b"]
        finally:
            storage.close()

    def test_get_stats(self):
        storage = LMDBStorage(
            db_path=self.DB_PATH,
            capacity_mb=10,
            enable_logging=False,
        )
        try:
            storage.put("a", "1")
            stats = storage.get_stats()
            assert isinstance(stats, dict)
            assert "entries" in stats
        finally:
            storage.close()

    def test_context_manager(self):
        with LMDBStorage(
            db_path=self.DB_PATH,
            capacity_mb=10,
            enable_logging=False,
        ) as storage:
            storage.put("ctx_key", "ctx_value")
            assert storage.get("ctx_key") == "ctx_value"

    def test_compression(self):
        with LMDBStorage(
            db_path=self.DB_PATH,
            capacity_mb=10,
            enable_logging=False,
            compression_threshold=10,
        ) as storage:
            small_val = "hello"
            large_val = "x" * 1000
            assert storage.put("small", small_val) is True
            assert storage.put("large", large_val) is True
            assert storage.get("small") == small_val
            assert storage.get("large") == large_val

    def test_export_to_json(self, tmp_path):
        with LMDBStorage(
            db_path=self.DB_PATH,
            capacity_mb=10,
            enable_logging=False,
        ) as storage:
            storage.put("key1", "value1")
            storage.put("key2", "value2")

            json_path = str(tmp_path / "export.json")
            assert storage.export_to_json(json_path) is True

            with open(json_path) as f:
                data = json.load(f)
            assert len(data) == 2
            entries = {(e["key"], e["value"]) for e in data}
            assert ("key1", "value1") in entries
            assert ("key2", "value2") in entries

    def test_import_from_json(self, tmp_path):
        json_path = str(tmp_path / "import.json")
        data = [
            {"key": "k1", "value": "v1"},
            {"key": "k2", "value": "v2"},
        ]
        with open(json_path, "w") as f:
            json.dump(data, f)

        with LMDBStorage(
            db_path=self.DB_PATH,
            capacity_mb=10,
            enable_logging=False,
        ) as storage:
            assert storage.import_from_json(json_path) is True
            assert storage.get("k1") == "v1"
            assert storage.get("k2") == "v2"
            assert storage.num_keys() == 2

    def test_import_invalid_json_returns_false(self, tmp_path):
        bad_path = str(tmp_path / "bad.json")
        with open(bad_path, "w") as f:
            f.write("not json")

        with LMDBStorage(
            db_path=self.DB_PATH,
            capacity_mb=10,
            enable_logging=False,
        ) as storage:
            assert storage.import_from_json(bad_path) is False

    def test_import_not_list_returns_false(self, tmp_path):
        bad_path = str(tmp_path / "not_list.json")
        with open(bad_path, "w") as f:
            json.dump({"key": "value"}, f)

        with LMDBStorage(
            db_path=self.DB_PATH,
            capacity_mb=10,
            enable_logging=False,
        ) as storage:
            assert storage.import_from_json(bad_path) is False

    def test_import_file_not_found_returns_false(self):
        with LMDBStorage(
            db_path=self.DB_PATH,
            capacity_mb=10,
            enable_logging=False,
        ) as storage:
            assert storage.import_from_json("/nonexistent/file.json") is False

    def test_import_skips_invalid_entries(self, tmp_path):
        json_path = str(tmp_path / "mixed.json")
        data = [
            {"key": "k1", "value": "v1"},
            {"invalid": "entry"},
            {"key": "", "value": "empty_key"},
        ]
        with open(json_path, "w") as f:
            json.dump(data, f)

        with LMDBStorage(
            db_path=self.DB_PATH,
            capacity_mb=10,
            enable_logging=False,
        ) as storage:
            assert storage.import_from_json(json_path) is True
            assert storage.get("k1") == "v1"

    def test_overwrite_existing_key(self):
        with LMDBStorage(
            db_path=self.DB_PATH,
            capacity_mb=10,
            enable_logging=False,
        ) as storage:
            assert storage.put("key", "old") is True
            assert storage.put("key", "new") is True
            assert storage.get("key") == "new"

    def test_large_compressed_value(self):
        with LMDBStorage(
            db_path=self.DB_PATH,
            capacity_mb=50,
            enable_logging=False,
            compression_threshold=1,
        ) as storage:
            large = "A" * 10000
            assert storage.put("large", large) is True
            assert storage.get("large") == large

    def test_none_value_put(self):
        with LMDBStorage(
            db_path=self.DB_PATH,
            capacity_mb=10,
            enable_logging=False,
        ) as storage:
            assert storage.put("key", None) is False

    def test_empty_key_get(self):
        with LMDBStorage(
            db_path=self.DB_PATH,
            capacity_mb=10,
            enable_logging=False,
        ) as storage:
            assert storage.get("") is None

    def test_empty_key_exists(self):
        with LMDBStorage(
            db_path=self.DB_PATH,
            capacity_mb=10,
            enable_logging=False,
        ) as storage:
            assert storage.exists("") is False
