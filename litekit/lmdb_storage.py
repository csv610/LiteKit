import gzip
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import lmdb


@dataclass
class LMDBConfig:
    """
    Configuration dataclass for LMDBStorage.

    Attributes:
        db_path (str): Path to the LMDB database file. Defaults to "storage.lmdb".
        capacity_mb (int): Maximum database capacity in megabytes. Defaults to 100 MB.
        enable_logging (bool): Whether to enable logging for this instance. Defaults to True.
        compression_threshold (int): Size in bytes above which values will be compressed.
            Defaults to 100 bytes.
        max_key_size (int): Maximum allowed key size in bytes. Defaults to 511 (LMDB default).

    Example:
        >>> config = LMDBConfig(
        ...     db_path="mydata.lmdb",
        ...     capacity_mb=200,
        ...     compression_threshold=1024
        ... )
        >>> storage = LMDBStorage(config=config)
    """
    db_path: str = "storage.lmdb"
    capacity_mb: int = 100
    enable_logging: bool = True
    compression_threshold: int = 100
    max_key_size: int = 511

    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        if self.capacity_mb <= 0:
            raise ValueError("capacity_mb must be greater than 0")
        if self.compression_threshold < 0:
            raise ValueError("compression_threshold must be non-negative")
        if self.max_key_size <= 0:
            raise ValueError("max_key_size must be greater than 0")
        if not self.db_path or self.db_path.strip() == "":
            raise ValueError("db_path must be a non-empty string")

class LMDBStorage:
    """
    A key-value storage class using LMDB with optional compression.

    This class provides a simple interface for storing and retrieving string key-value pairs
    in an LMDB database. It automatically compresses large values to save space and includes
    built-in logging for monitoring operations.

    Features:
        - Automatic compression for values exceeding a configurable threshold
        - Built-in logging with per-instance log files
        - Context manager support for automatic cleanup
        - JSON import/export capabilities
        - Memory-efficient key iteration with generator support

    Example:
        Basic usage with context manager:

        >>> with LMDBStorage("mydata.lmdb", capacity_mb=100) as storage:
        ...     # Store data
        ...     storage.put("user:1", "John Doe")
        ...
        ...     # Retrieve data
        ...     name = storage.get("user:1")
        ...     print(name)  # Output: John Doe
        ...
        ...     # Check existence
        ...     if storage.exists("user:1"):
        ...         storage.delete("user:1")

        Manual usage:

        >>> storage = LMDBStorage("mydata.lmdb")
        >>> storage.put("config", "debug=true")
        >>> value = storage.get("config")
        >>> storage.close()  # Remember to close manually

        JSON import/export:

        >>> storage = LMDBStorage("mydata.lmdb")
        >>> storage.import_from_json("data.json")
        >>> storage.export_to_json("backup.json")
        >>> storage.close()

    Attributes:
        db_path (str): Path to the LMDB database file
        compression_threshold (int): Byte threshold for compression
        capacity_mb (int): Maximum database size in megabytes
        logger (logging.Logger): Logger instance for this storage
        env (lmdb.Environment): LMDB environment handle
    """
    # Compression flag constants
    COMPRESSION_FLAG_COMPRESSED = b'\x01'
    COMPRESSION_FLAG_UNCOMPRESSED = b'\x00'

    # LMDB has a default max key size of 511 bytes
    DEFAULT_MAX_KEY_SIZE = 511

    def __init__(self, db_path=None, capacity_mb=None, enable_logging=None,
                 compression_threshold=None, config: Optional[LMDBConfig] = None):
        """
        Initializes the LMDB storage manager.

        Args:
            db_path (str, optional): The file path for the LMDB database. Defaults to "storage.lmdb".
            capacity_mb (int, optional): Maximum database capacity in megabytes. Defaults to 100 MB.
            enable_logging (bool, optional): Whether to enable logging for this instance. Defaults to True.
            compression_threshold (int, optional): Size in bytes above which values will be compressed.
                Defaults to 100 bytes.
            config (LMDBConfig, optional): Configuration dataclass. If provided, individual parameters
                are ignored. Defaults to None.

        Raises:
            Exception: If the LMDB database cannot be opened (e.g., permission denied, disk full).

        Example:
            Using individual parameters:
            >>> storage = LMDBStorage(db_path="mydata.lmdb", capacity_mb=200)

            Using config dataclass:
            >>> config = LMDBConfig(db_path="mydata.lmdb", capacity_mb=200)
            >>> storage = LMDBStorage(config=config)
        """
        # If config is provided, use it; otherwise create from parameters
        if config is not None:
            self.config = config
        else:
            self.config = LMDBConfig(
                db_path=db_path if db_path is not None else "storage.lmdb",
                capacity_mb=capacity_mb if capacity_mb is not None else 100,
                enable_logging=enable_logging if enable_logging is not None else True,
                compression_threshold=compression_threshold if compression_threshold is not None else 100
            )

        # Set instance attributes from config
        self.db_path = self.config.db_path
        self.compression_threshold = self.config.compression_threshold
        self.capacity_mb = self.config.capacity_mb
        self.max_key_size = self.config.max_key_size

        self.logger = self._setup_logger(self.config.enable_logging)
        self.env = self._open_database(self.config.capacity_mb)

    def _setup_logger(self, enable_logging):
        """
        Internal method to set up logging for this instance.

        Creates a unique logger for this storage instance with a file handler.
        The log file is named after the database file with underscores replacing dots.

        Args:
            enable_logging (bool): Whether to create a logger.

        Returns:
            logging.Logger or None: Configured logger instance, or None if logging is disabled.
        """
        if not enable_logging:
            return None

        # Use a unique logger name for each instance
        logger_name = f"{__name__}.{self.__class__.__name__}.{id(self)}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)

        # Ensure no duplicate handlers are added
        if not logger.handlers:
            file_handler = logging.FileHandler(f"{os.path.basename(self.db_path).replace('.', '_')}.log")
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            logger.addHandler(file_handler)
            # Prevents log messages from propagating to the root logger
            logger.propagate = False
            logger.info("Logger configured successfully")

        return logger

    def _open_database(self, capacity_mb):
        """
        Internal method to open LMDB database connection.

        Args:
            capacity_mb (int): Capacity in MB for the database size.

        Returns:
            lmdb.Environment: Opened LMDB environment.

        Raises:
            Exception: If the database cannot be opened (propagated from lmdb.open).
        """
        # LMDB map_size needs to be in bytes (1 MB = 1024 * 1024 bytes)
        map_size = 1024 * 1024 * capacity_mb
        try:
            env = lmdb.open(self.db_path, map_size=map_size)
            if self.logger:
                self.logger.info(f"LMDB database opened successfully: {self.db_path} (capacity: {capacity_mb}MB)")
            return env
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to open LMDB database {self.db_path}: {e}")
            raise

    def put(self, key: str, value: str) -> bool:
        """
        Stores a key-value pair in the LMDB database with optional compression.

        The value is automatically compressed if its size exceeds the compression threshold.
        Keys are validated to ensure they don't exceed LMDB's maximum key size (511 bytes).

        Args:
            key (str): The key to store. Must be non-empty and not exceed 511 bytes when UTF-8 encoded.
            value (str): The value to store. Will be compressed if larger than compression_threshold.

        Returns:
            bool: True if storage was successful, False if validation failed or an error occurred.

        Example:
            >>> storage = LMDBStorage()
            >>> storage.put("user:1", "John Doe")
            True
            >>> storage.put("", "value")  # Empty key
            False
        """
        if not self._validate_key_value(key, value):
            return False

        try:
            key_bytes = key.encode('utf-8')
            final_value = self._encode_value(value)

            with self.env.begin(write=True) as txn:
                txn.put(key_bytes, final_value)
                if self.logger:
                    self.logger.info(f"Successfully stored key '{key}' ({len(final_value)} bytes)")
                return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to store key '{key}': {e}")
            return False

    def _validate_key_value(self, key: str, value: str) -> bool:
        """
        Internal helper to validate key and value before storage.
        Args:
            key (str): The key to validate.
            value (str): The value to validate.
        Returns:
            bool: True if valid, False otherwise.
        """
        if not key or key.strip() == "":
            if self.logger:
                self.logger.warning("Attempted to store empty key")
            return False

        if value is None:
            if self.logger:
                self.logger.warning(f"Attempted to store None value for key '{key}'")
            return False

        key_bytes = key.encode('utf-8')
        if len(key_bytes) > self.max_key_size:
            if self.logger:
                self.logger.error(f"Key '{key}' exceeds maximum size of {self.max_key_size} bytes")
            return False

        return True

    def _encode_value(self, value: str) -> bytes:
        """
        Internal helper to encode and optionally compress a value.
        Args:
            value (str): The value to encode.
        Returns:
            bytes: Encoded and optionally compressed value with flag byte.
        """
        value_bytes = value.encode('utf-8')

        # Check if value is larger than the compression threshold
        if len(value_bytes) > self.compression_threshold:
            # Compress the value
            compressed_value = gzip.compress(value_bytes)
            if self.logger:
                compression_ratio = len(value_bytes) / len(compressed_value) if len(compressed_value) > 0 else 0
                self.logger.debug(f"Compressed value (ratio: {compression_ratio:.2f})")
            return self.COMPRESSION_FLAG_COMPRESSED + compressed_value
        else:
            # No compression needed
            if self.logger:
                self.logger.debug(f"Stored uncompressed value (size: {len(value_bytes)} bytes)")
            return self.COMPRESSION_FLAG_UNCOMPRESSED + value_bytes

    def _decode_value(self, stored_value: bytes) -> Optional[str]:
        """
        Internal helper to decode and decompress a stored value.
        Args:
            stored_value (bytes): The raw value from LMDB.
        Returns:
            str or None: The decoded value.
        """
        if not stored_value or len(stored_value) == 0:
            return None

        compression_flag = stored_value[0:1]

        if compression_flag == self.COMPRESSION_FLAG_COMPRESSED:
            # Compressed data
            return gzip.decompress(stored_value[1:]).decode('utf-8')
        elif compression_flag == self.COMPRESSION_FLAG_UNCOMPRESSED:
            # Uncompressed data
            return stored_value[1:].decode('utf-8')
        else:
            # Fallback for old data format (no flag byte)
            try:
                return gzip.decompress(stored_value).decode('utf-8')
            except (gzip.BadGzipFile, ValueError):
                # If decompression fails, treat as plain text
                return stored_value.decode('utf-8')

    def get(self, key: str) -> Optional[str]:
        """
        Retrieves a value by its key and decompresses it if needed.

        Args:
            key (str): The key to look up. Must be non-empty.

        Returns:
            str or None: The decompressed value if found, None if key doesn't exist or is invalid.

        Example:
            >>> storage = LMDBStorage()
            >>> storage.put("user:1", "John Doe")
            >>> storage.get("user:1")
            'John Doe'
            >>> storage.get("nonexistent")
            None
        """
        if not key or key.strip() == "":
            if self.logger:
                self.logger.warning("Attempted to retrieve empty key")
            return None

        try:
            with self.env.begin() as txn:
                stored_value = txn.get(key.encode('utf-8'))
                if stored_value is None:
                    if self.logger:
                        self.logger.debug(f"Key '{key}' not found")
                    return None

                value = self._decode_value(stored_value)
                if self.logger and value:
                    self.logger.debug(f"Retrieved value for key '{key}'")
                return value
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to retrieve key '{key}': {e}")
            return None

    def clear(self) -> int:
        """
        Deletes all entries from the database.
        Returns:
            int: Number of entries deleted.
        """
        try:
            with self.env.begin(write=True) as txn:
                count = txn.stat()["entries"]
                txn.drop(self.env.open_db(), delete=False)
            if self.logger:
                self.logger.info(f"Cleared {count} entries from database")
            return count
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to clear database: {e}")
            return 0
            
    def num_keys(self) -> int:
        """
        Returns the total number of keys in the database.
        Returns:
            int: Number of stored keys.
        """
        try:
            with self.env.begin() as txn:
                stat = txn.stat()
                count = stat['entries']
            if self.logger:
                self.logger.debug(f"Database contains {count} keys")
            return count
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to count keys: {e}")
            return 0
            
    def get_keys(self, as_generator: bool = False):
        """
        Retrieves all keys from the database.
        Args:
            as_generator (bool): If True, returns a generator for memory efficiency.
                                If False (default), returns a list for backwards compatibility.
        Returns:
            list or generator: All keys in the database.
        """
        if as_generator:
            return self._keys_generator()
        else:
            keys = []
            try:
                with self.env.begin() as txn:
                    cursor = txn.cursor()
                    for key in cursor.iternext(keys=True, values=False):
                        keys.append(key.decode('utf-8'))
                if self.logger:
                    self.logger.debug(f"Retrieved {len(keys)} keys from database")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to retrieve keys: {e}")
            return keys

    def _keys_generator(self):
        """
        Internal generator method that yields keys one at a time.
        Memory-efficient for large databases.
        Yields:
            str: Individual keys from the database.
        """
        try:
            with self.env.begin() as txn:
                cursor = txn.cursor()
                for key in cursor.iternext(keys=True, values=False):
                    yield key.decode('utf-8')
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to generate keys: {e}")

    def exists(self, key: str) -> bool:
        """
        Check if a key exists in the database without retrieving the value.
        Args:
            key (str): The key to check.
        Returns:
            bool: True if key exists, False otherwise.
        """
        try:
            with self.env.begin() as txn:
                exists = txn.get(key.encode('utf-8'), default=None) is not None
                if self.logger:
                    self.logger.debug(f"Key '{key}' exists: {exists}")
                return exists
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to check existence of key '{key}': {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        Delete a specific key from the database.
        Args:
            key (str): The key to delete.
        Returns:
            bool: True if deletion was successful, False if key not found or error occurred.
        """
        try:
            with self.env.begin(write=True) as txn:
                success = txn.delete(key.encode('utf-8'))
                if success:
                    if self.logger:
                        self.logger.info(f"Successfully deleted key '{key}'")
                else:
                    if self.logger:
                        self.logger.warning(f"Key '{key}' not found for deletion")
                return success
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to delete key '{key}': {e}")
            return False
            
    def get_stats(self) -> dict:
        """
        Return database statistics.
        Returns:
            dict: Dictionary containing database statistics.
        """
        try:
            with self.env.begin() as txn:
                stats = txn.stat()
                if self.logger:
                    self.logger.debug(f"Database stats retrieved: {stats}")
                return stats
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to get database stats: {e}")
            return {}
            
    def _close_log_handlers(self):
        """
        Internal helper to close and remove all log handlers.

        Safely closes file handlers and removes them from the logger to prevent
        resource leaks. Errors during cleanup are silently ignored.
        """
        try:
            if hasattr(self, 'logger') and self.logger and self.logger.handlers:
                for handler in self.logger.handlers[:]:
                    handler.close()
                    self.logger.removeHandler(handler)
        except Exception:
            # Silently ignore errors during cleanup
            pass

    def close(self):
        """
        Closes the LMDB environment and releases all resources.

        This method should be called when you're done with the storage to ensure
        proper cleanup of the database connection and log file handlers.
        If using a context manager, this is called automatically.

        Note:
            After calling close(), the storage instance should not be used.
        """
        try:
            self.env.close()
            if self.logger:
                self.logger.info("LMDB environment closed successfully")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error closing LMDB environment: {e}")

        self._close_log_handlers()

    def __del__(self):
        """
        Destructor to ensure proper cleanup if close() wasn't called explicitly.

        This is a safety mechanism that automatically closes the database connection
        and log handlers when the object is garbage collected. However, it's best
        practice to explicitly call close() or use a context manager.

        Note:
            Errors during cleanup are silently ignored to prevent issues during
            garbage collection.
        """
        try:
            if hasattr(self, 'env') and self.env:
                self.env.close()
        except Exception:
            # Silently ignore errors during cleanup
            pass

        self._close_log_handlers()

    def export_to_json(self, json_file_path: str) -> bool:
        """
        Exports all key-value pairs to a JSON file.

        The JSON file will contain an array of objects with 'key' and 'value' fields.
        Values are automatically decompressed during export.

        Args:
            json_file_path (str): The path to the output JSON file.

        Returns:
            bool: True if the export was successful, False otherwise.

        Example:
            >>> storage = LMDBStorage()
            >>> storage.put("config:theme", "dark")
            >>> storage.put("config:lang", "en")
            >>> storage.export_to_json("config_backup.json")
            True

            The resulting JSON file will contain:
            [
                {"key": "config:theme", "value": "dark"},
                {"key": "config:lang", "value": "en"}
            ]
        """
        data = []
        try:
            with self.env.begin() as txn:
                cursor = txn.cursor()
                for key_bytes, stored_value in cursor.iternext():
                    key = key_bytes.decode('utf-8')
                    value = self._decode_value(stored_value)
                    if value is not None:
                        data.append({"key": key, "value": value})

            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)

            if self.logger:
                self.logger.info(f"Successfully exported {len(data)} entries to '{json_file_path}'")
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to export data to JSON: {e}")
            return False

    def import_from_json(self, json_file_path: str) -> bool:
        """
        Imports key-value pairs from a JSON file and stores them in the database.

        The JSON file should contain an array of objects with 'key' and 'value' fields.
        Invalid entries are skipped and logged. Values are automatically compressed
        if they exceed the compression threshold.

        Args:
            json_file_path (str): The path to the input JSON file.

        Returns:
            bool: True if the import completed (even with some failures), False if
                  the JSON file couldn't be loaded or parsed.

        Example:
            Given a JSON file 'data.json':
            [
                {"key": "user:1", "value": "Alice"},
                {"key": "user:2", "value": "Bob"}
            ]

            >>> storage = LMDBStorage()
            >>> storage.import_from_json("data.json")
            True
            >>> storage.get("user:1")
            'Alice'
        """
        # Load JSON data from file
        data = self._load_json_file(json_file_path)
        if data is None:
            return False

        # Validate and import entries
        imported_count, failed_count = self._import_entries(data)

        # Log results
        if self.logger:
            self.logger.info(f"Successfully imported {imported_count} entries from '{json_file_path}'")
            if failed_count > 0:
                self.logger.warning(f"Failed to import {failed_count} entries")

        return True

    def _load_json_file(self, json_file_path: str) -> Optional[list]:
        """
        Internal helper to load and validate JSON file.
        Args:
            json_file_path (str): Path to JSON file.
        Returns:
            list or None: Loaded data if successful, None otherwise.
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                if self.logger:
                    self.logger.error(f"JSON file '{json_file_path}' does not contain a list of objects.")
                return None

            return data
        except FileNotFoundError:
            if self.logger:
                self.logger.error(f"JSON file not found: {json_file_path}")
            return None
        except json.JSONDecodeError as e:
            if self.logger:
                self.logger.error(f"Failed to parse JSON file '{json_file_path}': {e}")
            return None
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to load JSON file: {e}")
            return None

    def _import_entries(self, data: list) -> tuple:
        """
        Internal helper to validate and import entries from parsed JSON data.
        Args:
            data (list): List of dictionaries with 'key' and 'value' fields.
        Returns:
            tuple: (imported_count, failed_count)
        """
        imported_count = 0
        failed_count = 0

        for item in data:
            if not isinstance(item, dict) or 'key' not in item or 'value' not in item:
                if self.logger:
                    self.logger.warning(f"Skipping invalid entry in JSON file: {item}")
                failed_count += 1
                continue

            # Use the put method to handle validation, compression, and storage
            if self.put(item['key'], item['value']):
                imported_count += 1
            else:
                failed_count += 1

        return imported_count, failed_count

            
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

# Example usage:
if __name__ == "__main__":
    # Define a sample JSON file to import
    sample_json_data = [
        {"key": "product_1", "value": "Laptop, 16GB RAM, 512GB SSD"},
        {"key": "product_2", "value": "Monitor, 27-inch, 4K resolution"},
        {"key": "long_description", "value": "This is a very long text to demonstrate both import and export functionality, and how large values are handled by the compression logic. " * 50}
    ]
    sample_json_path = "products_to_import.json"

    # Write the sample data to a JSON file
    with open(sample_json_path, 'w', encoding='utf-8') as f:
        json.dump(sample_json_data, f, indent=4)
    print(f"Created a sample JSON file at '{sample_json_path}' for import.")

    # Create storage instance with a 50MB capacity and logging enabled
    with LMDBStorage("storage_with_json.lmdb", capacity_mb=50) as storage:
        # Import data from the JSON file
        print("\nImporting data from JSON file...")
        storage.import_from_json(sample_json_path)

        # Retrieve a value to show it's been imported correctly
        retrieved_value = storage.get("product_1")
        print(f"\nRetrieved value for 'product_1': {retrieved_value}")
        print(f"Number of keys after import: {storage.num_keys()}")

        # Export all data to a new JSON file
        exported_json_path = "exported_products.json"
        print(f"\nExporting all data to '{exported_json_path}'...")
        storage.export_to_json(exported_json_path)
        print(f"Data exported successfully. You can inspect '{exported_json_path}'.")

        # Now let's try to export the original data again after clearing
        print("\nClearing database and re-exporting to demonstrate empty file...")
        storage.clear()
        storage.export_to_json(exported_json_path)
        print(f"Database cleared and exported. '{exported_json_path}' should now be an empty list.")

    print("\nScript finished. Check the generated LMDB file and JSON files.")

