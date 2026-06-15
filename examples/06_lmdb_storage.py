"""Example: Key-value storage with LMDBStorage."""

from litekit.lmdb_storage import LMDBStorage

with LMDBStorage(db_path="/tmp/example.lmdb", capacity_mb=10) as db:
    db.put("greeting", "Hello, world!")
    db.put("answer", "42")

    print("greeting:", db.get("greeting"))
    print("answer:", db.get("answer"))
    print("answer exists?", db.exists("answer"))

    db.delete("answer")
    print("answer after delete:", db.get("answer"))
    print("num keys:", db.num_keys())
    print("all keys:", db.get_keys())

    db.export_to_json("/tmp/example_export.json")
    db.clear()
    print("keys after clear:", db.get_keys())

    db.import_from_json("/tmp/example_export.json")
    print("keys after import:", db.get_keys())
