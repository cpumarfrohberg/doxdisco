import hashlib
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

MAX_PICKLE_SIZE = 100_000_000  # 100MB


def validate_path(filepath: str) -> Path:
    path = Path(filepath).resolve()
    if ".." in str(path):
        raise ValueError("Invalid path")
    return path


def save_parsed_data(
    data: list[dict[str, Any]], filepath: str = "parsed_data.pkl"
) -> None:
    filepath = validate_path(filepath)

    filepath.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "data": data,
        "metadata": {
            "total_files": len(data),
            "saved_at": datetime.now().isoformat(),
            "version": "2.0",
        },
    }

    with open(filepath, "wb") as f:
        pickle.dump(save_data, f)

    _create_hash(filepath)
    print(f"✅ Saved {len(data)} documents to {filepath}")


def load_parsed_data(filepath: str = "parsed_data.pkl") -> list[dict[str, Any]]:
    filepath = validate_path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if filepath.stat().st_size > MAX_PICKLE_SIZE:
        raise ValueError("File too large")

    try:
        _verify_hash(filepath)
    except ValueError as e:
        raise ValueError(f"Failed to verify pickle file integrity: {e}") from e

    with open(filepath, "rb") as f:
        loaded = pickle.load(f)

    if not isinstance(loaded, dict) or "data" not in loaded:
        raise ValueError("Invalid pickle file")

    data = loaded["data"]
    if not isinstance(data, list):
        raise ValueError("Data must be a list")

    print(f"📚 Loaded {len(data)} documents from {filepath}")
    return data


def _create_hash(filepath: Path) -> None:
    hash_file = filepath.with_suffix(filepath.suffix + ".hash")
    with open(filepath, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    with open(hash_file, "w") as f:
        f.write(file_hash)


def _verify_hash(filepath: Path) -> None:
    hash_file = filepath.with_suffix(filepath.suffix + ".hash")

    if not hash_file.exists():
        raise ValueError("No integrity hash - refusing to load")

    with open(filepath, "rb") as f:
        current_hash = hashlib.sha256(f.read()).hexdigest()

    with open(hash_file, "r") as f:
        stored_hash = f.read().strip()

    if current_hash != stored_hash:
        raise ValueError("File integrity check failed")


def validate_data_structure(data: list[dict[str, Any]]) -> None:
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} must be a dictionary")

        if "filename" not in item or "content" not in item:
            raise ValueError(f"Item {i} missing required fields")

        if not isinstance(item["filename"], str) or not isinstance(
            item["content"], str
        ):
            raise ValueError(f"Item {i} has invalid types")
