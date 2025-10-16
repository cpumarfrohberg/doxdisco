# utils.py
import os
from pathlib import Path
from typing import List
from zipfile import ZIP_DEFLATED, ZipFile


def read_zipignore(zipignore_path: str = ".zipignore") -> List[str]:
    if not Path(zipignore_path).exists():
        return []  # Return empty list if .zipignore doesn't exist

    with open(zipignore_path, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


class WorkingDirectory:
    def __init__(self, target_path: str):
        self.path = Path(target_path)
        self.original_cwd = None

    def __enter__(self):
        if self.path:
            self.path = self.path.resolve()  # Resolve to absolute path

            if self.path.exists():
                if self.path.is_file():
                    raise NotADirectoryError(
                        f"'{self.path}' is a file, not a directory"
                    )
                elif not self.path.is_dir():
                    raise NotADirectoryError(
                        f"'{self.path}' is not a directory (might be a symlink, device, etc.)"
                    )
            else:
                raise NotADirectoryError(f"Path '{self.path}' does not exist")

            self.original_cwd = os.getcwd()
            os.chdir(self.path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_cwd:
            os.chdir(self.original_cwd)
        # Return False to re-raise exceptions, True to suppress them
        return False


class ZipArchiveManager:
    def __init__(self, zip_file: str, mode: str, compress_level: int):
        self.zip_file = Path(zip_file)
        self.mode = mode
        self.compress_level = compress_level
        self.archive = None

    @classmethod
    def for_compression(cls, zip_file: str, compress_level: int = 3):
        return cls(zip_file, "w", compress_level)

    @classmethod
    def for_extraction(cls, zip_file: str):
        return cls(zip_file, "r")

    def __enter__(self):
        if self.mode not in ("r", "w"):
            raise ValueError(
                f"Invalid mode '{self.mode}'. Mode must be 'r' (read) or 'w' (write)"
            )

        if self.mode == "w":
            self.archive = ZipFile(
                self.zip_file, "w", ZIP_DEFLATED, compresslevel=self.compress_level
            )
        else:
            if not self.zip_file.exists():
                raise FileNotFoundError(
                    f"Zip file '{self.zip_file}' not found for reading"
                )
            self.archive = ZipFile(self.zip_file, "r")

        return self.archive

    def __exit__(self, exc_type, _exc_val, _exc_tb):
        if self.archive:
            self.archive.close()
        if exc_type and self.mode == "w" and self.zip_file.exists():
            self.zip_file.unlink()
        return False
