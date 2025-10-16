# main.py
import fnmatch
from pathlib import Path
from typing import Iterator, NamedTuple
from zipfile import ZIP_DEFLATED, ZipFile


class ZipItem(NamedTuple):
    item_type: str
    item_path: Path
    arcname: str


class ZipConfig(NamedTuple):
    exclude_patterns: list[str]
    include_dirs: bool = True


def _iter_zip_items(path: Path, config: ZipConfig):
    if path.is_file():
        relative_path = path.relative_to(Path.cwd())
        yield ZipItem("file", path, str(relative_path))
        return

    for item_type, item_path in _traverse_files_and_dirs(
        path, config.exclude_patterns, config.include_dirs
    ):
        relative_path = item_path.relative_to(Path.cwd())

        if item_type == "dir":
            yield ZipItem("dir", item_path, f"{relative_path}/")
        else:
            yield ZipItem("file", item_path, str(relative_path))


def _traverse_files_and_dirs(
    source_path: Path, exclude_patterns: list[str], include_dirs: bool = True
) -> Iterator[tuple[str, Path]]:
    if source_path.is_file():
        print(f"Processing file: {source_path}")
        yield ("file", source_path)
    elif source_path.is_dir():
        print(f"Processing subdirectory: {source_path}")

        # Add the root directory itself if include_dirs is True
        if include_dirs:
            print(f"  Adding root directory: {source_path}")
            yield ("dir", source_path)

        for item in source_path.rglob("*"):
            # Calculate relative path for pattern matching
            relative_path = item.relative_to(source_path)

            if any(
                fnmatch.fnmatch(item.name, p) or fnmatch.fnmatch(str(relative_path), p)
                for p in exclude_patterns
            ):
                continue

            if item.is_dir():
                if include_dirs:
                    print(f"  Adding directory: {item}")
                    yield ("dir", item)
            else:
                print(f"  Adding file: {item}")
                yield ("file", item)


def _validate_item(path: Path, item_name: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Item '{item_name}' does not exist")
    return path


def _validate_output_directory(output_zip: Path) -> None:
    output_dir = output_zip.parent
    if not output_dir.exists():
        raise NotADirectoryError(f"Output directory '{output_dir}' does not exist")
    if not output_dir.is_dir():
        raise NotADirectoryError(f"'{output_dir}' is not a directory")


def pack_items(
    items: list[str],
    output_zip: Path,
    exclude_patterns: list[str],
    compress_level: int,
    include_dirs: bool = True,
    overwrite: bool = False,
) -> Iterator[str]:
    yield f"Starting to pack {len(items)} items from current directory"

    # Validate output directory
    _validate_output_directory(output_zip)

    if output_zip.exists() and not overwrite:
        yield f"Output file {output_zip} already exists. Skipping."
        return

    with ZipFile(
        str(output_zip), "w", ZIP_DEFLATED, compresslevel=compress_level
    ) as archive:
        files_added, dirs_added = 0, 0
        for item_str in items:
            if item_str == ".":
                target_path = Path.cwd()
            else:
                target_path = Path.cwd() / item_str

            # Validate that the item exists
            _validate_item(target_path, item_str)

            yield f"Processing item: {item_str}"

            config = ZipConfig(
                exclude_patterns=exclude_patterns,
                include_dirs=include_dirs,
            )
            for zip_item in _iter_zip_items(target_path, config):
                if zip_item.item_type == "dir":
                    archive.writestr(zip_item.arcname, "")
                    dirs_added += 1
                    yield f"Added directory: {zip_item.item_path.name}"
                else:
                    archive.write(zip_item.item_path, arcname=zip_item.arcname)
                    files_added += 1
                    yield f"Added file: {zip_item.item_path.name}"

        archive.printdir()
        print(f"Added {files_added} files and {dirs_added} directories to {output_zip}")

        yield f"Completed: {files_added} files, {dirs_added} directories"


def _check_existing_files(archive: ZipFile, target_path: Path) -> list[str]:
    return [
        member.filename
        for member in archive.infolist()
        if not member.is_dir() and (target_path / member.filename).exists()
    ]


def extract_items(
    zip_file: Path,
    item_name: str | None = None,
    overwrite: bool = False,
) -> None:
    if not zip_file.exists():
        raise FileNotFoundError(f"Zip file '{zip_file}' does not exist")

    if not zip_file.is_file():
        raise ValueError(f"'{zip_file}' is not a file")

    target_path = Path.cwd()

    with ZipFile(str(zip_file), "r") as archive:
        existing_files = _check_existing_files(archive, target_path)
        if not overwrite and existing_files:
            print(
                f"Warning: {len(existing_files)} files already exist and would be overwritten."
            )
            print("Use --force-overwrite to overwrite or --ask to confirm each file.")
            return

        if item_name is None:
            archive.extractall(target_path)
            print(f"Extracted all items from {zip_file} to current directory")
        else:
            archive.extract(item_name, target_path)
            print(f"Extracted {item_name} from {zip_file} to current directory")
