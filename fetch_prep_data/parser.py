# GitHub data parsing utilities with bank-grade security validation
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

import frontmatter

from config import FileProcessingConfig, GitHubConfig

from .reader import RawRepositoryFile

MAX_FILES = GitHubConfig.MAX_FILES.value


def parse_data(data_raw: list[RawRepositoryFile]) -> list[dict[str, Any]]:
    """
    Parse raw GitHub repository files into structured data with security validation.
    Handles files with invalid frontmatter by skipping frontmatter and using content only.
    Converts datetime and date objects to ISO format strings for JSON compatibility.

    Args:
        data_raw: List of RawRepositoryFile objects from GitHub

    Returns:
        List of parsed document dictionaries with frontmatter metadata

    Raises:
        ValueError: If input validation fails
    """

    if len(data_raw) > MAX_FILES:
        raise ValueError(f"Too many files: {len(data_raw)} (max: {MAX_FILES})")

    data_parsed = []
    for f in data_raw:
        if len(f.content) > FileProcessingConfig.MAX_CONTENT_SIZE.value:
            print(f"⚠️  Skipping oversized file {f.filename}: {len(f.content)} bytes")
            continue

        try:
            post = frontmatter.loads(f.content)
            data = post.to_dict()
            data["filename"] = f.filename

            # Convert datetime and date objects to ISO format strings for JSON compatibility
            data = _convert_datetime_to_string(data)
            data_parsed.append(data)

        except (frontmatter.FrontMatterError, UnicodeDecodeError) as e:
            print(f"⚠️  Skipping frontmatter for {f.filename}: {str(e)[:50]}...")

            # Create document with just content and filename
            data = {
                "content": f.content,
                "filename": f.filename,
                "title": "",
                "description": "",
            }
            data_parsed.append(data)

    return data_parsed


def _convert_datetime_to_string(obj: Any) -> Any:
    """
    Recursively convert datetime and date objects to ISO format strings.

    Handles nested dictionaries, lists, and other sequences by recursively
    traversing the data structure and converting any datetime/date objects found.

    Args:
        obj: Any Python object that may contain datetime or date objects

    Returns:
        Object with datetime and date objects converted to ISO strings

    Example:
        >>> data = {"key": datetime.now(), "items": [{"created": datetime.now()}]}
        >>> _convert_datetime_to_string(data)
        {"key": "2024-01-01T12:00:00", "items": [{"created": "2024-01-01T12:00:00"}]}
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Mapping):
        return {k: _convert_datetime_to_string(v) for k, v in obj.items()}
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [_convert_datetime_to_string(x) for x in obj]
    return obj
