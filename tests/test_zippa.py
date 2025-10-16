import zipfile
from pathlib import Path

import pytest

from zippa.main import extract_items, pack_items
from zippa.utils import read_zipignore


def _assert_zip_contents(zip_path, expected_files, excluded_files):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        file_list = zip_ref.namelist()

        assert all(
            expected_file in file_list for expected_file in expected_files
        ), f"Missing expected files: {[f for f in expected_files if f not in file_list]}"

        assert all(
            excluded_file not in file_list for excluded_file in excluded_files
        ), f"Unexpected files found: {[f for f in excluded_files if f in file_list]}"


@pytest.fixture
def temp_cwd(monkeypatch):
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        monkeypatch.chdir(temp_dir)
        yield Path(temp_dir)


@pytest.fixture
def test_files_in_cwd(temp_cwd):
    (temp_cwd / "lorem.md").write_text("Lorem ipsum dolor sit amet...")
    (temp_cwd / "leo.md").write_text("Test content")
    (temp_cwd / "dummy_dir").mkdir()
    (temp_cwd / "dummy_dir" / "notes.txt").write_text("Test notes")
    (temp_cwd / "dummy_dir" / "subdir").mkdir()
    (temp_cwd / "dummy_dir" / "subdir" / "nested_file.txt").write_text("nested content")
    return temp_cwd


@pytest.fixture
def symlink_test_files(temp_cwd):
    """Create test files with various symlink scenarios"""
    import os

    (temp_cwd / "original.txt").write_text("Original content")
    (temp_cwd / "normal.txt").write_text("Normal content")
    (temp_cwd / "real_dir").mkdir()
    (temp_cwd / "real_dir" / "file_in_dir.txt").write_text("Content in real dir")

    os.symlink("original.txt", temp_cwd / "file_symlink.txt")
    os.symlink("nonexistent.txt", temp_cwd / "broken_symlink.txt")
    os.symlink("real_dir", temp_cwd / "dir_symlink")
    os.symlink("file_b.txt", temp_cwd / "file_a.txt")
    os.symlink("file_a.txt", temp_cwd / "file_b.txt")
    os.symlink("../../../nonexistent_file.txt", temp_cwd / "escape_symlink.txt")

    return temp_cwd


@pytest.fixture
def test_files(tmp_path):
    (tmp_path / "lorem.md").write_text("Lorem ipsum dolor sit amet...")
    (tmp_path / "leo.md").write_text("Test content")
    (tmp_path / "dummy_dir").mkdir()
    (tmp_path / "dummy_dir" / "notes.txt").write_text("Test notes")
    (tmp_path / "dummy_dir" / "subdir").mkdir()
    (tmp_path / "dummy_dir" / "subdir" / "nested_file.txt").write_text("nested content")
    return tmp_path


@pytest.fixture
def test_zip(tmp_path):
    test_zip = tmp_path / "test.zip"
    with zipfile.ZipFile(test_zip, "w") as zip_ref:
        zip_ref.writestr("file1.txt", "content1")
        zip_ref.writestr("file2.txt", "content2")
        zip_ref.writestr("dir/", "")
        zip_ref.writestr("dir/file3.txt", "content3")
    return test_zip


@pytest.mark.parametrize(
    "compress_level,include_dirs,expected_files,excluded_files",
    [
        (
            3,
            True,
            [
                "lorem.md",
                "leo.md",
                "dummy_dir/notes.txt",
                "dummy_dir/subdir/nested_file.txt",
                "dummy_dir/subdir/",
                "dummy_dir/",
            ],
            ["test_zippa.py", "__init__.py", "test_output.zip", "__pycache__"],
        ),
        (
            0,
            True,
            [
                "lorem.md",
                "leo.md",
                "dummy_dir/notes.txt",
                "dummy_dir/subdir/nested_file.txt",
                "dummy_dir/subdir/",
                "dummy_dir/",
            ],
            ["test_zippa.py", "__init__.py", "test_output.zip", "__pycache__"],
        ),
        (
            9,
            True,
            [
                "lorem.md",
                "leo.md",
                "dummy_dir/notes.txt",
                "dummy_dir/subdir/nested_file.txt",
                "dummy_dir/subdir/",
                "dummy_dir/",
            ],
            ["test_zippa.py", "__init__.py", "test_output.zip", "__pycache__"],
        ),
        (
            3,
            False,
            [
                "lorem.md",
                "leo.md",
                "dummy_dir/notes.txt",
                "dummy_dir/subdir/nested_file.txt",
            ],
            [
                "test_zippa.py",
                "__init__.py",
                "test_output.zip",
                "__pycache__",
                "dummy_dir/subdir/",
            ],
        ),
    ],
    ids=[
        "Default compression with directories",
        "No compression with directories",
        "Maximum compression with directories",
        "Default compression without directories",
    ],
)
def test_pack_items(
    compress_level, include_dirs, expected_files, excluded_files, test_files_in_cwd
):
    output_zip = test_files_in_cwd / f"test_output_{compress_level}_{include_dirs}.zip"

    zipignore_path = Path(__file__).parent.parent / ".zipignore"
    exclude_patterns = (
        read_zipignore(str(zipignore_path)) if zipignore_path.exists() else []
    )

    print(f"Exclude patterns: {exclude_patterns}")
    print(f"Zipignore path: {zipignore_path}")
    print(f"Zipignore exists: {zipignore_path.exists()}")

    list(
        pack_items(
            items=["lorem.md", "leo.md", "dummy_dir"],
            output_zip=output_zip,
            exclude_patterns=exclude_patterns,
            compress_level=compress_level,
            include_dirs=include_dirs,
            overwrite=False,
        )
    )

    assert output_zip.exists()
    assert output_zip.stat().st_size > 0
    _assert_zip_contents(output_zip, expected_files, excluded_files)


def test_assert_zip_contents(test_zip):
    _assert_zip_contents(
        test_zip,
        expected_files=["file1.txt", "file2.txt", "dir/file3.txt", "dir/"],
        excluded_files=["nonexistent.txt"],
    )


def test_extract_items(tmp_path):
    (tmp_path / "file1.txt").write_text("content1")
    (tmp_path / "file2.txt").write_text("content2")

    test_zip = tmp_path / "test.zip"
    with zipfile.ZipFile(test_zip, "w") as zip_ref:
        zip_ref.write(tmp_path / "file1.txt", "file1.txt")
        zip_ref.write(tmp_path / "file2.txt", "file2.txt")

    extract_items(zip_file=test_zip, overwrite=False)

    assert (tmp_path / "file1.txt").exists()
    assert (tmp_path / "file2.txt").exists()
    assert (tmp_path / "file1.txt").read_text() == "content1"
    assert (tmp_path / "file2.txt").read_text() == "content2"


def test_extract_empty_zip(tmp_path):
    empty_zip = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty_zip, "w"):
        pass  # Empty zip

    extract_items(zip_file=empty_zip, overwrite=False)


def test_extract_directories_only(temp_cwd):
    dirs_zip = temp_cwd / "dirs.zip"
    with zipfile.ZipFile(dirs_zip, "w") as zip_ref:
        zip_ref.writestr("dir1/", "")
        zip_ref.writestr("dir2/subdir/", "")

    extract_items(zip_file=dirs_zip, overwrite=False)
    assert (temp_cwd / "dir1").exists()
    assert (temp_cwd / "dir1").is_dir()
    assert (temp_cwd / "dir2").exists()
    assert (temp_cwd / "dir2").is_dir()
    assert (temp_cwd / "dir2" / "subdir").exists()
    assert (temp_cwd / "dir2" / "subdir").is_dir()


def test_pack_special_characters(temp_cwd):
    special_files = [
        "file with spaces.txt",
        "file-with-dashes.txt",
        "file_with_underscores.txt",
        "file.with.dots.txt",
        "файл.txt",  # Unicode
        "file[1].txt",  # Brackets
    ]

    for filename in special_files:
        (temp_cwd / filename).write_text("content")

    output_zip = temp_cwd / "special.zip"
    list(pack_items(special_files, output_zip, [], 3))

    assert output_zip.exists()
    with zipfile.ZipFile(output_zip, "r") as zip_ref:
        zip_files = zip_ref.namelist()
        for filename in special_files:
            assert filename in zip_files


@pytest.mark.parametrize(
    "test_case,items,expected_files,should_fail",
    [
        ("file_symlink", ["original.txt", "file_symlink.txt"], ["original.txt"], False),
        ("broken_symlink", ["normal.txt", "broken_symlink.txt"], ["normal.txt"], True),
        ("dir_symlink", ["real_dir", "dir_symlink"], ["real_dir/"], False),
        ("circular_symlink", ["file_a.txt", "file_b.txt"], [], True),
        ("escape_symlink", ["normal.txt", "escape_symlink.txt"], ["normal.txt"], True),
    ],
    ids=[
        "file_symlink",
        "broken_symlink",
        "dir_symlink",
        "circular_symlink",
        "escape_symlink",
    ],
)
def test_pack_items_symlinks(
    symlink_test_files, test_case, items, expected_files, should_fail
):
    """Test packing files with different types of symlinks"""
    output_zip = symlink_test_files / f"symlinks_{test_case}.zip"

    if should_fail:
        with pytest.raises(FileNotFoundError):
            # Just iterate through the generator - exception happens during iteration
            for _ in pack_items(
                items=items,
                output_zip=output_zip,
                exclude_patterns=[],
                compress_level=3,
                include_dirs=True,
                overwrite=False,
            ):
                pass
    else:
        list(
            pack_items(
                items=items,
                output_zip=output_zip,
                exclude_patterns=[],
                compress_level=3,
                include_dirs=True,
                overwrite=False,
            )
        )

        assert output_zip.exists()

        zip_files = zipfile.ZipFile(output_zip).namelist()
        print(f"{test_case} - Files in zip: {zip_files}")

        for expected_file in expected_files:
            assert expected_file in zip_files
