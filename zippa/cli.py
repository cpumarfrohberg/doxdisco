# cli.py
import zipfile
from pathlib import Path

import typer

from .main import extract_items, pack_items
from .utils import read_zipignore

VERBOSE_OPTION = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
DEFAULT_ZIP_NAME = "compressed.zip"

app = typer.Typer()


@app.command()
def pack(
    items: list[str] = typer.Argument(..., help="Files or directories to zip"),
    output: str = typer.Option(
        DEFAULT_ZIP_NAME, "--output", "-o", help="Output zip file"
    ),
    exclude: list[str] = typer.Option(
        [], "--exclude", "-x", help="Additional file patterns to exclude"
    ),
    exclude_file: str = typer.Option(
        ".zipignore", "--exclude-file", help="Path to .zipignore file"
    ),
    compress_level: int = typer.Option(
        3, "--compress-level", "-c", help="Compression level (0-9)", min=0, max=9
    ),
    ask_overwrite: bool = typer.Option(
        False, "--ask", help="Ask before overwriting existing files"
    ),
    force_overwrite: bool = typer.Option(
        False, "--force-overwrite", "-f", help="Force overwrite without asking"
    ),
    verbose: bool = VERBOSE_OPTION,
):
    """Zip directories and files with exclusion support"""

    zipignore_patterns = read_zipignore(exclude_file)

    # Combine with command-line exclusions
    all_exclude_patterns = zipignore_patterns + exclude

    if force_overwrite:
        overwrite = True
    elif ask_overwrite:
        overwrite = typer.confirm(f"Output file '{output}' already exists. Overwrite?")
    else:
        overwrite = False

    # Consume the generator to execute the packing
    for message in pack_items(
        items,
        Path(output),
        all_exclude_patterns,
        compress_level,
        overwrite=overwrite,
    ):
        if verbose:
            typer.echo(f"  {message}")

    if verbose:
        with zipfile.ZipFile(output, "r") as zip_ref:
            total_size = sum(
                info.file_size for info in zip_ref.infolist() if not info.is_dir()
            )
            compressed_size = sum(
                info.compress_size for info in zip_ref.infolist() if not info.is_dir()
            )
            compression_ratio = (
                (1 - compressed_size / total_size) * 100 if total_size > 0 else 0
            )

            typer.echo("Compression results:")
            typer.echo(f"  Original size: {total_size:,} bytes")
            typer.echo(f"  Compressed size: {compressed_size:,} bytes")
            typer.echo(f"  Compression ratio: {compression_ratio:.1f}%")


@app.command()
def extract(
    source: str = typer.Argument(..., help="Zip file to extract"),
    item_name: str = typer.Option(
        None, "--item-name", "-i", help="Item name to extract"
    ),
    ask_overwrite: bool = typer.Option(
        False, "--ask", help="Ask before overwriting existing files"
    ),
    force_overwrite: bool = typer.Option(
        False, "--force-overwrite", "-f", help="Force overwrite without asking"
    ),
    verbose: bool = VERBOSE_OPTION,
):
    """Extract items from a zip file"""

    if verbose:
        typer.echo("Warning: Both --verbose and --quiet specified. Using verbose mode.")

    # Handle overwrite logic at CLI level
    overwrite = force_overwrite

    if not overwrite and ask_overwrite:
        overwrite = typer.confirm("Some files may already exist. Overwrite?")

    extract_items(
        zip_file=Path(source),
        item_name=item_name,
        overwrite=overwrite,
    )


@app.command()
def list(
    source: str = typer.Argument(..., help="Zip file to list contents"),
    verbose: bool = VERBOSE_OPTION,
):
    """List contents of a zip file"""

    if verbose:
        typer.echo(f"Listing contents of {source}")

    # TODO: Implement list functionality using ZipArchiveManager
    # This would show the contents without extracting
    typer.echo(f"Contents of {source}:")
    typer.echo("(List functionality not yet implemented)")


if __name__ == "__main__":
    app()
