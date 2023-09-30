import gzip
from pathlib import Path
from typing import Generator

from logzero import logger


def read_lines(file_path: str, print_log: bool = True) -> Generator[str, None, None]:
    """Read lines from a file.

    Args:
        file_path (str): File path.
        print_log (bool, optional): Print log. Defaults to True.
    Yields:
        Generator[str, None, None]: Generator of lines.
    """
    assert Path(file_path).exists(), f"Not found: {file_path}"
    if print_log:
        logger.info(f"Load: {file_path}")

    if file_path.endswith(".gzip") or file_path.endswith(".gz"):
        with gzip.open(filename=file_path, mode="rt", encoding="utf_8") as fi:
            for line in fi:
                yield line.rstrip("\n")

    else:
        with open(file_path, errors="ignore") as fi:
            for line in fi:
                yield line.rstrip("\n")


def count_n_lines(file_path: str) -> int:
    """Count the number of lines in a file.

    Args:
        file_path (str): File path.
    Returns:
        int: The number of lines.
    """
    for idx, _ in enumerate(read_lines(file_path, print_log=False), 1):
        pass
    return idx
