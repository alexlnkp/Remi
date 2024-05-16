
from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import List

def grab_dataset(path: Path) -> List[str]:
    """
    This function reads and returns the contents of files in the specified path.

    Args:
        `path`: `Path` - The path to the directory containing the files.

    Returns:
        `List`[`str`] - A list of strings, each representing the content of a file.
    """
    # Read the contents of each file in the directory
    return [
        open(path, "r").read() + "\n"
        for path in [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    ]

def preprocess_dataset(datasets: List[str], separator: str) -> List[List[str]]:
    """
    Separate each conversation by the separator.

    Args:
        `dataset`: `List`[`str`] - A list of strings, each representing the content of a file.

    Returns:
        `List`[`str`] - A list of strings, each representing the content of a file without separators.
    """
    # Separate each conversation by the separator
    return [d for d in [dataset.split(separator) for dataset in datasets]]