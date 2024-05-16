#!.venv/bin/python

from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import List

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model

from infer.utils import (
    check_gpu,
    clear_cuda_cache,
    decode_response,
    get_model_and_tokenizer,
)

DATASET_PATH = "ft/xlx_ft_dataset"
SEPARATOR_CHAR = "-----------------------------------------------------------------\n"

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
    return [ds.split(separator) for ds in datasets]

datasets_contents: List[str] = grab_dataset(DATASET_PATH)
dataset: List[List[str]] = preprocess_dataset(datasets_contents, SEPARATOR_CHAR)

# LORA_CONF = LoraConfig(
#     r=8, target_modules=["query_key_value"],
#     bias="none",
#     task_type=TaskType.CAUSAL_LM,
#     lora_alpha=32, lora_dropout=0.1,
# )

# if __name__ == "__main__":
#     cuda_available: bool = check_gpu()
#     print("Cuda available" if cuda_available else "Cuda not available")

#     model, tokenizer = get_model_and_tokenizer("JosephusCheung/LL7M")
#     model = get_peft_model(model, LORA_CONF)
