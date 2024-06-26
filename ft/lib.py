from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModelForCausalLM,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import BatchEncoding, LlamaForCausalLM, LlamaTokenizer

from infer.utils import get_model_and_tokenizer

LORA_CONF = LoraConfig(
    r=8,
    target_modules=[
        "q_proj",
        "o_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=32,
    lora_dropout=0.1,
)


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
    return [
        d
        for d in [dataset.split(separator) for dataset in datasets if dataset]
        if d != ""
    ]


def merge_datasets(datasets: List[List[str]]) -> List[str]:
    """
    Merge a list of lists of strings into a single list of strings.

    Args:
        `datasets`: `List[List[str]]` - A list of lists of strings.

    Returns:
        `List[str]` - A list of strings, with each string being the concatenation
        of all the strings in the input lists.
    """
    # Initialize an empty list to store the merged dataset
    m_dataset: List[str] = []

    # Iterate over each dataset
    for dataset in datasets:
        # Iterate over each value in the dataset
        for value in dataset:
            # Append the value to the merged dataset
            m_dataset.append(value)

    # Return the merged dataset
    return m_dataset


def dictionarize_dataset(dataset: List[str]) -> List[Dict[str, str]]:
    """
    Converts a list of strings into a list of dictionaries, where each dictionary
    contains the input and target strings.

    Args:
        `dataset`: `List[str]` - A list of strings, each representing a conversation.

    Returns:
        `List[Dict[str, str]]` - A list of dictionaries, where each dictionary
        contains the input and target strings.
    """
    input_target: List[Dict[str, str]] = []

    # Iterate over each conversation
    for val in dataset:
        # Split the conversation string into input and target strings
        split: List[str] = val.split("## Response:\n")

        # Ensure the input string includes the delimiter
        split[0] += "## Response:\n"

        # Create a dictionary with the input and target strings
        input_target.append({split[0]: split[0] + split[-1].strip("\n")})

    return input_target


def get_input_and_target_list(
    input_target: List[Dict[str, str]]
) -> Tuple[List[str], List[str]]:
    """
    Extracts the input and target lists from the dictionaries in the input_target list.

    Args:
        `input_target`: `List[Dict[str, str]]` - A list of dictionaries, where each dictionary
        contains the input and target strings.

    Returns:
        `Tuple[List[str], List[str]]` - A tuple containing the input and target lists.
    """
    input_list: List[str] = []
    target_list: List[str] = []

    # Iterate over the input_target list
    for dict in input_target:
        # Extract the input and target strings from the dictionary
        for key, val in dict.items():
            input_list.append(key)
            target_list.append(val)

    # Return the input and target lists
    return input_list, target_list


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


class RemiDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for training and evaluating Remi.

    This class encapsulates the input and target data for the model.
    """

    def __init__(self, inputs: BatchEncoding, targets: BatchEncoding) -> None:
        """
        Initializes a RemiDataset object.

        Args:
            `inputs`: `BatchEncoding` - The input batch encoding.
            `targets`: `BatchEncoding` - The target batch encoding.
        """
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get an item from the dataset.

        Args:
            `idx`: `int` - The index of the item.

        Returns:
            `Dict`[`str`, `torch.Tensor`] - A dictionary containing the input_ids and labels.
        """
        # Get the input_ids tensor
        input_ids: torch.Tensor = (
            self.inputs["input_ids"][idx].clone().detach().squeeze()
        )

        # Get the target_ids tensor
        target_ids: torch.Tensor = (
            self.targets["input_ids"][idx].clone().detach().squeeze()
        )

        # Return the input_ids and labels tensors in a dictionary
        return {"input_ids": input_ids, "labels": target_ids}

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            `int` - The length of the dataset.
        """
        return len(self.inputs.input_ids)


def prepare_encodings(
    x_train: List[str],
    x_test: List[str],
    y_train: List[str],
    y_test: List[str],
    tokenizer: LlamaTokenizer,
    device: torch.device,
    max_sequence_length: int = 2048,
) -> tuple[BatchEncoding, BatchEncoding, BatchEncoding, BatchEncoding]:
    x_train_encodings: BatchEncoding = (
        tokenizer.batch_encode_plus(
            x_train,
            return_tensors="pt",
            max_length=max_sequence_length,
            padding="longest",
            truncation=True,
        )
        .convert_to_tensors()
        .to(device)
    )
    x_test_encodings: BatchEncoding = (
        tokenizer(
            x_test,
            return_tensors="pt",
            max_length=max_sequence_length,
            padding="longest",
            truncation=True,
        )
        .convert_to_tensors()
        .to(device)
    )

    y_train_encodings: BatchEncoding = (
        tokenizer(
            y_train,
            return_tensors="pt",
            max_length=max_sequence_length,
            padding="longest",
            truncation=True,
        )
        .convert_to_tensors()
        .to(device)
    )
    y_test_encodings: BatchEncoding = (
        tokenizer(
            y_test,
            return_tensors="pt",
            max_length=max_sequence_length,
            padding="longest",
            truncation=True,
        )
        .convert_to_tensors()
        .to(device)
    )
    x_train_encodings["input_ids"] = x_train_encodings["input_ids"].permute(1, 0)
    y_train_encodings["input_ids"] = y_train_encodings["input_ids"].permute(1, 0)
    x_test_encodings["input_ids"] = x_test_encodings["input_ids"].permute(1, 0)
    y_test_encodings["input_ids"] = y_test_encodings["input_ids"].permute(1, 0)

    return x_train_encodings, x_test_encodings, y_train_encodings, y_test_encodings


def prepare_tokenizer_and_peft_model(
    model_path_or_name: str, peft_conf: PeftConfig
) -> tuple[LlamaTokenizer, PeftModelForCausalLM]:
    _model: LlamaForCausalLM
    tokenizer: LlamaTokenizer

    _model, tokenizer = get_model_and_tokenizer(model_path_or_name)

    _model.gradient_checkpointing_enable()
    _model = prepare_model_for_kbit_training(_model)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model: PeftModelForCausalLM = get_peft_model(_model, peft_conf)
    return tokenizer, model
