#!.venv/bin/python

from typing import List

from peft import LoraConfig, TaskType, get_peft_model, PeftModelForCausalLM
from sklearn.model_selection import train_test_split
from torch import device as torch_device
from torch.optim import Optimizer
from torch.optim import AdamW

from ft.lib import grab_dataset, preprocess_dataset
from ft.train import evaluate, train_epoch, DataLoader
from infer.utils import (
    check_gpu,
    get_model_and_tokenizer,
)
from transformers import BatchEncoding

DATASET_PATH = "ft/xlx_ft_dataset"
SEPARATOR_CHAR = "-----------------------------------------------------------------\n"

# region Hyperparams

LEARNING_RATE = 1e-4
TEST_SIZE = 0.2
BATCH_SIZE = 1
EPOCHS = 10

# endregion

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
    # lora_alpha=32,
    # lora_dropout=0.1,
)


if __name__ == "__main__":
    cuda_available: bool = check_gpu()
    device: torch_device = "cuda" if cuda_available else "cpu"
    print("Cuda available" if cuda_available else "Cuda not available")

    datasets_contents: List[str] = grab_dataset(DATASET_PATH)
    datasets: List[List[str]] = preprocess_dataset(datasets_contents, SEPARATOR_CHAR)

    train_dataset, val_dataset = train_test_split(datasets, test_size=TEST_SIZE)

    model, tokenizer = get_model_and_tokenizer("JosephusCheung/LL7M")

    train_inputs: BatchEncoding = tokenizer(
        train_dataset, is_split_into_words=True, return_tensors="pt"
    ).to(device)
    val_inputs: BatchEncoding = tokenizer(
        val_dataset, is_split_into_words=True, return_tensors="pt"
    ).to(device)

    model: PeftModelForCausalLM = get_peft_model(model, LORA_CONF)

    train_data: DataLoader = DataLoader([train_inputs], batch_size=BATCH_SIZE, shuffle=True)
    val_data: DataLoader = DataLoader([val_inputs], batch_size=BATCH_SIZE, shuffle=False)

    optimizer: Optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        train_loss: float = train_epoch(model, train_data, optimizer, device)
        val_loss: float = evaluate(model, val_data, device)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} - Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f}"
        )
