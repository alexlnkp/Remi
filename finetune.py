#!.venv/bin/python

from typing import List

from peft import LoraConfig, TaskType, get_peft_model
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ft.lib import grab_dataset, preprocess_dataset
from infer.utils import (
    check_gpu,
    clear_cuda_cache,
    decode_response,
    get_model_and_tokenizer,
)

DATASET_PATH = "ft/xlx_ft_dataset"
SEPARATOR_CHAR = "-----------------------------------------------------------------\n"

# region Hyperparams

LEARNING_RATE = 1e-4
TEST_SIZE = 0.2

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
    print("Cuda available" if cuda_available else "Cuda not available")

    datasets_contents: List[str] = grab_dataset(DATASET_PATH)
    datasets: List[List[str]] = preprocess_dataset(datasets_contents, SEPARATOR_CHAR)

    train_dataset, val_dataset = train_test_split(datasets, test_size=TEST_SIZE)

    model, tokenizer = get_model_and_tokenizer("JosephusCheung/LL7M")

    train_inputs = tokenizer(
        train_dataset, is_split_into_words=True, truncation=True, return_tensors="pt"
    ).to("cuda" if cuda_available else "cpu")
    val_inputs = tokenizer(
        val_dataset, is_split_into_words=True, truncation=True, return_tensors="pt"
    ).to("cuda" if cuda_available else "cpu")

    model = get_peft_model(model, LORA_CONF)

    train_data = DataLoader(train_inputs, batch_size=16, shuffle=True)
    val_data = DataLoader(val_inputs, batch_size=16, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
