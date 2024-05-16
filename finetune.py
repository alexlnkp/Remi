#!.venv/bin/python

from typing import List

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model

from ft.lib import grab_dataset, preprocess_dataset
from infer.utils import (
    check_gpu,
    clear_cuda_cache,
    decode_response,
    get_model_and_tokenizer,
)

DATASET_PATH = "ft/xlx_ft_dataset"
SEPARATOR_CHAR = "-----------------------------------------------------------------\n"

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
    task_type=TaskType.CAUSAL_LM
    # lora_alpha=32,
    # lora_dropout=0.1,
)

if __name__ == "__main__":
    datasets_contents: List[str] = grab_dataset(DATASET_PATH)
    dataset: List[List[str]] = preprocess_dataset(datasets_contents, SEPARATOR_CHAR)

    cuda_available: bool = check_gpu()
    print("Cuda available" if cuda_available else "Cuda not available")

    model, tokenizer = get_model_and_tokenizer("JosephusCheung/LL7M")
    model = get_peft_model(model, LORA_CONF)
