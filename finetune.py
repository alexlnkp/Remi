#!.venv/bin/python

from typing import List

from peft import LoraConfig, PeftModelForCausalLM, TaskType, get_peft_model
from sklearn.model_selection import train_test_split
from torch import device as torch_device
from torch.utils.data import DataLoader
from transformers import BatchEncoding, Trainer, TrainingArguments

from ft.lib import grab_dataset, preprocess_dataset
from infer.utils import check_gpu, get_model_and_tokenizer

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
    device: torch_device = "cpu"
    if cuda_available:
        print("Cuda available")
        device = "cuda"
    else:
        print("Cuda not available")

    train_dataset, val_dataset = train_test_split(
        preprocess_dataset(grab_dataset(DATASET_PATH), SEPARATOR_CHAR),
        test_size=TEST_SIZE,
    )

    model, tokenizer = get_model_and_tokenizer("JosephusCheung/LL7M")

    train_inputs: BatchEncoding = tokenizer(
        train_dataset, is_split_into_words=True, return_tensors="pt"
    ).to(device)
    val_inputs: BatchEncoding = tokenizer(
        val_dataset, is_split_into_words=True, return_tensors="pt"
    ).to(device)

    model: PeftModelForCausalLM = get_peft_model(model, LORA_CONF)

    train_data: DataLoader = DataLoader(
        train_inputs, batch_size=BATCH_SIZE, shuffle=True
    )
    val_data: DataLoader = DataLoader(val_inputs, batch_size=BATCH_SIZE, shuffle=False)

    training_args = TrainingArguments(
        output_dir="./ft/results",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        logging_dir="./ft/logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        save_steps=500,
        save_total_limit=2,
        fp16=cuda_available,
        gradient_accumulation_steps=4,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained("help")
