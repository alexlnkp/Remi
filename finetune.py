#!.venv/bin/python

from typing import List

from sklearn.model_selection import train_test_split
from torch import device as torch_device
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

from Remi.ft import (
    LORA_CONF,
    RemiDataset,
    dictionarize_dataset,
    get_input_and_target_list,
    grab_dataset,
    merge_datasets,
    prepare_encodings,
    prepare_tokenizer_and_peft_model,
    preprocess_dataset,
)
from Remi.infer.utils import check_gpu

DATASET_PATH = "ft/xlx_ft_dataset"
SEPARATOR_CHAR = "-----------------------------------------------------------------\n"

# region Hyperparams

LEARNING_RATE = 2.5e-5
TEST_SIZE = 0.1
BATCH_SIZE = 4
EPOCHS = 20
MAX_SEQUENCE_LENGTH = 2048

# endregion


if __name__ == "__main__":
    cuda_available: bool = check_gpu()

    print(f"Cuda {'not ' if not cuda_available else ''}available")
    device: torch_device = (
        torch_device("cuda") if cuda_available else torch_device("cpu")
    )

    f_dataset: List[str] = merge_datasets(
        preprocess_dataset(grab_dataset(DATASET_PATH), SEPARATOR_CHAR)
    )

    x_dataset, y_dataset = get_input_and_target_list(dictionarize_dataset(f_dataset))

    x_train: List[str] = []
    x_test: List[str] = []
    y_train: List[str] = []
    y_test: List[str] = []

    x_train, x_test, y_train, y_test = train_test_split(
        x_dataset, y_dataset, test_size=TEST_SIZE, random_state=0
    )
    tokenizer, model = prepare_tokenizer_and_peft_model(
        "JosephusCheung/LL7M", LORA_CONF
    )

    x_train_encodings, x_test_encodings, y_train_encodings, y_test_encodings = (
        prepare_encodings(
            x_train,
            x_test,
            y_train,
            y_test,
            tokenizer,
            device,
            MAX_SEQUENCE_LENGTH,
        )
    )

    # x_train_encodings = y_train_encodings = 10
    train_data: RemiDataset = RemiDataset(x_train_encodings, y_train_encodings)

    # x_test_encodings = y_test_encodings = 2
    val_data: RemiDataset = RemiDataset(x_test_encodings, y_test_encodings)

    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_data_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    training_args = TrainingArguments(
        output_dir="./ft/ckpt",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_checkpointing=True,
        logging_dir="./ft/logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        save_steps=500,
        save_total_limit=2,
        fp16=cuda_available,
        use_cpu=not cuda_available,
        gradient_accumulation_steps=4,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
    )

    trainer.train()

    model.save_pretrained("remi")
