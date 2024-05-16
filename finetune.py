#!.venv/bin/python

from typing import List

from peft import (
    LoraConfig,
    PeftModelForCausalLM,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from sklearn.model_selection import train_test_split
from torch import device as torch_device
from torch.utils.data import DataLoader
from transformers import (
    BatchEncoding,
    LlamaForCausalLM,
    LlamaTokenizer,
    Trainer,
    TrainingArguments,
)

from ft.lib import (
    RemiDataset,
    dictionarize_dataset,
    get_input_and_target_list,
    grab_dataset,
    merge_datasets,
    preprocess_dataset,
)
from infer.utils import check_gpu, get_model_and_tokenizer

DATASET_PATH = "ft/xlx_ft_dataset"
SEPARATOR_CHAR = "-----------------------------------------------------------------\n"

# region Hyperparams

LEARNING_RATE = 2.5e-5
TEST_SIZE = 0.1
BATCH_SIZE = 8
EPOCHS = 10
MAX_SEQUENCE_LENGTH = 2048

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

    print(f"Cuda {'not ' if not cuda_available else ''}available")
    device: torch_device = "cuda" if cuda_available else "cpu"

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
    _model: LlamaForCausalLM
    tokenizer: LlamaTokenizer

    _model, tokenizer = get_model_and_tokenizer("JosephusCheung/LL7M")

    _model.gradient_checkpointing_enable()
    _model = prepare_model_for_kbit_training(_model)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model: PeftModelForCausalLM = get_peft_model(_model, LORA_CONF)

    x_train_encodings: BatchEncoding = (
        tokenizer.batch_encode_plus(
            x_train,
            return_tensors="pt",
            max_length=MAX_SEQUENCE_LENGTH,
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
            max_length=MAX_SEQUENCE_LENGTH,
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
            max_length=MAX_SEQUENCE_LENGTH,
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
            max_length=MAX_SEQUENCE_LENGTH,
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

    # x_train_encodings = y_train_encodings = 10
    train_data: RemiDataset = RemiDataset(x_train_encodings, y_train_encodings)

    # x_test_encodings = y_test_encodings = 2
    val_data: RemiDataset = RemiDataset(x_test_encodings, y_test_encodings)

    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_data_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    training_args = TrainingArguments(
        output_dir="./ft/results",
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

    model.save_pretrained("help")
