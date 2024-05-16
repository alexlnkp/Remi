import torch
from peft.peft_model import PeftModelForCausalLM
from torch.utils.data import DataLoader


def train_epoch(
    model: PeftModelForCausalLM,
    train_data: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0
    for batch in train_data:
        input_ids = batch.to(device)

        optimizer.zero_grad()
        outputs = model.generate(**input_ids, max_length=4096)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_data)
    return avg_loss


def evaluate(
    model: PeftModelForCausalLM, val_data: DataLoader, device: torch.device
) -> float:
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_data:
            input_ids = batch.to(device)

            outputs = model.generate(**input_ids, max_length=4096)
            loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(val_data)
    return avg_loss
