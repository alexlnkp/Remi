import argparse
import os

import cursor
import regex as re
import torch
from transformers import BitsAndBytesConfig

NF4_CONF = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)


def check_gpu() -> None:
    print("GPU is available" if torch.cuda.is_available() else "GPU is not available")


def collect_user_input(user_input_text: str) -> str:
    cursor.show()
    usr_input: str = str(input(user_input_text))
    cursor.hide()
    return usr_input


def argument_init() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--debug", "-d", action="store_true")
    return argparser


def clear_terminal() -> None:
    if os.name != "nt":
        print("\033c", end="")
        return
    # trunk-ignore(bandit/B605)
    # trunk-ignore(bandit/B607)
    os.system("cls")


def decode_response(tokenizer, generated_ids, assistant_name: str) -> tuple[str, str]:
    response: str = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    index: str = re.findall(r"{}: (.*)".format(assistant_name), response)[-1]
    return response, index
