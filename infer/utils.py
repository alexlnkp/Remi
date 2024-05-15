import argparse
import importlib.util
import os
from typing import Tuple

import cursor
import regex as re
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
)

NF4_CONF = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)


def get_model_and_tokenizer(model_name: str):
    cursor.hide()
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", quantization_config=NF4_CONF
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    return model, tokenizer


def get_uinput_and_response_format():
    if importlib.util.find_spec("colorama") is not None:
        from colorama import Fore, Style  # type: ignore
        from colorama import init as colorama_init  # type: ignore

        colorama_init(autoreset=True)
        USER_INPUT_TEXT: str = f"{Style.DIM}{Fore.BLUE}>{Style.RESET_ALL} "
        RESPONSE_META: str = f"{Style.BRIGHT}{Fore.LIGHTRED_EX}"
    else:
        print("Colorama is not available, will not use fancy output text... :(")
        USER_INPUT_TEXT: str = "> "
        RESPONSE_META: str = ""
    return USER_INPUT_TEXT, RESPONSE_META


def check_gpu() -> bool:
    if torch.cuda.is_available():
        print("GPU is available")
        return True
    print("GPU is not available")
    return False


def collect_user_input(user_input_text: str) -> str:
    cursor.show()
    usr_input: str = str(input(user_input_text))
    cursor.hide()
    return usr_input


def argument_init() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--debug", "-d", action="store_true")
    argparser.add_argument("--ignore-history", "-i", action="store_true")
    return argparser


def clear_terminal() -> None:
    if os.name != "nt":
        print("\033c", end="")
        return
    # trunk-ignore(bandit/B605)
    # trunk-ignore(bandit/B607)
    os.system("cls")


def decode_response(
    tokenizer: PreTrainedTokenizer, generated_ids: torch.Tensor, assistant_name: str
) -> Tuple[str, str]:
    """
    Decode the response from the model and extract the response index.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to decode the response.
        generated_ids (torch.Tensor): The generated IDs from the model.
        assistant_name (str): The name of the assistant.

    Returns:
        Tuple[str, str]: A tuple containing the decoded response and the response index.
    """
    response: str = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    index: str = re.findall(r"{}: (.*)".format(assistant_name), response)[-1]
    if re.match(r'(.*)"## History:" and "## Input:"(.*)', index):
        index = re.findall(r"## Response:\n(.*)", response)[-1]
    return response, index
