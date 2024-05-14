#!.venv/bin/python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import regex as re
import os
import importlib.util
import cursor
import argparse

colorama_is_available: bool = importlib.util.find_spec("colorama") is not None
if colorama_is_available:
    from colorama import Fore, Back, Style, init as colorama_init # type: ignore
    colorama_init(autoreset=True)
    USER_INPUT_TEXT: str = f"{Style.DIM}{Fore.BLUE}>{Style.RESET_ALL} "
    RESPONSE_META:   str = f"{Style.BRIGHT}{Fore.LIGHTRED_EX}"
else:
    print("Colorama is not available, will not use fancy output text... :(")
    USER_INPUT_TEXT: str = "> "
    RESPONSE_META:   str = ""

ASSISTANT_NAME:   str = "Remi"
USER_NAME:        str = "Alex"

HISTORY_DATA:     str = open("context/history.txt", "r").read().format(ASSISTANT_NAME=ASSISTANT_NAME, USER_NAME=USER_NAME)

role_template: str = open("context/role.txt",       "r").read()
guidelines:    str = open("context/guidelines.txt", "r").read()

INPUT_DATA: str = role_template.replace("\n", " ") + guidelines.format(ASSISTANT_NAME=ASSISTANT_NAME, USER_NAME=USER_NAME).replace("\n", " ")

del role_template, guidelines

NF4_CONF = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.float16
)

def check_gpu() -> None:
    print("GPU is available" if torch.cuda.is_available() else "GPU is not available")

def collect_user_input() -> str:
    cursor.show()
    usr_input: str = str(input(USER_INPUT_TEXT))
    cursor.hide()
    return usr_input

def argument_init() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--debug", "-d", action="store_true")
    return argparser

def clear_terminal() -> None:
    if os.name != "nt":
        print("\033c", end=""); return
    os.system("cls")

def decode_response(generated_ids) -> tuple[str, str]:
    response: str = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    index:    str = re.findall(r'{}: (.*)'.format(ASSISTANT_NAME), response)[-1]
    return response, index
    

if __name__ == "__main__":
    cursor.hide()
    argparser = argument_init()
    args = argparser.parse_args()

    check_gpu()

    model = AutoModelForCausalLM.from_pretrained("JosephusCheung/LL7M", device_map="auto", quantization_config=NF4_CONF)
    tokenizer = AutoTokenizer.from_pretrained("JosephusCheung/LL7M", padding_side="left")
    clear_terminal()

    model.eval()

    history: str = HISTORY_DATA

    while True:
        usr_input = collect_user_input()
        if usr_input == "xlxquit": break

        chat = f"## History:\n{history}\n## Input:\n{INPUT_DATA}\nUser: {usr_input}\n## Response:\n"
        
        model_inputs = tokenizer([chat], return_tensors="pt").to("cuda")
        generated_ids = model.generate(**model_inputs, max_length=4096, repetition_penalty=1.14)
        response, index = decode_response(generated_ids)

        history += f"User: {usr_input}\n{ASSISTANT_NAME}: {index}\n"

        if not args.debug:
            print(RESPONSE_META + index.strip() + '\n')
            continue
        
        print(response)

        