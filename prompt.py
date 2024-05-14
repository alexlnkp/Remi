#!.venv/bin/python
import importlib.util

import cursor
from transformers import AutoModelForCausalLM, AutoTokenizer

from infer.utils import (
    NF4_CONF,
    argument_init,
    check_gpu,
    clear_terminal,
    collect_user_input,
    decode_response,
)

colorama_is_available: bool = importlib.util.find_spec("colorama") is not None
if colorama_is_available:
    from colorama import Fore, Style
    from colorama import init as colorama_init  # type: ignore

    colorama_init(autoreset=True)
    USER_INPUT_TEXT: str = f"{Style.DIM}{Fore.BLUE}>{Style.RESET_ALL} "
    RESPONSE_META: str = f"{Style.BRIGHT}{Fore.LIGHTRED_EX}"
else:
    print("Colorama is not available, will not use fancy output text... :(")
    USER_INPUT_TEXT: str = "> "
    RESPONSE_META: str = ""

ASSISTANT_NAME: str = "Remi"
USER_NAME: str = "Alex"

HISTORY_DATA: str = (
    open("context/history.txt", "r")
    .read()
    .format(ASSISTANT_NAME=ASSISTANT_NAME, USER_NAME=USER_NAME)
)

role_template: str = open("context/role.txt", "r").read()
guidelines: str = open("context/guidelines.txt", "r").read()

INPUT_DATA: str = role_template.replace("\n", " ") + guidelines.format(
    ASSISTANT_NAME=ASSISTANT_NAME, USER_NAME=USER_NAME
).replace("\n", " ")

del role_template, guidelines

if __name__ == "__main__":
    cursor.hide()
    argparser = argument_init()
    args = argparser.parse_args()

    check_gpu()

    model = AutoModelForCausalLM.from_pretrained(
        "JosephusCheung/LL7M", device_map="auto", quantization_config=NF4_CONF
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "JosephusCheung/LL7M", padding_side="left"
    )
    clear_terminal()

    model.eval()

    history: str = HISTORY_DATA

    while True:
        usr_input = collect_user_input(USER_INPUT_TEXT)
        if usr_input == "xlxquit":
            break

        chat = f"## History:\n{history}\n## Input:\n{INPUT_DATA}\nUser: {usr_input}\n## Response:\n"

        model_inputs = tokenizer([chat], return_tensors="pt").to("cuda")
        generated_ids = model.generate(
            **model_inputs, max_length=4096, repetition_penalty=1.16
        )
        response, index = decode_response(tokenizer, generated_ids, ASSISTANT_NAME)

        history += f"User: {usr_input}\n{ASSISTANT_NAME}: {index}\n"

        if not args.debug:
            print(RESPONSE_META + index.strip() + "\n")
            continue

        print(response)
