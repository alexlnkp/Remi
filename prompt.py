#!.venv/bin/python

from infer.utils import (
    argument_init,
    check_gpu,
    clear_terminal,
    collect_user_input,
    decode_response,
    get_model_and_tokenizer,
    get_uinput_and_response_format,
)

USER_INPUT_TEXT, RESPONSE_META = get_uinput_and_response_format()

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
    argparser = argument_init()
    args = argparser.parse_args()

    cuda_available: bool = check_gpu()

    model, tokenizer = get_model_and_tokenizer("JosephusCheung/LL7M")
    clear_terminal()

    model.eval()

    history: str = HISTORY_DATA

    while True:
        usr_input: str = collect_user_input(USER_INPUT_TEXT)
        if usr_input == "xlxquit":
            break

        chat: str = (
            f"## History:\n{history}\n## Input:\n{INPUT_DATA}\nUser: {usr_input}\n## Response:\n"
        )

        model_inputs = tokenizer([chat], return_tensors="pt").to(
            "cuda" if cuda_available else "cpu"
        )
        generated_ids = model.generate(
            **model_inputs, max_length=4096, repetition_penalty=1.16
        )
        response, index = decode_response(tokenizer, generated_ids, ASSISTANT_NAME)

        history += f"User: {usr_input}\n{ASSISTANT_NAME}: {index}\n"

        if not args.debug:
            print(RESPONSE_META + index.strip() + "\n")
            continue

        print(response)
