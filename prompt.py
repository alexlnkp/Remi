#!.venv/bin/python

import torch

from infer.utils import (
    check_gpu,
    clear_cuda_cache,
    clear_terminal,
    collect_user_input,
    decode_response,
    get_model_and_tokenizer,
    get_uinput_and_response_format,
    hide_cursor,
    infer_argument_init,
)

USER_INPUT_TEXT, RESPONSE_META = get_uinput_and_response_format()

ASSISTANT_NAME: str = "Remi"
USER_NAME: str = "Alex"

role_template: str = open("context/role.txt", "r").read()
guidelines: str = open("context/guidelines.txt", "r").read()

INPUT_DATA: str = (
    role_template.replace("\n", " ")
    + " "
    + guidelines.format(ASSISTANT_NAME=ASSISTANT_NAME, USER_NAME=USER_NAME).replace(
        "\n", " "
    )
)


del role_template, guidelines

if __name__ == "__main__":
    argparser = infer_argument_init()
    args = argparser.parse_args()

    cuda_available: bool = check_gpu()
    print("Cuda available" if cuda_available else "Cuda not available")

    device: torch.device = "cuda" if cuda_available else "cpu"

    hide_cursor()
    model, tokenizer = get_model_and_tokenizer("JosephusCheung/LL7M")
    clear_terminal()

    if not args.no_adapter:
        model.load_adapter("413x1nkp/LL7M-Remi", adapter_name="remi")
        model.set_adapter("remi")
        model.enable_adapters()

    model.eval()

    history: str = (
        (
            open(args.history, "r")
            .read()
            .format(ASSISTANT_NAME=ASSISTANT_NAME, USER_NAME="User")
        )
        + "\n"
        if args.history
        else ""
    )

    while True:
        usr_input: str = collect_user_input(USER_INPUT_TEXT)
        if usr_input == "xlxquit":
            break

        chat: str = (
            f"## History:\n{history}\n## Input:\nSystem: {INPUT_DATA}\nUser: {usr_input}\n## Response:\n"
        )

        model_inputs = tokenizer(
            [chat],
            return_tensors="pt",
        ).to(device)

        generated_ids = model.generate(
            **model_inputs, max_new_tokens=4096, repetition_penalty=1.16
        )
        response, index = decode_response(tokenizer, generated_ids, ASSISTANT_NAME)

        history += f"User: {usr_input}\n{ASSISTANT_NAME}: {index}\n"

        clear_cuda_cache()
        del model_inputs, generated_ids

        if not args.debug and usr_input != "fuck":
            print(RESPONSE_META + index.strip() + "\n")
            continue

        print(response)
