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
    BatchEncoding,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    PreTrainedTokenizer,
)

NF4_CONF = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)


def get_uinput_and_response_format() -> tuple[str, str]:
    """
    Returns a tuple containing the user input text and response meta string.

    The user input text is the string used to prompt the user for input, and the
    response meta string is used to format the response from the model.

    If the colorama library is available,
    the user input text and the response meta string will be colored

    If the colorama library is not available, the user input text and response
    meta string will be plain text.
    """
    user_input_text: str = "> "
    response_meta: str = ""

    if importlib.util.find_spec("colorama") is not None:
        # Import colorama if available
        from colorama import Fore, Style  # type: ignore
        from colorama import init as colorama_init  # type: ignore

        # Initialize colorama if available
        colorama_init(autoreset=True)

        # Format the user input text with a dimmed blue color
        user_input_text = "{}{}>{} ".format(Style.DIM, Fore.BLUE, Style.RESET_ALL)

        # Format the response meta string with a bright light red color
        response_meta = "{}{}".format(Style.BRIGHT, Fore.LIGHTRED_EX)
    else:
        # If colorama is not available, print a message
        print("Colorama is not available, will not use fancy output text... :(")

    return user_input_text, response_meta


USER_INPUT_TEXT, RESPONSE_META = get_uinput_and_response_format()


def hide_cursor() -> None:
    cursor.hide()


def show_cursor() -> None:
    cursor.show()


def get_model_and_tokenizer(
    model_name: str,
) -> tuple[LlamaForCausalLM, LlamaTokenizer]:
    """
    Load a pre-trained LLaMA model and tokenizer from Hugging Face Hub.

    Args:
        `model_name`: `str` - The name of the pre-trained model to load.

    Returns:
        `tuple[LlamaForCausalLM, LlamaTokenizer]` - A tuple containing the loaded model and tokenizer.
    """
    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        # Use NF4 quantization configuration for 4-bit training
        quantization_config=NF4_CONF,
    )
    tokenizer: LlamaTokenizer = AutoTokenizer.from_pretrained(
        model_name,
        # Set padding side to left to align with the default behavior of the LLaMA tokenizer
        padding_side="left",
    )

    return model, tokenizer


def check_gpu() -> bool:
    """
    Checks if the GPU has CUDA available.

    Returns:
        `bool` - `True` if the GPU is available and can be used, `False` otherwise.
    """
    return torch.cuda.is_available()


def collect_user_input(user_input_text: str) -> str:
    """
    Asks the user for input and returns the input text.

    Args:
        `user_input_text`: `str` - The text used to prompt the user for input.

    Returns:
        `str` - The input text entered by the user.
    """
    cursor.show()
    # Ask the user for input
    usr_input: str = input(user_input_text)
    cursor.hide()
    # Return the input text
    return usr_input


def infer_argument_init() -> argparse.ArgumentParser:
    """
    Returns an ArgumentParser instance with the arguments required to run the inference script.

    The arguments are:
    - `--debug` (-d):
        Debug mode. If set, the response from the model will not be formatted.
    - `--history` (-h):
        Path to the history file. If not provided, no history will be used. The history file is used to
        store the conversation history and pass it to the model as input.

    Returns:
        `argparse.ArgumentParser` - An ArgumentParser instance with the arguments required to run the
        inference script.
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--debug",
        "-d",
        required=False,
        action="store_true",
        help="debug mode, don't format response from model",
    )
    argparser.add_argument(
        "--history",
        required=False,
        type=str,
        metavar="FILE",
        default="",
        help="path to history file (e.g. 'context/history.txt'). if not provided, no history will be used",
    )
    argparser.add_argument(
        "--no-adapter",
        "-a",
        required=False,
        action="store_true",
        help="don't use LoRA adapter for Remi",
    )
    return argparser


def clear_terminal() -> None:
    """
    Clear the terminal screen.

    If the OS is not Windows, print the ANSI escape sequence to clear the screen.
    If the OS is Windows, call the Windows command to clear the screen.
    """
    if os.name != "nt":
        # ANSI escape sequence to clear the screen
        # This works on most Unix-like systems
        print("\033c", end="")
    else:
        # Windows command to clear the screen
        # This works on Windows platforms
        # TODO: Use a cross-platform way to clear the screen
        # (e.g. using a library like blessings)
        # trunk-ignore(bandit/B605)
        # trunk-ignore(bandit/B607)
        os.system("cls")


def clear_cuda_cache() -> None:
    torch.cuda.empty_cache()


def decode_response(
    tokenizer: PreTrainedTokenizer, generated_ids: torch.Tensor, assistant_name: str
) -> Tuple[str, str]:
    """
    Decode the response from the model and extract the response index.

    Args:
        `tokenizer`: `PreTrainedTokenizer` - The tokenizer used to decode the response.
        `generated_ids`: `torch.Tensor` - The generated IDs from the model.
        `assistant_name`: `str` - The name of the assistant.

    Returns:
        Tuple[str, str]: A tuple containing the decoded response and the response index.
    """
    response: str = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    index: str = re.findall(r"{}: (.*)".format(assistant_name), response)[-1]
    if re.match(r'(.*)". Continue the conversation accordingly(.*)', index):
        index = re.findall(r"## Response:\n(.*)", response)[-1]
    return response, index


def prepare_model_and_tokenizer_for_inference(
    args: argparse.Namespace,
) -> tuple[LlamaForCausalLM, LlamaTokenizer]:
    """
    Load the pre-trained LLaMA model and tokenizer, and prepare the model for inference.

    Args:
        `args`: `argparse.Namespace` - The parsed command-line arguments.

    Returns:
        `tuple[LlamaForCausalLM, LlamaTokenizer]`: A tuple containing the loaded model and tokenizer.
    """
    model, tokenizer = get_model_and_tokenizer("JosephusCheung/LL7M")

    if not args.no_adapter:
        # Load the adapter for the Remi instruction set
        # The adapter is a pre-trained model that is responsible for generating text
        # based on human instructions.
        model.load_adapter("413x1nkp/LL7M-Remi", adapter_name="remi")
        # Set the adapter to be the active adapter for the model
        model.set_adapter("remi")
        # Enable the adapter for the model
        model.enable_adapters()

    # Set the model to evaluation mode
    model.eval()

    # Clear the terminal screen to provide a clean output
    clear_terminal()

    # Return the loaded model and tokenizer
    return model, tokenizer


def tokenize_chat(
    device: torch.device,
    tokenizer: LlamaTokenizer,
    chat: str,
) -> BatchEncoding:
    """
    Tokenize the given chat input and move the result to the specified device.

    Args:
        `device`: `torch.device` - The device to move the tokenized input to.
        `tokenizer`: `LlamaTokenizer` - The tokenizer used to tokenize the input.
        `chat`: `str` - The input chat to be tokenized.

    Returns:
        `BatchEncoding` - A batch encoding containing the tokenized input.
    """
    model_inputs = tokenizer(
        [chat],
        return_tensors="pt",
    ).to(device)

    return model_inputs


def generate_response(
    model_inputs: BatchEncoding, model: LlamaForCausalLM
) -> torch.Tensor:
    """
    Generate a response from the model given a set of input parameters.

    Args:
        `model_inputs`: `BatchEncoding` - A batch encoding containing the input parameters.
        `model`: `LlamaForCausalLM` - The LLaMA model used to generate the response.

    Returns:
        `torch.Tensor` - A tensor containing the generated IDs of the response.

    Notes:
        The generated IDs are obtained by calling the `generate` method of the LLaMA model.
        The `generate` method takes in the input parameters, specified in the `model_inputs`
        batch encoding, and generates a response based on those parameters. The response is
        generated in the form of IDs, which are then converted to text using the tokenizer.
    """
    generated_ids = (
        model.generate(
            model_inputs["input_ids"].clone().detach(),
            max_new_tokens=4096,
            repetition_penalty=1.16,
        )
        .clone()
        .detach()
    )

    return generated_ids


def mainloop(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    device: torch.device,
    args: argparse.Namespace,
    history: str,
    input_data: str,
    assistant_name: str,
) -> None:
    """
    Main loop of the inferencer.

    This function is responsible for generating responses to user input using the
    LLaMA model and tokenizer. The generated responses are then appended to the
    history string, which is used to generate the chat context for the next set of
    user inputs.

    Args:
        `model`: `LLaMAForCausalLM` - The LLaMA model used to generate responses.
        `tokenizer`: `LlamaTokenizerFast` - The tokenizer used to tokenize the input.
        `device`: `torch.device` - The device to move the model and tokenizer to.
        `args`: `argparse.Namespace` - The parsed command line arguments.
        `history`: `str` - The history of the conversation.
        `input_data`: `str` - The initial input data for the conversation.
        `assistant_name`: `str` - The name of the assistant generating the responses.
    """
    while True:
        # Ask the user for input
        usr_input: str = collect_user_input(USER_INPUT_TEXT)

        # Check if the user wants to quit
        if usr_input == "xlxquit":
            return

        # Create the chat string containing the history, input data, and user input
        chat: str = (
            "## History:\n{}\n## Input:\nSystem: {}\nUser: {}\n## Response:\n".format(history, input_data, usr_input)
        )

        # Tokenize the chat string
        model_inputs = tokenize_chat(device, tokenizer, chat)

        # Generate a response using the LLaMA model
        generated_ids = generate_response(model_inputs, model)

        # Decode the response and extract the response index
        response, index = decode_response(tokenizer, generated_ids, assistant_name)

        # Append the user input and generated response to the history string
        history += "User: {}\n{}: {}\n".format(usr_input, assistant_name, index)

        # Clear the CUDA cache to free up memory
        clear_cuda_cache()
        del model_inputs, generated_ids

        # Print the response if the debug flag is not set or the input is not "fuck"
        if not args.debug and usr_input != "fuck":
            print(RESPONSE_META + index.strip() + "\n")
            continue

        # Print the generated response
        print(response)


def get_history(args: argparse.Namespace, assistant_name: str) -> str:
    """
    Returns the history string to be used in the inference script.

    The history string is obtained from the `--history` command line argument, if provided.
    If the `--history` argument is not provided, an empty string is returned.

    The history string is formatted to include the `assistant_name` and `USER_NAME`
    placeholders, which are replaced with the actual values when the history string is
    formatted. The placeholders are used to make it easier to update the history string
    without having to modify the code.

    Args:
        `args`: `argparse.Namespace` - The parsed command line arguments.
        `assistant_name`: `str` - The name of the assistant generating the responses.

    Returns:
        `str` - The history string to be used in the inference script.
    """
    history: str = (
        (
            open(args.history, "r")
            .read()
            .format(ASSISTANT_NAME=assistant_name, USER_NAME="User")
        )
        + "\n"
        if args.history
        else ""
    )

    return history


def main(
    assistant_name: str,
    input_data: str,
) -> None:
    """
    Main entry point of the inferencer.

    This function is responsible for initializing the inferencer based on the command-line
    arguments and running the inference loop.

    Args:
        `assistant_name`: `str` - The name of the assistant generating the responses.
        `input_data`: `str` - The initial input data for the conversation.
    """
    argparser = infer_argument_init()
    args = argparser.parse_args()

    cuda_available: bool = check_gpu()
    print("Cuda available" if cuda_available else "Cuda not available")

    device: torch.device = (
        torch.device("cuda") if cuda_available else torch.device("cpu")
    )

    hide_cursor()

    # Load the pre-trained LLaMA model and tokenizer, and prepare the model for inference
    model, tokenizer = prepare_model_and_tokenizer_for_inference(args)

    # Get the history string to be used in the inference script
    history = get_history(args, assistant_name)

    # Run the inference loop
    mainloop(model, tokenizer, device, args, history, input_data, assistant_name)

    # Show the cursor again
    show_cursor()
