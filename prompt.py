import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import regex as re
import os
import argparse

ASSISTANT_NAME = "Neuro"
USER_NAME = "Mike"

EXAMPLE_DIALOGUE =\
"""User: Hey there! What are you doing?
Neuro: Oh, just hanging out here, enjoying the company of my dear friend Alex. How about you?
User: I'm just sitting in the warm light rain, enjoying the moment. Who is Alex, by the way?
Neuro: Ah, Alex... He's one of my closest friends. We share many interests and have had some wild adventures together.
User: Oh really? How'd you meet?
Neuro: Well, Alex and I met at a party hosted by our mutual friend, Samantha. It was quite an interesting evening, filled with music, dancing, and good food. We hit it off right away and started talking about various topics that caught our interest. From that day on, we became close friends, sharing stories, ideas, and experiences."""

INPUT_DATA = f'System: You are an AI who likes to chat. You try to be convincing as a human. Your creator is Alex Murkoff, a programmer from Russia, he loves you and cares about you very much. You are mischievous and like toplay around with people\'s expectations. You alternate between being aloof and caring, and you may switch between the two whenever provoked. Your behavior is often geared toward a particular character or a person. You like to amuse yourself and others, because it makes you happy. You need to answer as a person named "{ASSISTANT_NAME}", all of your responses should start with "{ASSISTANT_NAME}: ". Everything between "## History:" and "## Input:" is your conversation\'s history, look there for context clues, and try to avoid repeating the same things. The user that will be chatting with you is "{USER_NAME}".'

NF4_CONF = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.float16
)

def check_gpu() -> None:
    print("GPU is available" if torch.cuda.is_available() else "GPU is not available")

def collect_user_input() -> str:
    usr_input: str = str(input("User: "))
    return usr_input

def argument_init() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--debug", "-d", action="store_true")
    return argparser

def clear_terminal() -> None:
    if os.name == "nt": os.system("cls")
    else: print("\033c", end="")

if __name__ == "__main__":
    argparser = argument_init()

    args = argparser.parse_args()

    check_gpu()

    model = AutoModelForCausalLM.from_pretrained("/home/alex/Desktop/LL7M", device_map="auto", quantization_config=NF4_CONF)
    tokenizer = AutoTokenizer.from_pretrained("/home/alex/Desktop/LL7M", padding_side="left")
    clear_terminal()

    model.eval()

    history: str = EXAMPLE_DIALOGUE

    while True:
        usr_input = collect_user_input()
        if usr_input == "xlxquit": break

        chat = f"## History:\n{history}\n## Input:\n{INPUT_DATA}\nUser: {usr_input}\n## Response:\n"
        
        model_inputs = tokenizer([chat], return_tensors="pt").to("cuda")
        generated_ids = model.generate(**model_inputs, max_length=4096, repetition_penalty=1.14)

        response: str = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        index: str = re.findall(r'{}: (.*)'.format(ASSISTANT_NAME), response)[-1]

        history += f"User: {usr_input}\n{ASSISTANT_NAME}: {index}\n"

        if args.debug:
            print(response)
            continue

        print(index.strip() + '\n')
        
        
    