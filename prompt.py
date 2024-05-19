#!.venv/bin/python

from infer.utils import main

"""
This script is the main entry point of the inference script. It is responsible
for initializing the inferencer based on the command-line arguments and running
the inference loop.
"""


ASSISTANT_NAME: str = "Remi"
"""
The name of the assistant generating the responses.
"""


USER_NAME: str = "Alex"
"""
The name of the user in the conversation.
"""


INPUT_DATA: str = (
    open("context/role.txt", "r").read().replace("\n", " ")
    + " "
    + open("context/guidelines.txt", "r")
    .read()
    .format(ASSISTANT_NAME=ASSISTANT_NAME, USER_NAME=USER_NAME)
    .replace("\n", " ")
)
"""
The initial input data for the conversation. This is the text that is passed
to the inferencer as the initial context.
"""


if __name__ == "__main__":
    # Run the inferencer
    main(ASSISTANT_NAME, INPUT_DATA)
