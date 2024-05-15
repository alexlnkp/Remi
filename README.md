# Remi
Remi is a chatbot that utilizes a pre-trained language model, LL7M, to provide engaging conversations.

## Getting started
### Prerequisites
- Python 3.12 or higher
- `torch` and `transformers` libraries

### Installation
1. Clone the repository. `git clone https://github.com/alexlnkp/Remi.git`
2. Create a Python virtual environment. `python -m venv .venv`
3. Install required libraries in the virtual environment. `pip install -r requirements.txt`

### Usage
- Activate the virtual environment. `source .venv/bin/activate` (Linux) or `. .venv/Scripts/activate` (Windows)
- To run Remi in chatting mode, use `python prompt.py` or `./prompt.py` (Linux only)

## Contributing
Your contributions will be met with gratitude and will help greatly! If **You** would like to help Remi, please fork the repository and submit a pull request.

### TODO:
- General:
- [ ] Fine-tune the LLM to directly improve quality of the responses
- [ ] Do some fancy prompt-engineering.

- TTS related:
- [ ] Bootstrap a TTS in an external script to have an option to convert generated output directly to audio.
- [ ] ^ Using `Bark`, TTS can produce output with emotions. Therefore, also bootstrapping a tone recognition of the generated text would make the chatbot sound more natural.

- User input related:
- [ ] Bootstrap Whisper or any other voice recognition AI model to convert user speech into text which is then used as an input for inference of an LLM.
- [ ] Bootstrap the aforementioned tone recognition of the text to the user's voice recognition to add on layers of communications.

## License
This project is licensed under the **MIT** License - see the `LICENSE` file for details.

## Credit
[LL7M](https://huggingface.co/JosephusCheung/LL7M) - the pre-trained LLM used by Remi.

## Contact
For any questions or concerns, please contact the project maintainers at 413x1nkp@gmail.com.