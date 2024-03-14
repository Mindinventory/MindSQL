import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizerFast

from .illm import ILlm
from .._utils.constants import LLAMA_VALUE_ERROR, LLAMA_PROMPT_EXCEPTION, CONFIG_REQUIRED_ERROR


class HuggingFace(ILlm):
    def __init__(self, config=None):
        """
        Initialize the class with an optional config parameter.

        Parameters:
            config (any): The configuration parameter.

        Returns:
            None
        """
        if config is None:
            raise ValueError(CONFIG_REQUIRED_ERROR)

        if 'model_name' not in config:
            raise ValueError(LLAMA_VALUE_ERROR)
        model_name = config.pop('model_name') or 'gpt2'

        self.tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **config)

    def system_message(self, message: str) -> any:
        """
        Create a system message.

        Parameters:
            message (str): The content of the system message.

        Returns:
            any: A formatted system message.

        Example:
            system_msg = system_message("System update: Server maintenance scheduled.")
        """
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> any:
        """
        Create a user message.

        Parameters:
            message (str): The content of the user message.

        Returns:
            any: A formatted user message.
        """
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> any:
        """
        Create an assistant message.

        Parameters:
            message (str): The content of the assistant message.

        Returns:
            any: A formatted assistant message.
        """
        return {"role": "assistant", "content": message}

    def invoke(self, prompt, **kwargs) -> str:
        """
        Submit a prompt to the model for generating a response.

        Parameters:
            prompt (str): The prompt parameter.
            **kwargs: Additional keyword arguments (optional).
                - temperature (float): The temperature parameter for controlling randomness in generation.

        Returns:
            str: The generated response from the model.
        """
        if prompt is None or len(prompt) == 0:
            raise Exception(LLAMA_PROMPT_EXCEPTION)

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000)
        temperature = kwargs.get("temperature", 0.1)

        with torch.no_grad():
            output = self.model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                                         max_length=2000, temperature=temperature,
                                         pad_token_id=self.tokenizer.pad_token_id,
                                         eos_token_id=self.tokenizer.eos_token_id,
                                         bos_token_id=self.tokenizer.bos_token_id, **kwargs)

        data = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return data
