from llama_cpp import Llama

from .._utils.constants import LLAMA_VALUE_ERROR, LLAMA_PROMPT_EXCEPTION, CONFIG_REQUIRED_ERROR
from .illm import ILlm


class LlamaCpp(ILlm):
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

        if 'model_path' not in config:
            raise ValueError(LLAMA_VALUE_ERROR)
        path = config.pop('model_path')

        self.model = Llama(model_path=path, **config)

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

        temperature = kwargs.get("temperature", 0.1)
        return self.model(prompt=prompt, temperature=temperature, echo=False)["choices"][0]["text"]
