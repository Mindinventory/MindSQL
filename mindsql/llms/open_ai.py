from openai import OpenAI

from . import ILlm
from .._utils.constants import OPENAI_VALUE_ERROR, OPENAI_PROMPT_EMPTY_EXCEPTION


class OpenAi(ILlm):
    def __init__(self, config=None, client=None):
        """
        Initialize the class with an optional config parameter.

        Parameters:
            config (any): The configuration parameter.
            client (any): The client parameter.

        Returns:
            None
        """
        self.config = config
        self.client = client

        if client is not None:
            self.client = client
            return

        if 'api_key' not in config:
            raise ValueError(OPENAI_VALUE_ERROR)
        api_key = config.pop('api_key')
        self.client = OpenAI(api_key=api_key, **config)

    def system_message(self, message: str) -> any:
        """
        Create a system message.

        Parameters:
            message (str): The message parameter.

        Returns:
            any
        """
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> any:
        """
        Create a user message.

        Parameters:
            message (str): The message parameter.

        Returns:
            any
        """
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> any:
        """
        Create an assistant message.

        Parameters:
            message (str): The message parameter.

        Returns:
            any
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
            raise Exception(OPENAI_PROMPT_EMPTY_EXCEPTION)

        model = self.config.get("model", "gpt-3.5-turbo")
        temperature = kwargs.get("temperature", 0.1)
        max_tokens = kwargs.get("max_tokens", 500)
        response = self.client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                                       max_tokens=max_tokens, stop=None, temperature=temperature)
        return response.choices[0].message.content
