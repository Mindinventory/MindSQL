from anthropic import Anthropic

from . import ILlm
from .._utils.constants import ANTHROPIC_VALUE_ERROR, PROMPT_EMPTY_EXCEPTION


class AnthropicAi(ILlm):
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
            raise ValueError(ANTHROPIC_VALUE_ERROR)
        api_key = config.pop('api_key')
        self.client = Anthropic(api_key=api_key, **config)

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
                - max_tokens (int): Maximum number of tokens to be generated.
        Returns:
            str: The generated response from the model.
        """
        if prompt is None or len(prompt) == 0:
            raise Exception(PROMPT_EMPTY_EXCEPTION)

        model = self.config.get("model", "claude-3-opus-20240229")
        temperature = kwargs.get("temperature", 0.1)
        max_tokens = kwargs.get("max_tokens", 1024)
        response = self.client.messages.create(model=model, messages=[{"role": "user", "content": prompt}],
                                               max_tokens=max_tokens, temperature=temperature)
        for content in response.content:
            if isinstance(content, dict) and content.get("type") == "text":
                return content["text"]
            elif hasattr(content, "text"):
                return content.text
