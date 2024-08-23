from ollama import Client, Options

from .illm import ILlm
from .._utils.constants import PROMPT_EMPTY_EXCEPTION, OLLAMA_CONFIG_REQUIRED
from .._utils import logger

log = logger.init_loggers("Ollama Client")


class Ollama(ILlm):
    def __init__(self, model_config: dict, client_config=None, client: Client = None):
        """
        Initialize the class with an optional config parameter.

        Parameters:
            model_config (dict): The model configuration parameter.
            config (dict): The configuration parameter.
            client (Client): The client parameter.

        Returns:
            None
        """
        self.client = client
        self.client_config = client_config
        self.model_config = model_config

        if self.client is not None:
            if self.client_config is not None:
                log.warning("Client object provided. Ignoring client_config parameter.")
            return

        if client_config is None:
            raise ValueError(OLLAMA_CONFIG_REQUIRED.format(type="Client"))

        if model_config is None:
            raise ValueError(OLLAMA_CONFIG_REQUIRED.format(type="Model"))

        if 'model' not in model_config:
            raise ValueError(OLLAMA_CONFIG_REQUIRED.format(type="Model name"))

        self.client = Client(**client_config)

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
            str
        """
        if not prompt:
            raise ValueError(PROMPT_EMPTY_EXCEPTION)

        model = self.model_config.get('model')
        temperature = kwargs.get('temperature', 0.1)

        response = self.client.chat(
            model=model,
            messages=[self.user_message(prompt)],
            options=Options(
                temperature=temperature
            )
        )

        return response['message']['content']
