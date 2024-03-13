import google.generativeai as genai

from .._utils.constants import GOOGLE_GEN_AI_VALUE_ERROR, GOOGLE_GEN_AI_APIKEY_ERROR
from . import ILlm


class GoogleGenAi(ILlm):
    def __init__(self, config=None):
        """
        Initialize the class with an optional config parameter.

        Parameters:
            config (any): The configuration parameter.

        Returns:
            None
        """
        if config is None:
            raise ValueError(GOOGLE_GEN_AI_VALUE_ERROR)

        if 'api_key' not in config:
            raise ValueError(GOOGLE_GEN_AI_APIKEY_ERROR)
        api_key = config.pop('api_key')
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro', **config)

    def system_message(self, message: str) -> any:
        """
        Create a system message.

        Parameters:
            message (str): The content of the system message.

        Returns:
            any: A formatted system message.
        """
        return {"role": "system", "parts": message}

    def user_message(self, message: str) -> any:
        """
        Create a user message.

        Parameters:
            message (str): The content of the user message.

        Returns:
            any: A formatted user message.
        """
        return {"role": "user", "parts": message}

    def assistant_message(self, message: str) -> any:
        """
        Create an assistant message.

        Parameters:
            message (str): The content of the assistant message.

        Returns:
            any: A formatted assistant message.
        """
        return {'role': 'model', 'parts': message}

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
            raise Exception("Prompt cannot be empty.")

        temperature = kwargs.get("temperature", 0.1)
        response = self.model.generate_content(prompt,
                                               generation_config=genai.GenerationConfig(temperature=temperature))
        return response.text
