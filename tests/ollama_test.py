import unittest
from unittest.mock import MagicMock, patch
from ollama import Client, Options

from mindsql.llms import ILlm
from mindsql.llms import Ollama
from mindsql._utils.constants import PROMPT_EMPTY_EXCEPTION, OLLAMA_CONFIG_REQUIRED


class TestOllama(unittest.TestCase):

    def setUp(self):
        # Common setup for each test case
        self.model_config = {'model': 'sqlcoder'}
        self.client_config = {'host': 'http://localhost:11434/'}
        self.client_mock = MagicMock(spec=Client)

    def test_initialization_with_client(self):
        ollama = Ollama(model_config=self.model_config, client=self.client_mock)
        self.assertEqual(ollama.client, self.client_mock)
        self.assertIsNone(ollama.client_config)
        self.assertEqual(ollama.model_config, self.model_config)

    def test_initialization_with_client_config(self):
        ollama = Ollama(model_config=self.model_config, client_config=self.client_config)
        self.assertIsNotNone(ollama.client)
        self.assertEqual(ollama.client_config, self.client_config)
        self.assertEqual(ollama.model_config, self.model_config)

    def test_initialization_missing_client_and_client_config(self):
        with self.assertRaises(ValueError) as context:
            Ollama(model_config=self.model_config)
        self.assertEqual(str(context.exception), OLLAMA_CONFIG_REQUIRED.format(type="Client"))

    def test_initialization_missing_model_config(self):
        with self.assertRaises(ValueError) as context:
            Ollama(model_config=None, client_config=self.client_config)
        self.assertEqual(str(context.exception), OLLAMA_CONFIG_REQUIRED.format(type="Model"))

    def test_initialization_missing_model_name(self):
        with self.assertRaises(ValueError) as context:
            Ollama(model_config={}, client_config=self.client_config)
        self.assertEqual(str(context.exception), OLLAMA_CONFIG_REQUIRED.format(type="Model name"))

    def test_system_message(self):
        ollama = Ollama(model_config=self.model_config, client=self.client_mock)
        message = ollama.system_message("Test system message")
        self.assertEqual(message, {"role": "system", "content": "Test system message"})

    def test_user_message(self):
        ollama = Ollama(model_config=self.model_config, client=self.client_mock)
        message = ollama.user_message("Test user message")
        self.assertEqual(message, {"role": "user", "content": "Test user message"})

    def test_assistant_message(self):
        ollama = Ollama(model_config=self.model_config, client=self.client_mock)
        message = ollama.assistant_message("Test assistant message")
        self.assertEqual(message, {"role": "assistant", "content": "Test assistant message"})

    @patch.object(Client, 'chat', return_value={'message': {'content': 'Test response'}})
    def test_invoke_success(self, mock_chat):
        ollama = Ollama(model_config=self.model_config, client=Client())
        response = ollama.invoke("Test prompt")

        # Check if the response is as expected
        self.assertEqual(response, 'Test response')

        # Verify that the chat method was called with the correct arguments
        mock_chat.assert_called_once_with(
            model=self.model_config['model'],
            messages=[{"role": "user", "content": "Test prompt"}],
            options=Options(temperature=0.1)
        )

    def test_invoke_empty_prompt(self):
        ollama = Ollama(model_config=self.model_config, client=self.client_mock)
        with self.assertRaises(ValueError) as context:
            ollama.invoke("")
        self.assertEqual(str(context.exception), PROMPT_EMPTY_EXCEPTION)


if __name__ == '__main__':
    unittest.main()
