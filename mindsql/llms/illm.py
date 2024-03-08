import abc


class ILlm(metaclass=abc.ABCMeta):
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'system_message') and
                callable(subclass.system_message) and
                hasattr(subclass, 'user_message') and
                callable(subclass.user_message) and
                hasattr(subclass, 'assistant_message') and
                callable(subclass.assistant_message) and
                hasattr(subclass, 'invoke') and
                callable(subclass.invoke) or
                NotImplemented)

    @abc.abstractmethod
    def system_message(self, message: str) -> any:
        """
        A method to handle system messages.

        Parameters:
            message (str): The message received from the system.

        Returns:
            any: The return type of the function.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def user_message(self, message: str) -> any:
        """
        A method to handle user messages.

        Parameters:
            message (str): The message received from the user.

        Returns:
            any: The return type of the function.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def assistant_message(self, message: str) -> any:
        """
        A method to handle assistant messages.

        Parameters:
            message (str): The message received from the assistant.

        Returns:
            any: The return type of the function.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def invoke(self, prompt, **kwargs) -> str:
        """
        A method to invoke the LLM.

        Parameters:
            prompt (str): The prompt to be sent to the LLM.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The response from the LLM.
        """
        raise NotImplementedError
