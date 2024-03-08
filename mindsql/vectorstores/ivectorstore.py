import abc
import pandas as pd


class IVectorstore(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'retrieve_relevant_question_sql') and
                callable(subclass.retrieve_relevant_question_sql) and
                hasattr(subclass, 'retrieve_relevant_ddl') and
                callable(subclass.retrieve_relevant_ddl) and
                hasattr(subclass, 'index_question_sql') and
                callable(subclass.index_question_sql) and
                hasattr(subclass, 'index_ddl') and
                callable(subclass.index_ddl) and
                hasattr(subclass, 'index_documentation') and
                callable(subclass.index_documentation) and
                hasattr(subclass, 'fetch_all_vectorstore_data') and
                callable(subclass.fetch_all_vectorstore_data) and
                hasattr(subclass, 'retrieve_relevant_documentation') and
                callable(subclass.retrieve_relevant_documentation) and
                hasattr(subclass, 'delete_vectorstore_data') and
                callable(subclass.delete_vectorstore_data) or
                NotImplemented)

    @abc.abstractmethod
    def retrieve_relevant_question_sql(self, question: str, **kwargs) -> list:
        """
        This method retrieves the related question and SQL based on the provided question and optional keyword
        arguments.

        Parameters:
            question (str): The question for which related question and SQL is to be retrieved.
            **kwargs: Optional keyword arguments.

        Returns:
            list: A list of related question and SQL items.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def retrieve_relevant_ddl(self, question: str, **kwargs) -> list:
        """
        A method to get related DDL statements based on a question and optional keyword arguments.

        Parameters:
            question: str - the question for which related DDL statements are to be retrieved
            **kwargs: additional keyword arguments

        Returns:
            list - a list of related DDL statements
        """
        raise NotImplementedError

    @abc.abstractmethod
    def retrieve_relevant_documentation(self, question: str, **kwargs) -> list:
        """
        A method to get related documentation based on a question and optional keyword arguments.

        Parameters:
            question: str - the question for which related documentation is to be retrieved
            **kwargs: additional keyword arguments

        Returns:
            list - a list of related documentation
        """
        raise NotImplementedError

    @abc.abstractmethod
    def index_question_sql(self, question: str, sql: str, **kwargs) -> str:
        """
        A method to add a question and SQL pair to the vectorstore.

        Parameters:
            question (str): The question to be added to the vectorstore.
            sql (str): The SQL to be added to the vectorstore.
            **kwargs: Additional keyword arguments.

        Returns:
            str: A message confirming the successful addition of the question and SQL pair.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def index_ddl(self, ddl: str, **kwargs) -> str:
        """
        A method to add DDL to the vectorstore.

        Parameters:
            ddl (str): The DDL to be added to the vectorstore.
            **kwargs: Additional keyword arguments.

        Returns:
            str: A message confirming the successful addition of the DDL.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def index_documentation(self, documentation: str, **kwargs) -> str:
        """
        A method to add documentation to the vectorstore.

        Parameters:
            documentation (str): The documentation to be added to the vectorstore.
            **kwargs: Additional keyword arguments.

        Returns:
            str: A message confirming the successful addition of the documentation.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fetch_all_vectorstore_data(self, **kwargs) -> pd.DataFrame:
        """
        A method to fetch all data from the vectorstore.

        Parameters:
            **kwargs: Additional keyword arguments.

        Returns:
            pd.DataFrame: The data from the vectorstore.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def delete_vectorstore_data(self, item_id: str, **kwargs) -> bool:
        """
        A method to delete data from the vectorstore based on the provided item_id.

        Parameters:
            item_id (str): The unique identifier associated with the data to be deleted.
            **kwargs: Additional keyword arguments.

        Returns:
            bool: True if the deletion was successful, False otherwise.
        """
        raise NotImplementedError
