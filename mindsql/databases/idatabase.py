import abc
import pandas as pd
from typing import List


class IDatabase(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'create_connection') and
                callable(subclass.create_connection) and
                hasattr(subclass, 'execute_sql') and
                callable(subclass.execute_sql) and
                hasattr(subclass, 'get_databases') and
                callable(subclass.get_databases) and
                hasattr(subclass, 'get_table_names') and
                callable(subclass.get_table_names) and
                hasattr(subclass, 'get_all_ddls') and
                callable(subclass.get_all_ddls) and
                hasattr(subclass, 'get_ddl') and
                callable(subclass.get_ddl) and
                hasattr(subclass, 'get_dialect') and
                callable(subclass.get_dialect) and
                hasattr(subclass, 'validate_connection') and
                callable(subclass.validate_connection) or
                NotImplemented)

    @abc.abstractmethod
    def create_connection(self, url: str, **kwargs) -> any:
        """
        A method to create a connection to the database.

        Parameters:
            url (str): The URL of the database.
            **kwargs: Additional keyword arguments.

        Returns:
            any: The connection object.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def execute_sql(self, connection, sql: str) -> pd.DataFrame:
        """
        A method to execute SQL on the database.

        Parameters:
            connection (any): The connection object.
            sql (str): The SQL to be executed.

        Returns:
            pd.DataFrame: The result of the SQL query.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_databases(self, connection) -> List[str]:
        """
        A method to get the list of databases in the database.

        Parameters:
            connection (any): The connection object.

        Returns:
            List[str]: The list of databases.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_table_names(self, connection, database: str) -> pd.DataFrame:
        """
        A method to get the list of tables in the database.

        Parameters:
            connection (any): The connection object.
            database (str): The name of the database.

        Returns:
            pd.DataFrame: The list of tables.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_all_ddls(self, connection: any, database: str) -> pd.DataFrame:
        """
        A method to get all DDLs in the database.

        Parameters:
            database (str): DB name
            connection (any): The connection object.

        Returns:
            pd.DataFrame: The list of DDLs.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def validate_connection(self, connection: any) -> None:
        """
        A method to validate the connection.

        Parameters:
            connection (any): The connection object.

        Raises:
            ValueError: If the connection is None.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_ddl(self, connection: any, table_name: str, **kwargs) -> str:
        """
        A method to get the DDL of a table in the database.

        Parameters:
            connection (any): The connection object.
            table_name (str): The name of the table.

        Returns:
            str: The DDL of the table.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_dialect(self) -> str:
        """
        A method to get the dialect of the database

        Returns:
            str: The dialect of the database
        """
        raise NotImplementedError
