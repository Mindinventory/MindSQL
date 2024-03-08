import os
import sqlite3
import warnings
from sqlite3 import Connection
from typing import List
from urllib.parse import urlparse

import pandas as pd
import requests

from .._utils import logger
from .._utils.constants import ERROR_DOWNLOADING_SQLITE_DB_CONSTANT, ERROR_CONNECTING_TO_DB_CONSTANT, \
    INVALID_DB_CONNECTION_OBJECT, ERROR_WHILE_RUNNING_QUERY, SQLLITE_GET_DB_QUERY, SQLLIE_TABLE_INFO_SCHEMA_CONSTANT, \
    SQLLITE_TRAINING_DATASET_QUERY_CONSTANT, CONNECTION_ESTABLISH_ERROR_CONSTANT
from . import IDatabase

warnings.simplefilter(action='ignore', category=UserWarning)
log = logger.init_loggers("Sqlite")


class Sqlite(IDatabase):
    @staticmethod
    def __download_database(url: str, destination_path: str) -> None:
        """
        Download the database if it doesn't exist

        Parameters:
            url (str): The URL of the database
            destination_path (str): The path to save the database

        Returns:
            None
        """
        try:
            response = requests.get(url)
            response.raise_for_status()  # Check that the request was successful
            with open(destination_path, "wb") as f:
                f.write(response.content)
        except requests.RequestException as e:
            log.info(ERROR_DOWNLOADING_SQLITE_DB_CONSTANT.format(e))

    def create_connection(self, url: str, **kwargs) -> Connection | None:
        """
        A method to create a connection with the database.

        Parameters:
            url (str): The URL of the database

        Returns:
            Connection | None: A connection object
        """
        if urlparse(url).scheme == '' and os.path.isabs(url):
            path = url
        else:
            path = os.path.basename(urlparse(url).path)

        # Download the database if it doesn't exist
        if not os.path.exists(path):
            self.__download_database(url, path)

        try:
            conn = sqlite3.connect(path)
            return conn
        except sqlite3.Error as e:
            log.info(ERROR_CONNECTING_TO_DB_CONSTANT.format("SQLite", e))
            return None

    def validate_connection(self, connection):
        """
        A function that validates if the provided connection is a SQLite connection.

        Parameters:
            connection (Connection): A connection object

        Returns:
            None
        """
        if connection is None:
            raise ValueError(CONNECTION_ESTABLISH_ERROR_CONSTANT)
        
        if not isinstance(connection, sqlite3.Connection):
            raise ValueError(INVALID_DB_CONNECTION_OBJECT.format("SQLite"))

    def execute_sql(self, connection, sql: str) -> pd.DataFrame:
        """
        A method to run a SQL query on the database.

        Parameters:
            connection (Connection): A connection object
            sql (str): The SQL query to run

        Returns:
            pd.DataFrame: A DataFrame containing the query results
        """
        self.validate_connection(connection)
        try:
            result = pd.read_sql_query(sql, connection)
            return result
        except sqlite3.Error as e:
            log.info(ERROR_WHILE_RUNNING_QUERY.format(e))
            return pd.DataFrame()

    def get_databases(self, connection) -> List[str]:
        """
        Get a list of databases from the given connection and SQL query.

        Parameters:
            connection (Connection): A connection object

        Returns:
            List[str]: A list of unique database names
        """
        self.validate_connection(connection)
        try:
            df_databases = pd.read_sql_query(SQLLITE_GET_DB_QUERY, connection)
            return df_databases["name"].tolist()
        except Exception as e:
            log.info(e)
            return []

    def get_table_names(self, connection, database: str) -> pd.DataFrame:
        """
        A method to get the list of tables in the database.

        Parameters:
            connection (Connection): A connection object
            database (str): The name of the database

        Returns:
            pd.DataFrame: The list of tables
        """
        self.validate_connection(connection)
        try:
            result = pd.read_sql_query(SQLLIE_TABLE_INFO_SCHEMA_CONSTANT, connection)
            return result
        except sqlite3.Error as e:
            log.info(ERROR_WHILE_RUNNING_QUERY.format(e))
            return pd.DataFrame()

    def get_all_ddls(self, connection, database: str) -> pd.DataFrame:
        """
        A method to get all DDLs in the database.

        Parameters:
            connection (Connection): A connection object
            database (str): The name of the database

        Returns:
            pd.DataFrame: The list of DDLs
        """
        self.validate_connection(connection)

        df_tables = self.get_table_names(connection, database)
        df_ddl = pd.DataFrame(columns=['Table', 'DDL'])

        for _, row in df_tables.iterrows():
            table_name = row['name']
            ddl_df = self.get_ddl(connection, table_name)
            df_ddl = df_ddl._append({'Table': table_name, 'DDL': ddl_df}, ignore_index=True)
        return df_ddl

    def get_ddl(self, connection, table_name: str, **kwargs) -> str:
        """
        A method to get the DDL for the table.

        Parameters:
            connection (Connection): A connection object
            table_name (str): The name of the table

        Returns:
            str: The DDL for the table
        """
        self.validate_connection(connection)
        ddl_df = pd.read_sql_query(SQLLITE_TRAINING_DATASET_QUERY_CONSTANT.format(table_name), connection)
        return ddl_df["sql"].iloc[0]

    def get_dialect(self) -> str:
        """
        A method to get the dialect of the database.

        Returns:
            str: The dialect
        """
        return 'sqlite3'
