from typing import List, Optional
from urllib.parse import urlparse

import pandas as pd
import pyodbc

from . import IDatabase
from .._utils import logger
from .._utils.constants import ERROR_WHILE_RUNNING_QUERY, ERROR_CONNECTING_TO_DB_CONSTANT, INVALID_DB_CONNECTION_OBJECT, \
    CONNECTION_ESTABLISH_ERROR_CONSTANT, SQLSERVER_SHOW_DATABASE_QUERY, SQLSERVER_DB_TABLES_INFO_SCHEMA_QUERY, \
    SQLSERVER_SHOW_CREATE_TABLE_QUERY

log = logger.init_loggers("SQL Server")


class SQLServer(IDatabase):
    @staticmethod
    def create_connection(url: str, **kwargs) -> any:
        """
        Connects to a SQL Server database using the provided URL.

        Parameters:
            - url (str): The connection string to the SQL Server database in the format:
                'DRIVER={ODBC Driver 17 for SQL Server};SERVER=server_name;DATABASE=database_name;UID=user;PWD=password'
            - **kwargs: Additional keyword arguments for the connection

        Returns:
            - connection: A connection to the SQL Server database
        """

        try:
            connection = pyodbc.connect(url, **kwargs)
            return connection
        except pyodbc.Error as e:
            log.error(ERROR_CONNECTING_TO_DB_CONSTANT.format("SQL Server", e))

    def execute_sql(self, connection, sql:str) -> Optional[pd.DataFrame]:
        """
        A function that runs an SQL query using the provided connection and returns the results as a pandas DataFrame.

        Parameters:
            connection: The connection object for the database.
            sql (str): The SQL query to be executed

        Returns:
            pd.DataFrame: A DataFrame containing the results of the SQL query.
        """
        try:
            self.validate_connection(connection)
            cursor = connection.cursor()
            cursor.execute(sql)
            columns = [column[0] for column in cursor.description]
            data = cursor.fetchall()
            data = [list(row) for row in data]
            cursor.close()
            return pd.DataFrame(data, columns=columns)
        except pyodbc.Error as e:
            log.error(ERROR_WHILE_RUNNING_QUERY.format(e))
            return None

    def get_databases(self, connection) -> List[str]:
        """
        Get a list of databases from the given connection and SQL query.

        Parameters:
            connection: The connection object for the database.

        Returns:
            List[str]: A list of unique database names.
        """
        try:
            self.validate_connection(connection)
            cursor = connection.cursor()
            cursor.execute(SQLSERVER_SHOW_DATABASE_QUERY)
            databases = [row[0] for row in cursor.fetchall()]
            cursor.close()
            return databases
        except pyodbc.Error as e:
            log.error(ERROR_WHILE_RUNNING_QUERY.format(e))
            return []

    def get_table_names(self, connection, database: str) -> pd.DataFrame:
        """
        Retrieves the tables along with their schema (schema.table_name) from the information schema for the specified
        database.

        Parameters:
            connection: The database connection object.
            database (str): The name of the database.

        Returns:
            DataFrame: A pandas DataFrame containing the table names from the information schema.
        """
        self.validate_connection(connection)
        query = SQLSERVER_DB_TABLES_INFO_SCHEMA_QUERY.format(db=database)
        return self.execute_sql(connection, query)




    def get_all_ddls(self, connection: any, database: str) -> pd.DataFrame:
        """
        A method to get the DDLs for all the tables in the database.

        Parameters:
            connection (any): The connection object.
            database (str): The name of the database.

        Returns:
            DataFrame: A pandas DataFrame containing the DDLs for all the tables in the database.
        """
        df_tables = self.get_table_names(connection, database)
        ddl_df = pd.DataFrame(columns=['Table', 'DDL'])
        for index, row in df_tables.iterrows():
            ddl = self.get_ddl(connection, row.iloc[0])
            ddl_df = ddl_df._append({'Table': row.iloc[0], 'DDL': ddl}, ignore_index=True)

        return ddl_df



    def validate_connection(self, connection: any) -> None:
        """
        A function that validates if the provided connection is a SQL Server connection.

        Parameters:
            connection: The connection object for accessing the database.

        Raises:
            ValueError: If the provided connection is not a SQL Server connection.

        Returns:
            None
        """
        if connection is None:
            raise ValueError(CONNECTION_ESTABLISH_ERROR_CONSTANT)
        if not isinstance(connection, pyodbc.Connection):
            raise ValueError(INVALID_DB_CONNECTION_OBJECT.format("SQL Server"))

    def get_ddl(self, connection: any, table_name: str, **kwargs) -> str:
        schema_name, table_name = table_name.split('.')
        query = SQLSERVER_SHOW_CREATE_TABLE_QUERY.format(table=table_name, schema=schema_name)
        df_ddl = self.execute_sql(connection, query)
        return df_ddl['SQLQuery'][0]

    def get_dialect(self) -> str:
        return 'tsql'
