from typing import List
from urllib.parse import urlparse

import pandas as pd
import psycopg2
from psycopg2 import extensions

from . import IDatabase
from .._utils import logger
from .._utils.constants import ERROR_CONNECTING_TO_DB_CONSTANT, INVALID_DB_CONNECTION_OBJECT, ERROR_WHILE_RUNNING_QUERY, \
    POSTGRESQL_SHOW_DATABASE_QUERY, POSTGRESQL_DB_TABLES_INFO_SCHEMA_QUERY, \
    POSTGRESQL_SHOW_CREATE_TABLE_QUERY, CONNECTION_ESTABLISH_ERROR_CONSTANT

log = logger.init_loggers("Postgres")


class Postgres(IDatabase):
    @staticmethod
    def create_connection(url: str, **kwargs) -> any:
        """
        Connects to a PostgreSQL database using the provided URL.

        Parameters:
            - url (str): The URL in the format postgresql://username:password@host:port/database_name
            - **kwargs: Additional keyword arguments for the connection

        Returns:
            - connection: A connection to the PostgreSQL database

        Exceptions:
            - psycopg2.OperationalError: If an error occurs while connecting to the PostgreSQL database
        """
        try:
            parsed_url = urlparse(url)
            connection = psycopg2.connect(user=parsed_url.username, password=parsed_url.password,
                                          host=parsed_url.hostname, port=parsed_url.port,
                                          database=parsed_url.path.lstrip('/'))
            return connection
        except psycopg2.OperationalError as e:
            log.info(ERROR_CONNECTING_TO_DB_CONSTANT.format("PostgreSQL", e))

    def validate_connection(self, connection: any) -> None:
        """
        A function that validates if the provided connection is a PostgreSQL connection.

        Parameters:
            connection: The connection object for accessing the database.

        Raises:
            ValueError: If the provided connection is not a PostgreSQL connection.
        """
        if connection is None:
            raise ValueError(CONNECTION_ESTABLISH_ERROR_CONSTANT)
        if not isinstance(connection, psycopg2.extensions.connection):
            raise ValueError(INVALID_DB_CONNECTION_OBJECT.format("PostgreSQL"))

    def execute_sql(self, connection, sql: str) -> pd.DataFrame:
        """
        A function that runs an SQL query using the provided connection and returns the results as a pandas DataFrame.

        Parameters:
            connection: The connection object for accessing the database.
            sql (str): The SQL query to be executed.

        Returns:
            pd.DataFrame: A DataFrame containing the results of the SQL query.
        """
        try:
            self.validate_connection(connection)
            cursor = connection.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(results, columns=column_names)
            cursor.close()
            return df
        except psycopg2.Error as e:
            log.info(ERROR_WHILE_RUNNING_QUERY.format(e))

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
            df_databases = self.execute_sql(connection=connection, sql=POSTGRESQL_SHOW_DATABASE_QUERY)
        except Exception as e:
            log.info(e)
            return []

        return df_databases["DATABASE_NAME"].unique().tolist()

    def get_table_names(self, connection, database: str) -> pd.DataFrame:
        """
        Retrieves the tables from the information schema for the specified database.

        Parameters:
            connection: The database connection object.
            database (str): The name of the database.

        Returns:
            DataFrame: A pandas DataFrame containing the table names from the information schema.
        """
        self.validate_connection(connection)
        query = POSTGRESQL_DB_TABLES_INFO_SCHEMA_QUERY.format(db=database)
        df_tables = self.execute_sql(connection, query)
        return df_tables

    def get_all_ddls(self, connection, database: str) -> pd.DataFrame:
        """
        A method to get the DDLs for all the tables in the database.

        Parameters:
            connection (any): The connection object.
            database (str): The name of the database.

        Returns:
            DataFrame: A pandas DataFrame containing the DDLs for all the tables in the database.
        """
        self.validate_connection(connection)
        df_tables = self.get_table_names(connection, database)
        df_ddl = pd.DataFrame(columns=['Table', 'DDL'])
        for index, row in df_tables.iterrows():
            table_name = row.get('table_name')
            ddl_df = self.get_ddl(connection, table_name)
            df_ddl = df_ddl._append({'Table': table_name, 'DDL': ddl_df}, ignore_index=True)
        return df_ddl

    def get_ddl(self, connection, table_name: str, **kwargs) -> str:
        """
        A method to get the DDL for the table.

        Parameters:
            connection (any): The connection object.
            table_name (str): The name of the table.

        Returns:
            str: The DDL for the table.
        """
        self.validate_connection(connection)
        ddl_df = self.execute_sql(connection, POSTGRESQL_SHOW_CREATE_TABLE_QUERY.format(table=table_name))
        return ddl_df.get('create_statement').iloc[0]

    def get_dialect(self) -> str:
        return 'postgres'
