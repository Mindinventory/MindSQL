from typing import List
from urllib.parse import urlparse

import mysql.connector
import pandas as pd

from .._utils import logger
from .._utils.constants import SUCCESSFULLY_CONNECTED_TO_DB_CONSTANT, ERROR_CONNECTING_TO_DB_CONSTANT, \
    INVALID_DB_CONNECTION_OBJECT, ERROR_WHILE_RUNNING_QUERY, MYSQL_DB_TABLES_INFO_SCHEMA_QUERY, \
    MYSQL_SHOW_DATABASE_QUERY, MYSQL_SHOW_CREATE_TABLE_QUERY, CONNECTION_ESTABLISH_ERROR_CONSTANT
from . import IDatabase

log = logger.init_loggers("MySQL")


class MySql(IDatabase):
    def create_connection(self, url: str, **kwargs) -> any:
        """
        A method to create a connection with the database.

        Parameters:
            url (str): The URL in the format mysql://username:password@host:port/database_name
            **kwargs: Additional keyword arguments for the connection.

        Returns:
            any: The connection object.
        """
        url = urlparse(url)
        try:
            # Establish the connection using the URL string
            conn = mysql.connector.connect(host=url.hostname, port=url.port, user=url.username, password=url.password,
                                           database=url.path.lstrip('/'))

            if conn.is_connected():
                log.info(SUCCESSFULLY_CONNECTED_TO_DB_CONSTANT.format("MySQL"))
                return conn

        except mysql.connector.Error as e:
            log.info(ERROR_CONNECTING_TO_DB_CONSTANT.format("MySQL", e))

    def validate_connection(self, connection: any) -> None:
        """
        A function that validates if the provided connection is a MySQL connection.

        Parameters:
            connection: The connection object for accessing the database.

        Raises:
            ValueError: If the provided connection is not a MySQL connection.

        Returns:
            None
        """
        if connection is None:
            raise ValueError(CONNECTION_ESTABLISH_ERROR_CONSTANT)

        if not isinstance(connection, mysql.connector.connection_cext.CMySQLConnection):
            raise ValueError(INVALID_DB_CONNECTION_OBJECT.format("MySQL"))

    def execute_sql(self, connection, sql: str) -> pd.DataFrame:
        """
        A method to execute SQL on the database.

        Parameters:
            connection (any): The connection object.
            sql (str): The SQL to be executed.

        Returns:
            pd.DataFrame: The result of the SQL query.
        """
        try:
            self.validate_connection(connection)
            cursor = connection.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            column_names = [i[0] for i in cursor.description]
            df = pd.DataFrame(results, columns=column_names)
            cursor.close()
            return df
        except mysql.connector.Error as e:
            log.info(ERROR_WHILE_RUNNING_QUERY.format(e))

    def get_databases(self, connection) -> List[str]:
        """
        Get a list of databases from the given connection and SQL query.

        Parameters:
            connection (object): The connection object for the database.

        Returns:
            List[str]: A list of unique database names.
        """
        try:
            self.validate_connection(connection)
            df_databases = self.execute_sql(connection=connection, sql=MYSQL_SHOW_DATABASE_QUERY)
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
        df_tables = self.execute_sql(connection, MYSQL_DB_TABLES_INFO_SCHEMA_QUERY.format(database))
        return df_tables

    def get_all_ddls(self, connection, database: str) -> pd.DataFrame:
        """
        Get all DDLs from the specified database using the provided connection object.

        Parameters:
            connection (any): The connection object.
            database (str): The name of the database.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the DDLs for each table in the specified database.
        """
        self.validate_connection(connection)
        df_tables = self.get_table_names(connection, database)
        df_ddl = pd.DataFrame(columns=['Table', 'DDL'])
        for index, row in df_tables.iterrows():
            table_name = row['TABLE_NAME']
            ddl_df = self.get_ddl(connection, table_name)
            df_ddl = df_ddl._append({'Table': table_name, 'DDL': ddl_df}, ignore_index=True)
        return df_ddl

    def get_ddl(self, connection: any, table_name: str, **kwargs) -> str:
        """
        A method to get the DDL for the table.

        Parameters:
            connection (any): The connection object.
            table_name (str): The name of the table.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The DDL for the table.
        """
        ddl_df = self.execute_sql(connection, MYSQL_SHOW_CREATE_TABLE_QUERY.format(table_name))
        return ddl_df["Create Table"].iloc[0]

    def get_dialect(self) -> str:
        """
        A method to get the dialect of the database.

        Returns:
            str: The dialect of the database.
        """
        return 'mysql'
