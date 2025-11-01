from typing import List
from urllib.parse import urlparse

import mariadb
import pandas as pd

from .._utils import logger
from .._utils.constants import SUCCESSFULLY_CONNECTED_TO_DB_CONSTANT, ERROR_CONNECTING_TO_DB_CONSTANT, \
    INVALID_DB_CONNECTION_OBJECT, ERROR_WHILE_RUNNING_QUERY, MARIADB_DB_TABLES_INFO_SCHEMA_QUERY, \
    MARIADB_SHOW_DATABASE_QUERY, MARIADB_SHOW_CREATE_TABLE_QUERY, CONNECTION_ESTABLISH_ERROR_CONSTANT
from . import IDatabase

log = logger.init_loggers("MariaDB")


class MariaDB(IDatabase):
    def create_connection(self, url: str, **kwargs) -> any:
        url = urlparse(url)
        try:
            connection_params = {
                'host': url.hostname,
                'port': url.port or int(kwargs.get('port', 3306)),
                'user': url.username,
                'password': url.password,
                'database': url.path.lstrip('/') if url.path else None,
                'autocommit': True,
            }
            
            connection_params = {k: v for k, v in connection_params.items() if v is not None}
            connection_params.update({k: v for k, v in kwargs.items() if k not in ['port']})
            
            conn = mariadb.connect(**connection_params)
            log.info(SUCCESSFULLY_CONNECTED_TO_DB_CONSTANT.format("MariaDB"))
            return conn

        except mariadb.Error as e:
            log.info(ERROR_CONNECTING_TO_DB_CONSTANT.format("MariaDB", str(e)))
            return None

    def validate_connection(self, connection: any) -> None:
        if connection is None:
            raise ValueError(CONNECTION_ESTABLISH_ERROR_CONSTANT)
        if not hasattr(connection, 'cursor'):
            raise ValueError(INVALID_DB_CONNECTION_OBJECT.format("MariaDB"))

    def execute_sql(self, connection, sql: str) -> pd.DataFrame:
        try:
            self.validate_connection(connection)
            cursor = connection.cursor()
            cursor.execute(sql)
            
            if sql.strip().upper().startswith(('CREATE', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER')):
                connection.commit()
                cursor.close()
                return pd.DataFrame()
            
            results = cursor.fetchall()
            if cursor.description:
                column_names = [i[0] for i in cursor.description]
                df = pd.DataFrame(results, columns=column_names)
            else:
                df = pd.DataFrame()
            cursor.close()
            return df
        except mariadb.Error as e:
            log.info(ERROR_WHILE_RUNNING_QUERY.format(e))
            return pd.DataFrame()

    def get_databases(self, connection) -> List[str]:
        try:
            self.validate_connection(connection)
            df_databases = self.execute_sql(connection=connection, sql=MARIADB_SHOW_DATABASE_QUERY)
        except Exception as e:
            log.info(e)
            return []
        return df_databases["Database"].unique().tolist()

    def get_table_names(self, connection, database: str) -> pd.DataFrame:
        self.validate_connection(connection)
        df_tables = self.execute_sql(connection, MARIADB_DB_TABLES_INFO_SCHEMA_QUERY.format(database))
        return df_tables

    def get_all_ddls(self, connection, database: str) -> pd.DataFrame:
        self.validate_connection(connection)
        df_tables = self.get_table_names(connection, database)
        df_ddl = pd.DataFrame(columns=['Table', 'DDL'])
        for index, row in df_tables.iterrows():
            table_name = row.get('TABLE_NAME') or row.get('table_name')
            if table_name:
                ddl_df = self.get_ddl(connection, table_name)
                df_ddl = df_ddl._append({'Table': table_name, 'DDL': ddl_df}, ignore_index=True)
        return df_ddl

    def get_ddl(self, connection: any, table_name: str, **kwargs) -> str:
        ddl_df = self.execute_sql(connection, MARIADB_SHOW_CREATE_TABLE_QUERY.format(table_name))
        return ddl_df["Create Table"].iloc[0]

    def get_dialect(self) -> str:
        return 'mysql'
