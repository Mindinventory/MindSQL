import unittest
from unittest.mock import patch, MagicMock
import pyodbc
import pandas as pd
from mindsql.databases.sqlserver import SQLServer, ERROR_WHILE_RUNNING_QUERY, ERROR_CONNECTING_TO_DB_CONSTANT, \
    INVALID_DB_CONNECTION_OBJECT, CONNECTION_ESTABLISH_ERROR_CONSTANT
from mindsql.databases.sqlserver import log as logger


class TestSQLServer(unittest.TestCase):

    @patch('mindsql.databases.sqlserver.pyodbc.connect')
    def test_create_connection_success(self, mock_connect):
        mock_connect.return_value = MagicMock(spec=pyodbc.Connection)
        connection = SQLServer.create_connection(
            'DRIVER={ODBC Driver 17 for SQL Server};SERVER=server_name;DATABASE=database_name;UID=user;PWD=password')
        self.assertIsInstance(connection, pyodbc.Connection)

    @patch('mindsql.databases.sqlserver.pyodbc.connect')
    def test_create_connection_failure(self, mock_connect):
        mock_connect.side_effect = pyodbc.Error('Connection failed')
        with self.assertLogs(logger, level='ERROR') as cm:
            connection = SQLServer.create_connection(
                'DRIVER={ODBC Driver 17 for SQL Server};SERVER=server_name;DATABASE=database_name;UID=user;PWD=password')
            self.assertIsNone(connection)
            self.assertTrue(any(
                ERROR_CONNECTING_TO_DB_CONSTANT.format("SQL Server", 'Connection failed') in message for message in
                cm.output))

    @patch('mindsql.databases.sqlserver.pyodbc.connect')
    def test_execute_sql_success(self, mock_connect):
        # Mock the connection and cursor
        mock_connection = MagicMock(spec=pyodbc.Connection)
        mock_cursor = MagicMock()

        mock_connect.return_value = mock_connection
        mock_connection.cursor.return_value = mock_cursor

        # Mock cursor behavior
        mock_cursor.execute.return_value = None
        mock_cursor.description = [('column1',), ('column2',)]
        mock_cursor.fetchall.return_value = [(1, 'a'), (2, 'b')]

        connection = SQLServer.create_connection('fake_connection_string')
        sql = "SELECT * FROM table"
        sql_server = SQLServer()
        result = sql_server.execute_sql(connection, sql)
        expected_df = pd.DataFrame(data=[(1, 'a'), (2, 'b')], columns=['column1', 'column2'])
        pd.testing.assert_frame_equal(result, expected_df)

    @patch('mindsql.databases.sqlserver.pyodbc.connect')
    def test_execute_sql_failure(self, mock_connect):
        # Mock the connection and cursor
        mock_connection = MagicMock(spec=pyodbc.Connection)
        mock_cursor = MagicMock()

        mock_connect.return_value = mock_connection
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.execute.side_effect = pyodbc.Error('Query failed')

        connection = SQLServer.create_connection('fake_connection_string')
        sql = "SELECT * FROM table"
        sql_server = SQLServer()

        with self.assertLogs(logger, level='ERROR') as cm:
            result = sql_server.execute_sql(connection, sql)
            self.assertIsNone(result)
            self.assertTrue(any(ERROR_WHILE_RUNNING_QUERY.format('Query failed') in message for message in cm.output))

    @patch('mindsql.databases.sqlserver.pyodbc.connect')
    def test_get_databases_success(self, mock_connect):
        # Mock the connection and cursor
        mock_connection = MagicMock(spec=pyodbc.Connection)
        mock_cursor = MagicMock()

        mock_connect.return_value = mock_connection
        mock_connection.cursor.return_value = mock_cursor

        # Mock cursor behavior
        mock_cursor.execute.return_value = None
        mock_cursor.fetchall.return_value = [('database1',), ('database2',)]

        connection = SQLServer.create_connection('fake_connection_string')
        sql_server = SQLServer()
        result = sql_server.get_databases(connection)
        self.assertEqual(result, ['database1', 'database2'])

    @patch('mindsql.databases.sqlserver.pyodbc.connect')
    def test_get_databases_failure(self, mock_connect):
        # Mock the connection and cursor
        mock_connection = MagicMock(spec=pyodbc.Connection)
        mock_cursor = MagicMock()

        mock_connect.return_value = mock_connection
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.execute.side_effect = pyodbc.Error('Query failed')

        connection = SQLServer.create_connection('fake_connection_string')
        sql_server = SQLServer()

        with self.assertLogs(logger, level='ERROR') as cm:
            result = sql_server.get_databases(connection)
            self.assertEqual(result, [])
            self.assertTrue(any(ERROR_WHILE_RUNNING_QUERY.format('Query failed') in message for message in cm.output))

    @patch('mindsql.databases.sqlserver.SQLServer.execute_sql')
    def test_get_table_names_success(self, mock_execute_sql):
        mock_execute_sql.return_value = pd.DataFrame(data=[('schema1.table1',), ('schema2.table2',)],
                                                     columns=['table_name'])

        connection = MagicMock(spec=pyodbc.Connection)
        sql_server = SQLServer()
        result = sql_server.get_table_names(connection, 'database_name')
        expected_df = pd.DataFrame(data=[('schema1.table1',), ('schema2.table2',)], columns=['table_name'])
        pd.testing.assert_frame_equal(result, expected_df)

    @patch('mindsql.databases.sqlserver.SQLServer.execute_sql')
    def test_get_all_ddls_success(self, mock_execute_sql):
        mock_execute_sql.side_effect = [
            pd.DataFrame(data=[('schema1.table1',)], columns=['table_name']),
            pd.DataFrame(data=['CREATE TABLE schema1.table1 (...);'], columns=['SQLQuery'])
        ]

        connection = MagicMock(spec=pyodbc.Connection)
        sql_server = SQLServer()
        result = sql_server.get_all_ddls(connection, 'database_name')

        expected_df = pd.DataFrame(data=[{'Table': 'schema1.table1', 'DDL': 'CREATE TABLE schema1.table1 (...);'}])
        pd.testing.assert_frame_equal(result, expected_df)

    def test_validate_connection_success(self):
        connection = MagicMock(spec=pyodbc.Connection)
        sql_server = SQLServer()
        # Should not raise any exception
        sql_server.validate_connection(connection)

    def test_validate_connection_failure(self):
        sql_server = SQLServer()

        with self.assertRaises(ValueError) as cm:
            sql_server.validate_connection(None)
        self.assertEqual(str(cm.exception), CONNECTION_ESTABLISH_ERROR_CONSTANT)

        with self.assertRaises(ValueError) as cm:
            sql_server.validate_connection("InvalidConnectionObject")
        self.assertEqual(str(cm.exception), INVALID_DB_CONNECTION_OBJECT.format("SQL Server"))

    @patch('mindsql.databases.sqlserver.SQLServer.execute_sql')
    def test_get_ddl_success(self, mock_execute_sql):
        mock_execute_sql.return_value = pd.DataFrame(data=['CREATE TABLE schema1.table1 (...);'], columns=['SQLQuery'])

        connection = MagicMock(spec=pyodbc.Connection)
        sql_server = SQLServer()
        result = sql_server.get_ddl(connection, 'schema1.table1')
        self.assertEqual(result, 'CREATE TABLE schema1.table1 (...);')

    def test_get_dialect(self):
        sql_server = SQLServer()
        self.assertEqual(sql_server.get_dialect(), 'tsql')


if __name__ == '__main__':
    unittest.main()
