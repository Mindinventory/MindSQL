import re
import sys
from abc import ABC, abstractmethod
from typing import List, Union, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .. import _helper
from .._helper.helper import load_json_to_dict
from .._utils import prompts, logger
from .._utils.constants import CONNECTION_ESTABLISH_ERROR_CONSTANT, NO_DATA_FOUND_IN_JSON_CONSTANT, \
    BULK_DATA_SUCCESS_MESSAGE_CONSTANT, SQL_NOT_PROVIDED_CONSTANT, ADD_QUESTION_SQL_MESSAGE_CONSTANT, \
    ADD_DOCS_MESSAGE_CONSTANT, ADD_DDL_MESSAGE_CONSTANT, BULK_FALSE_ERROR
from .._utils.prompts import DDL_PROMPT, SQL_EXCEPTION_RESPONSE, FEW_SHOT_EXAMPLE, FINAL_RESPONSE_PROMPT, PLOTLY_PROMPT

log = logger.init_loggers("Minds Core")


class MindSQLCore(ABC):
    def __init__(self, config: any = None) -> None:
        """
        Initialize the class with an optional config parameter.

        Parameters:
            config (any): The configuration parameter.

        Returns:
            None
        """
        self.config = config

    def create_database_query(self, question: str, connection, tables: list, **kwargs) -> str:
        """
        A method to create the database query.

        Parameters:
            question (str): The question.
            connection (any): The connection object.
            tables (list): The list of tables.

        Returns:
            str: The database query.
        """
        question_sql_list = self.retrieve_relevant_question_sql(question, **kwargs)
        prompt = self.build_sql_prompt(question=question, connection=connection, question_sql_list=question_sql_list,
                                       tables=tables, **kwargs)
        log.info(prompt)
        llm_response = self.invoke(prompt, **kwargs)
        return _helper.helper.extract_sql(llm_response)

    # Vector Store Methods

    @abstractmethod
    def retrieve_relevant_question_sql(self, question: str, **kwargs) -> list:
        """
        This method retrieves the related question and SQL based on the provided question and optional keyword arguments.

        Parameters:
            question (str): The question for which related question and SQL is to be retrieved.
            **kwargs: Optional keyword arguments.

        Returns:
            list: A list of related question and SQL items.
        """
        pass

    @abstractmethod
    def retrieve_relevant_ddl(self, question: str, **kwargs) -> list:
        """
        A method to get related DDL statements based on a question and optional keyword arguments.

        Parameters:
            question: str - the question for which related DDL statements are to be retrieved
            **kwargs: additional keyword arguments

        Returns:
            list - a list of related DDL statements
        """
        pass

    @abstractmethod
    def retrieve_relevant_documentation(self, question: str, **kwargs) -> list:
        """
        A method to get related documentation based on a question and optional keyword arguments.

        Parameters:
            question: str - the question for which related documentation is to be retrieved
            **kwargs: additional keyword arguments

        Returns:
            list - a list of related documentation
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def index_ddl(self, ddl: str, **kwargs) -> str:
        """
        A method to add DDL to the vectorstore.

        Parameters:
            ddl (str): The DDL to be added to the vectorstore.
            **kwargs: Additional keyword arguments.

        Returns:
            str: A message confirming the successful addition of the DDL.
        """
        pass

    @abstractmethod
    def index_documentation(self, documentation: str, **kwargs) -> str:
        """
        A method to add documentation to the vectorstore.

        Parameters:
            documentation (str): The documentation to be added to the vectorstore.
            **kwargs: Additional keyword arguments.

        Returns:
            str: A message confirming the successful addition of the documentation.
        """
        pass

    @abstractmethod
    def fetch_all_vectorstore_data(self, **kwargs) -> pd.DataFrame:
        """
        A method to fetch all data from the vectorstore.

        Parameters:
            **kwargs: Additional keyword arguments.

        Returns:
            pd.DataFrame: The data from the vectorstore.
        """
        pass

    @abstractmethod
    def delete_vectorstore_data(self, item_id: str, **kwargs) -> bool:
        """
        A method to delete data from the vectorstore based on the provided item_id.

        Parameters:
            item_id (str): The unique identifier associated with the data to be deleted.
            **kwargs: Additional keyword arguments.

        Returns:
            bool: True if the deletion was successful, False otherwise.
        """
        pass

    # LLM Methods
    @abstractmethod
    def system_message(self, message: str) -> any:
        """
        A method to handle system messages.

        Parameters:
            message (str): The message received from the system.

        Returns:
            any: The return type of the function.
        """
        pass

    @abstractmethod
    def user_message(self, message: str) -> any:
        """
        A method to handle user messages.

        Parameters:
            message (str): The message received from the user.

        Returns:
            any: The return type of the function.
        """
        pass

    @abstractmethod
    def assistant_message(self, message: str) -> any:
        """
        A method to handle assistant messages.

        Parameters:
            message (str): The message received from the assistant.

        Returns:
            any: The return type of the function.
        """
        pass

    @abstractmethod
    def invoke(self, prompt, **kwargs) -> str:
        """
        A method to invoke the LLM.

        Parameters:
            prompt (str): The prompt to be sent to the LLM.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The response from the LLM.
        """
        pass

    # Database Methods
    @abstractmethod
    def create_connection(self, url: str, **kwargs) -> any:
        """
        A method to create a connection to the database.

        Parameters:
            url (str): The URL of the database.
            **kwargs: Additional keyword arguments.

        Returns:
            any: The connection object.
        """
        pass

    @abstractmethod
    def execute_sql(self, connection, sql: str) -> pd.DataFrame:
        """
        A method to execute SQL on the database.

        Parameters:
            connection (any): The connection object.
            sql (str): The SQL to be executed.

        Returns:
            pd.DataFrame: The result of the SQL query.
        """
        pass

    @abstractmethod
    def _get_databases(self, connection) -> List[str]:
        """
        A method to get the list of databases in the database.

        Parameters:
            connection (any): The connection object.

        Returns:
            List[str]: The list of databases.
        """
        pass

    @abstractmethod
    def _get_table_names(self, connection, database: str) -> pd.DataFrame:
        """
        A method to get the list of tables in the database.

        Parameters:
            connection (any): The connection object.
            database (str): The name of the database.

        Returns:
            pd.DataFrame: The list of tables.
        """
        pass

    @abstractmethod
    def get_all_ddls(self, connection: any, df: pd.DataFrame) -> pd.DataFrame:
        """
        A method to get all DDLs in the database.

        Parameters:
            connection (any): The connection object.
            df (pd.DataFrame): The dataframe.

        Returns:
            pd.DataFrame: The list of DDLs.
        """
        pass

    @staticmethod
    def _validate_connection(connection: any) -> None:
        """
        A method to validate the connection.

        Parameters:
            connection (any): The connection object.

        Raises:
            ValueError: If the connection is None.
        """
        if connection is None:
            raise ValueError(CONNECTION_ESTABLISH_ERROR_CONSTANT)

    @abstractmethod
    def get_ddl(self, connection: any, table_name: str, **kwargs) -> str:
        """
        A method to get the DDL of a table in the database.

        Parameters:
            connection (any): The connection object.
            table_name (str): The name of the table.

        Returns:
            str: The DDL of the table.
        """
        pass

    def get_dialect(self) -> str:
        """
        A method to get the dialect of the database

        Returns:
            str: The dialect of the database
        """
        pass

    # Core Methods

    @staticmethod
    def stuff_ddl_in_prompt(initial_prompt: str, ddl_list: list[str]) -> str:
        """
        A method to add DDL statements to the prompt.

        Parameters:
            initial_prompt (str): The initial prompt.
            ddl_list (list[str]): The list of DDL statements.

        Returns:
            str: The updated prompt with DDL statements.
        """
        if ddl_list:
            ddl_statements = "\n".join(ddl_list)
            prompt = f"{initial_prompt}\n{DDL_PROMPT.format(ddl_statements)}"
            return prompt
        return initial_prompt

    @staticmethod
    def stuff_documentation_in_prompt(initial_prompt: str, documentation_list: list[str]) -> str:
        """
        A method to add documentation statements to the prompt.

        Parameters:
            initial_prompt (str): The initial prompt.
            documentation_list (list[str]): The list of documentation statements.

        Returns:
            str: The updated prompt with documentation statements.
        """
        if documentation_list:
            doc_statements = "\n".join(documentation_list)
            prompt = f"{initial_prompt}\n{doc_statements}"
            return prompt
        return initial_prompt

    @staticmethod
    def stuff_sql_in_prompt(initial_prompt: str, sql_list: list[str]) -> str:
        """
        A method to add SQL statements to the prompt.

        Parameters:
            initial_prompt (str): The initial prompt.
            sql_list (list[str]): The list of SQL statements.

        Returns:
            str: The updated prompt with SQL statements.
        """
        if sql_list:
            sql_statements = "\n".join(sql_list)
            prompt = f"{initial_prompt}\n{sql_statements}"
            return prompt

    def build_sql_prompt(self, question: str, connection: any, question_sql_list: list[str], tables: list[str],
                         **kwargs) -> str:
        """
        A method to build the SQL prompt.

        Parameters:
            question (str): The question.
            connection (any): The connection object.
            question_sql_list (list[str]): The list of similar questions.
            tables (list[str]): The list of tables.

        Returns:
            str: The SQL prompt.
        """
        dialect_name = self.get_dialect()
        initial_prompt = self._create_initial_prompt(question_sql_list, dialect_name)

        ddl_statements = self._get_ddl_statements(connection, tables, question, **kwargs)
        initial_prompt = self.stuff_ddl_in_prompt(initial_prompt, ddl_statements)

        doc_statements = self.retrieve_relevant_documentation(question, **kwargs)
        initial_prompt = self.stuff_documentation_in_prompt(initial_prompt, doc_statements)
        final_prompt = f"{initial_prompt}\n'Question': {question}"
        return final_prompt

    @staticmethod
    def _create_initial_prompt(question_sql_list: list[str], dialect_name: str) -> str:
        """
        A method to create the initial prompt.

        Parameters:
            question_sql_list (list[str]): The list of similar questions.
            dialect_name (str): The dialect name.

        Returns:
            str: The initial prompt.
        """
        initial_prompt = prompts.DEFAULT_PROMPT.format(dialect_name=dialect_name)
        return f'{initial_prompt}\n{FEW_SHOT_EXAMPLE.format(MindSQLCore.__format_qsn_sql(question_sql_list))}'

    @staticmethod
    def __format_qsn_sql(question_sql_list: list):
        """
        A method to format the question and SQL.

        Parameters:
            question_sql_list (list): The list of question and SQL.

        Returns:
            str: The formatted string.
        """
        formatted_string = "\n\n"

        for query_dict in question_sql_list:
            formatted_string += "'Question': \"{}\"\n'SQLQuery': '{}'\n\n".format(query_dict.get('Question'),
                                                                                  query_dict.get('SQLQuery'))
        return formatted_string

    def _get_ddl_statements(self, connection: any, tables: list[str], question: str, **kwargs) -> list[str]:
        """
        A method to get the DDL statements.

        Parameters:
            connection (any): The connection object.
            tables (list[str]): The list of tables.
            question (str): The input question.

        Returns:
            list[str]: The list of DDL statements.
        """
        if tables and connection:
            ddl_statements = []
            for table_name in tables:
                ddl_statements.append(self.get_ddl(connection=connection, table_name=table_name))
        else:
            ddl_statements = self.retrieve_relevant_ddl(question, **kwargs)
        return ddl_statements

    def ask_db(self, connection, question: Union[str, None] = None, table_names: list = None, visualize: bool = False,
               **kwargs) -> dict:
        """
        A method to ask the database and return the result as a dictionary with the following keys:
        - sql (str): The SQL query.
        - sql_result (pd.DataFrame): The result of the SQL query.
        - response (str): The response from the LLM
        - chart (str): The chart
        - error (Exception): The error if any

        Parameters:
            connection (any): The connection object.
            question (str): The input question.
            table_names (list): The list of tables.
            visualize (bool): Whether to visualize the results.

        Returns:
            dict: The result dictionary.

        """

        result = {}
        try:
            sql = self.create_database_query(question=question, connection=connection, tables=table_names, **kwargs)
            result["sql"] = sql

            if _helper.helper.validate_sql(sql):
                df = self.execute_sql(connection, sql)
                result.update({"sql_result": df, "response": self.invoke(
                    FINAL_RESPONSE_PROMPT.format(context_df=df, user_query=question)),
                               "chart": self.visualize(question, df, visualize)})
                log.info(f"Query: {question} \nLLM Response: {result.get('response')}")
            else:
                log.info(SQL_EXCEPTION_RESPONSE)

        except Exception as e:
            log.warning(f"An unexpected error occurred: {e}")
            result["error"] = e
        return result

    def index(self, question: str = None, sql: str = None, ddl: str = None, documentation: str = None,
              bulk: bool = False, path: str = None) -> str:
        """
        A method to add a question and SQL pair to the vectorstore.

        Parameters:
            question (str): The question to be added to the vectorstore.
            sql (str): The SQL to be added to the vectorstore.
            ddl (str): The DDL to be added to the vectorstore.
            documentation (str): The documentation to be added to the vectorstore.
            bulk (bool): Whether to add bulk data.
            path (str): The path to the JSON file.

        Returns:
            str: A message confirming the successful addition of the question and SQL pair.
        """
        if bulk and path:
            json_data = load_json_to_dict(path)
            if json_data:
                for item in json_data:
                    if 'Question' in item and 'SQLQuery' in item:
                        self.index_question_sql(question=item.get('Question'), sql=item.get('SQLQuery'))
            else:
                raise Exception(NO_DATA_FOUND_IN_JSON_CONSTANT.format(path))
            log.info(BULK_DATA_SUCCESS_MESSAGE_CONSTANT)

        if path and not bulk:
            raise ValueError(BULK_FALSE_ERROR)

        if question and not sql:
            raise ValueError(SQL_NOT_PROVIDED_CONSTANT)

        if question and sql:
            log.info(ADD_QUESTION_SQL_MESSAGE_CONSTANT)
            return self.index_question_sql(question=question, sql=sql)

        if documentation:
            log.info(ADD_DOCS_MESSAGE_CONSTANT)
            return self.index_documentation(documentation)

        if ddl:
            log.info(ADD_DDL_MESSAGE_CONSTANT)
            return self.index_ddl(ddl)

    @staticmethod
    def __extract_plotly_code(markdown_string: str) -> str:
        """
        A method to extract the plotly code from the markdown string.

        Parameters:
            markdown_string (str): The markdown string.

        Returns:
            str: The extracted plotly code.
        """
        pattern = r"```[\w\s]*python\n([\s\S]*?)```|```([\s\S]*?)```"

        matches = re.findall(pattern, markdown_string, re.IGNORECASE)
        python_code = [match[0] or match[1] for match in matches]

        if not python_code:
            return markdown_string

        extracted_code = ''.join(python_code)
        sanitized_code = extracted_code.replace("fig.show()", "")
        return sanitized_code

    @staticmethod
    def __execute_plotly_code(plotly_code: str, data: pd.DataFrame) -> Optional[go.Figure]:
        """
        A method to execute the plotly code.

        Parameters:
            plotly_code (str): The plotly code.
            data (pd.DataFrame): The data.

        Returns:
            Optional[go.Figure]: The chart.
        """
        _locals = {"pd": pd, "go": go, "px": px, "df": data, "make_subplots": make_subplots}
        exec(plotly_code, globals(), _locals)
        return _locals.get("chart", None)

    def visualize(self, query: str, data: pd.DataFrame, visualize: bool = False) -> Optional[go.Figure]:
        """
        A method to visualize the data.

        Parameters:
            query (str): The query.
            data (pd.DataFrame): The data.
            visualize (bool): Whether to visualize the data.

        Returns:
            Optional[go.Figure]: The chart.
        """
        if visualize and not data.empty:
            try:
                if len(data.columns) == 1:
                    log.warning("Cannot create a chart for a one-dimensional DataFrame with only one column.")
                    return None
                prompt = PLOTLY_PROMPT.format(query=query, df=data)
                result = self.invoke(prompt)
                plotly_code = self.__extract_plotly_code(result)
                fig = self.__execute_plotly_code(plotly_code, data)

                try:
                    if 'IPython' in sys.modules:
                        # Running in a Jupyter environment
                        display = __import__("IPython.display", fromlist=["display"]).display
                        image = __import__("IPython.display", fromlist=["Image"]).Image
                        img_bytes = fig.to_image(format="png", scale=2)
                        display(image(img_bytes))
                except Exception as e:
                    log.warning(f"Unable to display the Plotly figure: {e}")

                return fig
            except Exception as e:
                log.warning(f"An unexpected error occurred while generating chart: {e}")
                return None
