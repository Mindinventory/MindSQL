import json
import re

import sqlparse
from sqlparse.exceptions import SQLParseError
from .._utils import logger
from .._utils.constants import LOG_AND_RETURN_CONSTANT, JSON_FILE_ERROR_CONSTANT

log = logger.init_loggers("Helper")


def _sanitize_plotly_code(raw_plotly_code: str) -> str:
    """
    A method to sanitize the plotly code.

    Parameters:
        raw_plotly_code (str): The raw plotly code.

    Returns:
        str: The sanitized plotly code.
    """
    return raw_plotly_code.replace("fig.show()", "")


def has_select_and_semicolon(llm_response: str) -> bool:
    """
    A method to check if the LLM response contains a SELECT statement and a semicolon.

    Parameters:
        llm_response (str): The LLM response.

    Returns:
        bool: True if the LLM response contains a SELECT statement and a semicolon, False otherwise.
    """
    index_select = llm_response.upper().find("SELECT")
    index_semicolon = llm_response.find(";")

    return index_select != -1 and index_semicolon != -1 and index_select < index_semicolon


def extract_sql(llm_response: str) -> str:
    """
    A method to extract the SQL from the LLM response.

    Parameters:
        llm_response (str): The LLM response.

    Returns:
        str: The extracted SQL.
    """

    def log_and_return(extracted_sql: str) -> str:
        """
        A helper function to log and return the extracted SQL.

        Parameters:
            extracted_sql (str): The extracted SQL.

        Returns:
            str: The extracted SQL.
        """
        log.info(LOG_AND_RETURN_CONSTANT.format(llm_response, extracted_sql))
        return extracted_sql

    sql_match = re.search(r"```(sql)?\n(.+?)```", llm_response, re.DOTALL)
    if sql_match:
        return log_and_return(sql_match.group(2).replace("`", ""))
    elif has_select_and_semicolon(llm_response):
        start_sql = llm_response.find("SELECT")
        end_sql = llm_response.find(";")
        return log_and_return(llm_response[start_sql:end_sql + 1].replace("`", ""))
    return llm_response


def validate_sql(sql: str) -> bool:
    """
    A method to validate the SQL.

    Parameters:
        sql (str): The SQL.

    Returns:
        bool: True if the SQL is valid, False otherwise.
    """
    try:
        parsed_statements = sqlparse.parse(sql)

        if not parsed_statements:
            return False

        for statement in parsed_statements:
            if any(token.ttype in sqlparse.tokens.Error for token in statement.tokens):
                return False

        return has_select_and_semicolon(sql)

    except SQLParseError:
        return False


def load_json_to_dict(json_filepath: str) -> dict:
    """
    A method to load the JSON file to a dictionary.

    Parameters:
        json_filepath (str): The path to the JSON file.

    Returns:
        dict: The dictionary.
    """
    try:
        with open(json_filepath, 'r') as json_file:
            json_data = json.load(json_file)
            return json_data
    except Exception as e:
        log.info(JSON_FILE_ERROR_CONSTANT.format(e))
        return {}
