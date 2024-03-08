import json
import os
import uuid

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from . import IVectorstore

sentence_transformer_ef = SentenceTransformer("WhereIsAI/UAE-Large-V1")


class Faiss(IVectorstore):
    def __init__(self, config=None):
        """
        Initialize the FAISS vector store.

        Parameters:
            config (any): The configuration parameter.

        Returns:
            None
        """
        if config is not None:
            directory = config.get("path", ".")
            self.dimension = config.get("dimension", 1024)
            self.embedding_function = config.get("embedding_function", sentence_transformer_ef)
            self.index_builder = config.get("index_builder", faiss.IndexFlatL2)
            self.metric_type = config.get("metric_type", "euclidean")
        else:
            directory = "./vectorstore"
            if not os.path.exists(directory):
                os.makedirs(directory)

            self.dimension = 1024
            self.embedding_function = sentence_transformer_ef
            self.index_builder = faiss.IndexFlatL2
            self.metric_type = "euclidean"

        self.sql_index, self.sql_chunk_ids, self.sql_index_mapping = self._initialize_index(
            os.path.join(directory, "sql"))
        self.ddl_index, self.ddl_chunk_ids, self.ddl_index_mapping = self._initialize_index(
            os.path.join(directory, "ddl"))
        self.documentation_index, self.documentation_chunk_ids, self.documentation_index_mapping = self._initialize_index(
            os.path.join(directory, "documentation"))
        self.ddl_metadata = {}

    def _initialize_index(self, index_folder: str) -> tuple:
        """
        Initializes the index for the given index_folder.

        Parameters:
        - index_folder (str): The folder path where the index will be stored.

        Returns:
        tuple: A tuple containing the following elements:
            - index: The initialized index.
            - chunk_ids (list): An empty list to store chunk IDs.
            - index_mapping (dict): An empty dictionary to store index mapping.
        """
        index = self.index_builder(self.dimension)
        chunk_ids = []
        index_mapping = {}
        return index, chunk_ids, index_mapping

    def index_question_sql(self, question: str, sql: str, **kwargs) -> str:
        """
        Adds a question and its corresponding SQL query to the SQL index.

        Parameters:
        - question (str): The question to be associated with the SQL query.
        - sql (str): The SQL query.
        - **kwargs: Additional keyword arguments.

        Returns:
        str: The chunk ID generated for the added question-SQL pair.
        """
        question_sql_json = json.dumps({"Question": question, "SQLQuery": sql}, ensure_ascii=False)
        chunk_id = str(uuid.uuid4()) + "-sql"

        vectors = self.embedding_function.encode([question_sql_json]).astype('float32')
        self.sql_index.add(vectors)
        self.sql_chunk_ids.append(question_sql_json)
        self.sql_index_mapping[chunk_id] = len(self.sql_chunk_ids) - 1

        return chunk_id

    def index_ddl(self, ddl: str, **kwargs) -> str:
        """
        Adds a Data Definition Language (DDL) statement to the DDL index.

        Parameters:
        - ddl (str): The DDL statement to be added.
        - **kwargs: Additional keyword arguments.

        Returns:
        str: The chunk ID generated for the added DDL statement.
        """
        chunk_id = str(uuid.uuid4()) + "-ddl"

        table_name = kwargs.get('table', None)
        vectors = self.embedding_function.encode([ddl]).astype('float32')
        self.ddl_index.add(vectors)
        self.ddl_chunk_ids.append(ddl)
        self.ddl_index_mapping[chunk_id] = len(self.ddl_chunk_ids) - 1
        self.ddl_metadata[chunk_id] = {'table': table_name}

        return chunk_id

    def index_documentation(self, documentation: str, **kwargs) -> str:
        """
        Adds documentation text to the documentation index.

        Parameters:
        - documentation (str): The documentation text to be added.
        - **kwargs: Additional keyword arguments.

        Returns:
        str: The chunk ID generated for the added documentation.
        """
        chunk_id = str(uuid.uuid4()) + "-doc"

        vectors = self.embedding_function.encode([documentation]).astype('float32')
        self.documentation_index.add(vectors)
        self.documentation_chunk_ids.append(documentation)
        self.documentation_index_mapping[chunk_id] = len(self.documentation_chunk_ids) - 1

        return chunk_id

    def fetch_all_vectorstore_data(self, **kwargs) -> pd.DataFrame:
        """
        Retrieves training data from the indexes and organizes it into a pandas DataFrame.

        Parameters:
        - **kwargs: Additional keyword arguments.

        Returns:
        pd.DataFrame: A DataFrame containing the training data with columns:
            - 'id': The chunk ID.
            - 'question': Empty placeholder (None).
            - 'content': Empty placeholder (None).
            - 'training_data_type': The type of training data ('sql', 'ddl', or 'documentation').
        """
        combined_data = []

        for index, chunk_ids, index_mapping, index_name in zip(
                [self.sql_index, self.ddl_index, self.documentation_index],
                [self.sql_chunk_ids, self.ddl_chunk_ids, self.documentation_chunk_ids],
                [self.sql_index_mapping, self.ddl_index_mapping, self.documentation_index_mapping],
                ["sql", "ddl", "documentation"]):
            n_total = index.n_total
            if n_total > 0:
                _, ids = index.search(np.arange(n_total), n_total)
                for idx, doc_vec in enumerate(ids):
                    combined_data.append([chunk_ids[idx], None, None, index_name])

        cols = ['id', 'question', 'content', 'training_data_type']
        return pd.DataFrame(combined_data, columns=cols)

    def delete_vectorstore_data(self, item_id: str, **kwargs) -> bool:
        """
        Removes a training data item identified by its item_id from the respective index.

        Parameters:
        - item_id (str): The ID of the item to be removed.
        - **kwargs: Additional keyword arguments.

        Returns:
        bool: True if the item was successfully removed, False otherwise.
        """
        if item_id.startswith("sql"):
            index = self.sql_index
            chunk_ids = self.sql_chunk_ids
            index_mapping = self.sql_index_mapping
        elif item_id.startswith("ddl"):
            index = self.ddl_index
            chunk_ids = self.ddl_chunk_ids
            index_mapping = self.ddl_index_mapping
        elif item_id.startswith("doc"):
            index = self.documentation_index
            chunk_ids = self.documentation_chunk_ids
            index_mapping = self.documentation_index_mapping
        else:
            return False

        doc_index = index_mapping[item_id]

        index.remove_ids(np.array([doc_index]))

        del chunk_ids[doc_index]

        for i, chunk_id in enumerate(chunk_ids):
            index_mapping[chunk_id] = i

        return True

    def remove_collection(self, collection_name: str) -> bool:
        """
        Removes all items from a specified collection.

        Parameters:
        - collection_name (str): The name of the collection to be removed.

        Returns:
        bool: True if the collection was successfully removed, False otherwise.
        """
        if collection_name == "sql":
            index = self.sql_index
            chunk_ids = self.sql_chunk_ids
            index_mapping = self.sql_index_mapping
        elif collection_name == "ddl":
            index = self.ddl_index
            chunk_ids = self.ddl_chunk_ids
            index_mapping = self.ddl_index_mapping
        elif collection_name == "documentation":
            index = self.documentation_index
            chunk_ids = self.documentation_chunk_ids
            index_mapping = self.documentation_index_mapping
        else:
            return False

        index.reset()
        chunk_ids.clear()
        index_mapping.clear()

        return True

    def retrieve_relevant_question_sql(self, question: str, **kwargs) -> list:
        """
        Retrieves similar question-SQL pairs based on the provided question.

        Parameters:
        - question (str): The question for which similar question-SQL pairs are to be retrieved.
        - **kwargs: Additional keyword arguments.
                    - k (int): Number of similar question-SQL pairs to retrieve (default is 2).

        Returns:
        list: A list of IDs representing the similar question-SQL pairs.
        """
        vectors = self.embedding_function.encode([question]).astype('float32')
        distances, indices = self.sql_index.search(vectors, kwargs.pop('k', 2))
        result = []
        if np.all(indices[0] == -1):
            return result

        for idx in indices[0]:
            if idx != -1:
                question_sql_json = self.sql_chunk_ids[idx]
                question_sql_dict = json.loads(question_sql_json)
                result.append(question_sql_dict)
        return result

    def retrieve_relevant_ddl(self, question: str, **kwargs) -> list:
        """
        Retrieves related Data Definition Language (DDL) statements based on the provided question.

        Parameters:
        - question (str): The question for which related DDL statements are to be retrieved.
        - **kwargs: Additional keyword arguments.
                    - k (int): Number of related DDL statements to retrieve (default is 2).

        Returns:
            list: A list of related DDL statements.
        """
        vectors = self.embedding_function.encode([question]).astype('float32')
        distances, indices = self.ddl_index.search(vectors, kwargs.pop('k', 2))
        result = []
        if np.all(indices[0] == -1):
            return result

        for idx in indices[0]:
            if idx != -1:
                ddl_statement = self.ddl_chunk_ids[idx]
                result.append(ddl_statement)
        return result

    def retrieve_relevant_documentation(self, question: str, **kwargs) -> list:
        """
        Retrieves related documentation based on the provided question.

        Parameters:
        - question (str): The question for which related documentation is to be retrieved.
        - **kwargs: Additional keyword arguments.
                    - k (int): Number of related documentation items to retrieve (default is 2).

        Returns:
        list: A list of IDs representing the related documentation items.
        """
        vectors = self.embedding_function.encode([question]).astype('float32')
        distances, indices = self.documentation_index.search(vectors, kwargs.pop('k', 2))
        result = []
        if np.all(indices[0] == -1):
            return result

        for idx in indices[0]:
            if idx != -1:
                doc_statement = self.documentation_chunk_ids[idx]
                result.append(doc_statement)
        return result
