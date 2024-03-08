import json
import os
import uuid
from typing import List

import chromadb
import pandas as pd
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from . import IVectorstore

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="WhereIsAI/UAE-Large-V1")


class ChromaDB(IVectorstore):
    def __init__(self, config=None):
        if config is not None:
            directory = config.get("path", ".")
            self.embedding_function = config.get("embedding_function", sentence_transformer_ef)
        else:
            directory = "./vectorstore"
            if not os.path.exists(directory):
                os.makedirs(directory)
            self.embedding_function = sentence_transformer_ef

        self.chroma_client = chromadb.PersistentClient(path=directory, settings=Settings(anonymized_telemetry=False))
        self.documentation_collection = self.chroma_client.get_or_create_collection(name="documentation",
                                                                                    embedding_function=self.embedding_function)
        self.ddl_collection = self.chroma_client.get_or_create_collection(name="ddl",
                                                                          embedding_function=self.embedding_function)
        self.sql_collection = self.chroma_client.get_or_create_collection(name="sql",
                                                                          embedding_function=self.embedding_function)

    def index_question_sql(self, question: str, sql: str, **kwargs) -> str:
        """
            Add a question and its corresponding SQL query to the vectorstore.

            Args:
                question (str): The question to be associated with the SQL query.
                sql (str): The SQL query to be stored.
                **kwargs: Additional keyword arguments (optional).

            Returns:
                str: A unique identifier (chunk_id) associated with the stored question and SQL query.

            Example:
                chunk_id = add_question_sql("What is the total sales?", "SELECT SUM(sales) FROM transactions")
        """
        question_sql_json = json.dumps({"Question": question, "SQLQuery": sql, }, ensure_ascii=False, )
        chunk_id = str(uuid.uuid4()) + "-sql"
        self.sql_collection.add(documents=question_sql_json, ids=chunk_id, )

        return chunk_id

    def index_ddl(self, ddl: str, **kwargs) -> str:
        """
        Add a Data Definition Language (DDL) statement to the vectorstore.

        Args:
            ddl (str): The DDL statement to be stored.
            **kwargs: Additional keyword arguments (optional).
                - table (str): Name of the table associated with the DDL statement.

        Returns:
            str: A unique identifier (chunk_id) associated with the stored DDL statement.

        Example:
            chunk_id = add_ddl("CREATE TABLE employees (id INT, name VARCHAR(255))", table="employees")
        """
        chunk_id = str(uuid.uuid4()) + "-ddl"
        collection_params = {"documents": ddl, "ids": chunk_id, }
        if 'table' in kwargs:
            collection_params["metadatas"] = {"table_name": kwargs['table']}
        self.ddl_collection.add(**collection_params)
        return chunk_id

    def index_documentation(self, documentation: str, **kwargs) -> str:
        """
            Add documentation to the database.

            Args:
                documentation (str): The documentation content to be stored.
                **kwargs: Additional keyword arguments (optional).

            Returns:
                str: A unique identifier (chunk_id) associated with the stored documentation.

            Example:
                chunk_id = add_documentation("This function performs data validation.")
        """
        chunk_id = str(uuid.uuid4()) + "-doc"
        self.documentation_collection.add(documents=documentation, ids=chunk_id, )
        return chunk_id

    def fetch_all_vectorstore_data(self, **kwargs) -> pd.DataFrame:
        """
        Retrieve training data from different collections in the database.

        Args:
            **kwargs: Additional keyword arguments (optional).

        Returns: pd.DataFrame: A DataFrame containing training data with columns 'id', 'question', 'content',
        and 'training_data_type'.

        Example:
            training_df = get_training_data()
        """
        sql_data = self.sql_collection.get()

        df = pd.DataFrame()

        if sql_data is not None:
            # Extract the documents and ids
            documents = [json.loads(doc) for doc in sql_data["documents"]]
            ids = sql_data["ids"]

            # Create a DataFrame
            df_sql = pd.DataFrame({"id": ids, "question": [doc["question"] for doc in documents],
                                   "content": [doc["sql"] for doc in documents], })

            df_sql["training_data_type"] = "sql"

            df = pd.concat([df, df_sql])

        ddl_data = self.ddl_collection.get()

        if ddl_data is not None:
            # Extract the documents and ids
            documents = [doc for doc in ddl_data["documents"]]
            ids = ddl_data["ids"]

            # Create a DataFrame
            df_ddl = pd.DataFrame(
                {"id": ids, "question": [None for doc in documents], "content": [doc for doc in documents], })

            df_ddl["training_data_type"] = "ddl"

            df = pd.concat([df, df_ddl])

        doc_data = self.documentation_collection.get()

        if doc_data is not None:
            # Extract the documents and ids
            documents = [doc for doc in doc_data["documents"]]
            ids = doc_data["ids"]

            # Create a DataFrame
            df_doc = pd.DataFrame(
                {"id": ids, "question": [None for doc in documents], "content": [doc for doc in documents], })

            df_doc["training_data_type"] = "documentation"

            df = pd.concat([df, df_doc])

        return df

    def delete_vectorstore_data(self, item_id: str, **kwargs) -> bool:
        """
        Remove training data from the respective collection based on the provided item_id.

        Args:
            item_id (str): The unique identifier associated with the training data.
            **kwargs: Additional keyword arguments (optional).

        Returns:
            bool: True if the removal was successful, False otherwise.

        Example:
            result = remove_training_data("example-id-sql")
        """
        if item_id.endswith("-sql"):
            self.sql_collection.delete(ids=[item_id])
            return True
        elif item_id.endswith("-ddl"):
            self.ddl_collection.delete(ids=[item_id])
            return True
        elif item_id.endswith("-doc"):
            self.documentation_collection.delete(ids=[item_id])
            return True
        else:
            return False

    def remove_collection(self, collection_name: str) -> bool:
        """
        This function can reset the collection to empty state.

        Args:
            collection_name (str): sql or ddl or documentation

        Returns:
            bool: True if collection is deleted, False otherwise
        """
        if collection_name == "sql":
            self.chroma_client.delete_collection(name="sql")
            self.sql_collection = self.chroma_client.get_or_create_collection(name="sql",
                                                                              embedding_function=self.embedding_function)
            return True
        elif collection_name == "ddl":
            self.chroma_client.delete_collection(name="ddl")
            self.ddl_collection = self.chroma_client.get_or_create_collection(name="ddl",
                                                                              embedding_function=self.embedding_function)
            return True
        elif collection_name == "documentation":
            self.chroma_client.delete_collection(name="documentation")
            self.documentation_collection = self.chroma_client.get_or_create_collection(name="documentation",
                                                                                        embedding_function=self.embedding_function)
            return True
        else:
            return False

    @staticmethod
    def _extract_documents(query_results) -> list:
        """
        Static method to extract the documents from the QueryResult.

        Args:
            query_results (chromadb.api.types.QueryResult): The dataframe to use.

        Returns:
            List[str] or None: The extracted documents, or an empty list or single document if an error occurred.
        """
        if query_results is None:
            return []
        if ('documents' in query_results and query_results['documents'] is not None and len(
                query_results['documents']) > 0):
            documents = query_results["documents"]

            if len(documents) == 1 and isinstance(documents[0], list):
                try:
                    documents = [json.loads(doc) for doc in documents[0]]
                except Exception as e:
                    return documents[0]

            return documents

    def retrieve_relevant_question_sql(self, question: str, **kwargs) -> list:
        """
        Get a list of similar questions based on the provided question using the SQL collection.

        Args:
            question (str): The question for which similar questions are sought.
            **kwargs: Additional keyword arguments (optional).

        Returns:
            list: A list of similar questions.

        Example:
            similar_questions = get_similar_question_sql("How to retrieve total sales?")
        """
        n = kwargs.get("n_results", 2)
        return ChromaDB._extract_documents(self.sql_collection.query(query_texts=[question], n_results=n))

    def retrieve_relevant_ddl(self, question: str, **kwargs) -> list:
        """
        Get a list of related Data Definition Language (DDL) statements based on the provided question.

        Args:
            question (str): The question for which related DDL statements are sought.
            **kwargs: Additional keyword arguments (optional).

        Returns:
            list: A list of related DDL statements.

        Example:
            related_ddl = get_related_ddl("How to create a table for employee records?")
        """
        n = kwargs.get("n_results", 2)
        return ChromaDB._extract_documents(self.ddl_collection.query(query_texts=[question], n_results=n, ))

    def retrieve_relevant_documentation(self, question: str, **kwargs) -> list:
        """
        Get a list of related documentation based on the provided question.

        Args:
            question (str): The question for which related documentation is sought.
            **kwargs: Additional keyword arguments (optional).

        Returns:
            list: A list of related documentation.

        Example:
            related_docs = get_related_documentation("How to use the data validation function?")
        """
        n = kwargs.get("n_results", 2)
        return ChromaDB._extract_documents(self.documentation_collection.query(query_texts=[question], n_results=n, ))
