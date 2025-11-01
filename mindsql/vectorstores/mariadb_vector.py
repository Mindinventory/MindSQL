import json
import uuid
from typing import List
import mariadb
import pandas as pd
from sentence_transformers import SentenceTransformer
from mindsql.vectorstores import IVectorstore


class MariaDBVectorStore(IVectorstore):
    def __init__(self, config=None):
        if config is None:
            raise ValueError("MariaDB configuration is required")
        
        self.collection_name = config.get('collection_name', 'mindsql_vectors')
        self.connection_params = {
            'host': config.get('host', 'localhost'),
            'port': config.get('port', 3306),
            'user': config.get('user'),
            'password': config.get('password'),
        }
        
        if 'database' in config and config['database']:
            self.connection_params['database'] = config['database']
        
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dimension = 384
        self._init_database()
    
    def _init_database(self):
        try:
            conn = mariadb.connect(**self.connection_params)
            cursor = conn.cursor()
            
            cursor.execute(f"DROP TABLE IF EXISTS {self.collection_name}")
            cursor.execute(f"DROP TABLE IF EXISTS {self.collection_name}_sql_pairs")
            
            cursor.execute(f"""
                CREATE TABLE {self.collection_name} (
                    id VARCHAR(36) PRIMARY KEY,
                    document TEXT NOT NULL,
                    embedding VECTOR({self.dimension}) NOT NULL,
                    metadata JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_created_at (created_at),
                    FULLTEXT(document)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            cursor.execute(f"""
                CREATE TABLE {self.collection_name}_sql_pairs (
                    id VARCHAR(36) PRIMARY KEY,
                    question TEXT NOT NULL,
                    sql_query TEXT NOT NULL,
                    embedding VECTOR({self.dimension}) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FULLTEXT(question, sql_query)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MariaDB vector store: {e}")
    
    def _format_vector_for_insertion(self, embedding_array):
        if len(embedding_array) != self.dimension:
            raise ValueError(f"Expected {self.dimension} dimensions, got {len(embedding_array)}")
        return '[' + ','.join(f'{float(x)}' for x in embedding_array) + ']'
    
    def add_ddl(self, ddl: str):
        embedding = self.embedding_model.encode(ddl).tolist()
        vector_json = self._format_vector_for_insertion(embedding)
        
        conn = mariadb.connect(**self.connection_params)
        cursor = conn.cursor()
        doc_id = str(uuid.uuid4())
        cursor.execute(f"""
            INSERT INTO {self.collection_name} 
            (id, document, embedding, metadata) 
            VALUES (?, ?, VEC_FromText(?), ?)
        """, (doc_id, ddl, vector_json, json.dumps({"type": "ddl"})))
        conn.commit()
        cursor.close()
        conn.close()
    
    def add_question_sql(self, question: str, sql: str):
        embedding = self.embedding_model.encode(question).tolist()
        vector_json = self._format_vector_for_insertion(embedding)
        
        conn = mariadb.connect(**self.connection_params)
        cursor = conn.cursor()
        doc_id = str(uuid.uuid4())
        cursor.execute(f"""
            INSERT INTO {self.collection_name}_sql_pairs 
            (id, question, sql_query, embedding) 
            VALUES (?, ?, ?, VEC_FromText(?))
        """, (doc_id, question, sql, vector_json))
        conn.commit()
        cursor.close()
        conn.close()
    
    def get_similar_question_sql(self, question: str, n_results: int = 5):
        conn = mariadb.connect(**self.connection_params)
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT question, sql_query,
                   MATCH(question, sql_query) AGAINST (? IN NATURAL LANGUAGE MODE) as text_score
            FROM {self.collection_name}_sql_pairs 
            WHERE MATCH(question, sql_query) AGAINST (? IN NATURAL LANGUAGE MODE)
            ORDER BY text_score DESC LIMIT ?
        """, (question, question, n_results))
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return [{'question': r[0], 'sql': r[1], 'similarity': r[2], 'text_score': r[2]} for r in results]
    
    def retrieve_relevant_ddl(self, question: str, **kwargs) -> list:
        conn = mariadb.connect(**self.connection_params)
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT document FROM {self.collection_name} 
            WHERE JSON_EXTRACT(metadata, '$.type') = 'ddl'
            ORDER BY created_at DESC LIMIT ?
        """, (kwargs.get('n_results', 5),))
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return [row[0] for row in results]
    
    def retrieve_relevant_documentation(self, question: str, **kwargs) -> list:
        conn = mariadb.connect(**self.connection_params)
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT document FROM {self.collection_name} 
            WHERE JSON_EXTRACT(metadata, '$.type') = 'documentation'
            ORDER BY created_at DESC LIMIT ?
        """, (kwargs.get('n_results', 5),))
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return [row[0] for row in results]
    
    def retrieve_relevant_question_sql(self, question: str, **kwargs) -> list:
        return self.get_similar_question_sql(question, kwargs.get('n_results', 3))
    
    def index_question_sql(self, question: str, sql: str, **kwargs) -> str:
        try:
            self.add_question_sql(question, sql)
            return "Successfully added question-SQL pair"
        except Exception as e:
            return f"Failed: {e}"
    
    def index_ddl(self, ddl: str, **kwargs) -> str:
        try:
            self.add_ddl(ddl)
            return "Successfully added DDL"
        except Exception as e:
            return f"Failed: {e}"
    
    def index_documentation(self, documentation: str, **kwargs) -> str:
        try:
            embedding = self.embedding_model.encode(documentation).tolist()
            vector_json = self._format_vector_for_insertion(embedding)
            
            conn = mariadb.connect(**self.connection_params)
            cursor = conn.cursor()
            doc_id = str(uuid.uuid4())
            cursor.execute(f"""
                INSERT INTO {self.collection_name} 
                (id, document, embedding, metadata) 
                VALUES (?, ?, VEC_FromText(?), ?)
            """, (doc_id, documentation, vector_json, json.dumps({"type": "documentation"})))
            conn.commit()
            cursor.close()
            conn.close()
            return "Successfully added documentation"
        except Exception as e:
            return f"Failed: {e}"
    
    def fetch_all_vectorstore_data(self, **kwargs) -> pd.DataFrame:
        conn = mariadb.connect(**self.connection_params)
        main_df = pd.read_sql(f"SELECT id, document, created_at FROM {self.collection_name}", conn)
        sql_pairs_df = pd.read_sql(f"SELECT id, question, sql_query, created_at FROM {self.collection_name}_sql_pairs", conn)
        conn.close()
        
        data = []
        for _, row in main_df.iterrows():
            data.append({'id': row['id'], 'content': row['document'], 'type': 'document', 'created_at': row['created_at']})
        for _, row in sql_pairs_df.iterrows():
            data.append({'id': row['id'], 'content': f"Q: {row['question']} | SQL: {row['sql_query']}", 
                        'type': 'question_sql', 'created_at': row['created_at']})
        return pd.DataFrame(data)
    
    def delete_vectorstore_data(self, item_id: str, **kwargs) -> bool:
        conn = mariadb.connect(**self.connection_params)
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {self.collection_name} WHERE id = ?", (item_id,))
        main_deleted = cursor.rowcount
        cursor.execute(f"DELETE FROM {self.collection_name}_sql_pairs WHERE id = ?", (item_id,))
        pairs_deleted = cursor.rowcount
        conn.commit()
        cursor.close()
        conn.close()
        return (main_deleted + pairs_deleted) > 0
    
    def add_documents(self, documents: List[str]):
        for doc in documents:
            self.add_ddl(doc)
