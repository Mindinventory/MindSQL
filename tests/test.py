import os
from dotenv import load_dotenv
from mindsql.core import MindSQLCore
from mindsql.databases import Sqlite
from mindsql.llms import GoogleGenAi
from mindsql.vectorstores import ChromaDB

load_dotenv()

api_key = os.getenv('API_KEY')
db_url = os.getenv('DB_URL')
example_path = os.getenv('EXAMPLE_PATH')

# Set up configuration dictionary
config = {'api_key': api_key}

# Create MindSQLCore instance with configured llm, vectorstore, and database
minds = MindSQLCore(
    llm=GoogleGenAi(config=config),
    vectorstore=ChromaDB(),
    database=Sqlite()
)

# Create a database connection using the specified URL
conn = minds.database.create_connection(url=db_url)

# Index all Data Definition Language (DDL) statements in the 'main' database into the vectorstore
minds.index_all_ddls(connection=conn, db_name='main')

# Index question-sql pair in bulk from the specified example path
minds.index(bulk=True, path=example_path)

# Ask a question to the database and visualize the result
response = minds.ask_db(
    question="Show all products whose unit price is more than 30",
    connection=conn,
    visualize=True
)

# Extract and display the chart from the response
chart = response["chart"]
chart.show()
