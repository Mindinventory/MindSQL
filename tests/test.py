import os

from mindsql.databases import Sqlite
from mindsql.llms import GoogleGenAi
from mindsql.vectorstores import chromadb as cd
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('API_KEY')
db_url = os.getenv('DB_URL')
example_path = os.getenv('EXAMPLE_PATH')


class MindsSqliteGeminiChroma(Sqlite, GoogleGenAi, cd.ChromaDB):
    def __init__(self, config):
        cd.ChromaDB.__init__(self, config=config)
        GoogleGenAi.__init__(self, config=config)
        Sqlite.__init__(self, config=config)


config = {
    'api_key': api_key
}

msql = MindsSqliteGeminiChroma(config=config)
conn = msql.create_connection(url=db_url)
ddls = msql.get_all_ddls(connection=conn, database='main')

for ind in ddls.index:
    msql.index_ddl(ddls["DDL"][ind])

items = msql.retrieve_relevant_ddl("Find the average unit price of products")


msql.index(bulk=True, path=example_path)
ques_sqls = msql.retrieve_relevant_question_sql("Find the average unit price of products")
response = msql.ask_db(question="Show all products whose unit price is more than 30", connection=conn, visualize=True,
                       table_names=['Product'])

chart = response["chart"]
chart.show()
