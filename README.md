# 🧠 MindSQL

MindSQL is a Python RAG (Retrieval-Augmented Generation) Library designed to streamline the interaction between users and their databases using just a few lines of code. With seamless integration for renowned databases such as PostgreSQL, MySQL, and SQLite, MindSQL also extends its capabilities to major databases like Snowflake and BigQuery by extending the core class. The library utilizes large language models (LLM) like GPT-4, Llama 2, Google Gemini, and supports knowledge bases like ChromaDB and Faiss.



## 🚀 Installation

To install MindSQL, you can use pip:

```commandline
pip install mindsql
```

MindSQL requires Python 3.10 or higher.

## 💡 Usage
```python
# !pip install mindsql


# Choose the Vector Store. LLM and DB You Want to Work With
class MindSqlGenAI(VECTOR_STORE, LLM, DATABASE):
    def __init__(self, config=None):
        VECTOR_STORE.__init__(self, config=config)
        LLM.__init__(self, config=config)
        DATABASE.__init__(self, config=config)


# Add Your Configurations
config = {"api_key": "YOUR-API-KEY"}

# Create a MindSQl Object and Connect to Your DB
minds = MindSqlGenAI(config=config)
connection = minds.create_connection("DATABASE_CONNECTION_URL")

# Index Your DB Schemas in mindsql
ddls = minds.get_all_ddls(connection=conn, database="YOUR_DATABASE_NAME")

for ind in ddls.index:
    minds.index_ddl(ddls["DDL"][ind])

# Provide Example Question-SQL Pairs Previously Used by You
minds.index(bulk=True, path="your-qsn-sql-example.json")

# Chat With Your Database!
res = minds.ask_db(question="YOUR_QUESTION", connection=connection)

# Close The Connection to Your DB
connection.close()
```
## 📁 Code Structure 

- **_utils:** Utility modules containing constants and a logger.
- **_helper:** The helper module.
- **core:** The main core module, `minds_core.py`.
- **databases:** Database-related modules.
- **llms:** Modules related to Language Models.
- **testing:** Testing scripts.
- **vectorstores:** Modules related to vector stores.
- **poetry.lock** and **pyproject.toml:** Poetry dependencies and configuration files.
- **tests:** Testcases.

## 🤝 Contributing Guidelines 

Thank you for considering contributing to our project! Please follow these guidelines for smooth collaboration:

1. Fork the repository and create your branch from master.
2. Ensure your code adheres to our coding standards and conventions.
3. Test your changes thoroughly and add a test case in the `tests` folder.
4. Submit a pull request with a clear description of the problem and solution.

## 🐛 Bug Reports

If you encounter a bug while using MindSQL, help us resolve it by following these steps:

1. Check existing issues to see if the bug has been reported.
2. If not, open a new issue with a detailed description, including steps to reproduce and relevant screenshots or error messages.

##  🚀 Feature Requests

We welcome suggestions for new features or improvements to MindSQL. Here's how you can request a new feature:

1. Check existing feature requests to avoid duplication.
2. If your feature request is unique, open a new issue and describe the feature you would like to see.
3. Provide as much context and detail as possible to help us understand your request.

## 📣 Feedback

We value your feedback and strive to improve MindSQL. Here's how you can share your thoughts with us:

- Open an issue to provide general feedback, suggestions, or comments.
- Be constructive and specific in your feedback to help us understand your perspective better.

Thank you for your interest in contributing to our project! We appreciate your support and look forward to working with you. 🚀


## 🌟 Contributors

| GitHub Profile      | Link + Image                                                                                    | Name            |
|---------------------|-------------------------------------------------------------------------------------------------|-----------------|
| siddhant-mi         | [![](https://github.com/siddhant-mi.png?size=50)](https://github.com/Aravinda93)                | Siddhant Pandey |
| ishika-mi           | [![](https://github.com/ishika-mi.png?size=50)](https://github.com/ishika-mi)                   | Ishika Shah     |
| Hasmukhsuthar05     | [![](https://github.com/Hasmukhsuthar05.png?size=50)](https://github.com/Hasmukhsuthar05)       | Hasmukh Suthar  |
| 	krishna-thakkar-mi | [![](https://github.com/krishna-thakkar-mi.png?size=50)](https://github.com/krishna-thakkar-mi) | Krishna Thakkar |
| UjjawalKRoy         | [![](https://github.com/UjjawalKRoy.png?size=50)](https://github.com/UjjawalKRoy)               | UjjawalKRoy     |