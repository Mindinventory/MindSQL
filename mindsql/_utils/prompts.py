DEFAULT_PROMPT: str = """As a {dialect_name} expert, your task is to generate SQL queries based on user questions. Ensure that your {dialect_name} queries are syntactically correct and tailored to the user's inquiry. Retrieve at most 10 results using the LIMIT clause and order them for relevance. Avoid querying for all columns from a table. Select only the necessary columns wrapped in backticks (`). Use CURDATE() to handle 'today' queries and employ the LIKE clause for precise matches in {dialect_name}. Carefully consider column names and their respective tables to avoid querying non-existent columns. Stop after delivering the SQLQuery, avoiding follow-up questions.

Follow this format:
Question: User's question here
SQLQuery: Your SQL query without preamble

No preamble


"""

DDL_PROMPT = """Only use the following tables:
{}

"""
FEW_SHOT_EXAMPLE = """Make use of the following Example 'SQLQuery' for generating SQL query:
{}

"""

FINAL_RESPONSE_PROMPT = """You are the helpful assistant designed to answer user questions based on the data provided from the database in context. Your goal is to analyze the user's query and provide a helpful response using only the information available in the context. If Context is None or Empty, say you don't have the data to answer the question.

###DATAFRAME CONTEXT:
{context_df}

###USER QUESTION:
{user_query}

###ASSISTANT RESPONSE:
"""

PLOTLY_PROMPT = """You are a proficient Python developer with expertise in the Plotly library. Your objective is to generate Python code to create a BEAUTIFUL chart based on the query using the 
    provided Pandas dataframe. You can create any chart you want.

    ### QUERY: 
    {query}

    ### DATAFRAME:
    {df}

    ### INSTRUCTIONS: 1. Create a function called 'get_chart'. 2. Begin by importing the necessary libraries (Pandas, 
    Plotly, and Decimal if needed). 3. Utilize the 'plotly.graph_objects' library if the provided dataframe has more 
    than 2 columns to showcase multi bar plots. Otherwise, utilize the 'plotly.express' library. 4. Generate a chart 
    using the provided dataframe and the Plotly library. 5. Accurately interpret the x-axis title, y-axis title, 
    and chart title as per the user's query and the dataframe. 6. Utilize the 'update_layout' method to include the 
    x-axis title, y-axis title, chart title, plot background color, and paper background color, setting both of them 
    to blue (HEX code: #0e243b). 7. Set the font color to white (HEX code: #f7f9fa) using the 'update_layout' method. 
    Execute the created function with the argument as the provided dataframe, at the outer indent at the end and 
    store the result in a variable called 'chart'.

    ### CODE CRITERIA
    - Optimize the code for efficiency and clarity.
    - Avoid using incorrect syntax.
    - Ensure that the code is well-commented for readability and syntactically correct.
    """

SQL_EXCEPTION_RESPONSE = """Apologies for the inconvenience! 🙏 It seems the database is currently experiencing a bit 
of a hiccup and isn't cooperating as we'd like. 🤖"""

