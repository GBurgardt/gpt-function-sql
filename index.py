import sqlite3
import arxiv
import ast
import concurrent
from csv import writer
from IPython.display import display, Markdown, Latex
import json
import openai
import os
import pandas as pd
from PyPDF2 import PdfReader
import requests
from scipy import spatial
from tenacity import retry, wait_random_exponential, stop_after_attempt
import tiktoken
from tqdm import tqdm
from termcolor import colored

import psycopg2

from sqlalchemy import create_engine
from sqlalchemy import inspect
from sqlalchemy import text

import argparse



GPT_MODEL = "gpt-3.5-turbo-0613"
EMBEDDING_MODEL = "text-embedding-ada-002"

def connect_to_postgres(database_url):
    """Connect to PostgreSQL database using provided database URL."""
    try:
        engine = create_engine(database_url)
        conn = engine.connect()
        # print("Connected to PostgreSQL database successfully")
        return conn
    except Exception as e:
        print(f"Unable_to_connect_to_PostgreSQL_database: {e}")
        raise e

def get_table_names(conn):
    """Return a list of table names for PostgreSQL."""
    inspector = inspect(conn)
    return inspector.get_table_names()

def get_column_names(conn, table_name):
    """Return a list of column names for PostgreSQL."""
    inspector = inspect(conn)
    return [column['name'] for column in inspector.get_columns(table_name)]

conn = connect_to_postgres('postgresql://postgres:postgres@localhost:5432')
# print("Opened database successfully")



def get_database_info(conn):
    """Return a list of dicts containing the table name and columns for each table in the database."""
    table_dicts = []
    for table_name in get_table_names(conn):
        columns_names = get_column_names(conn, table_name)
        table_dicts.append({"table_name": table_name, "column_names": columns_names})
    return table_dicts


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, functions=None, model=GPT_MODEL):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai.api_key,
    }
    json_data = {"model": model, "messages": messages}
    if functions is not None:
        json_data.update({"functions": functions})
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e



class Conversation:
    def __init__(self):
        self.conversation_history = []

    def add_message(self, role, content):
        message = {"role": role, "content": content}
        self.conversation_history.append(message)

    def display_conversation(self, detailed=False):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        for message in self.conversation_history:
            print(
                colored(
                    f"{message['role']}: {message['content']}\n\n",
                    role_to_color[message["role"]],
                )
            )
database_schema_dict = get_database_info(conn)
database_schema_string = "\n".join(
    [
        f"Table: {table['table_name']}\nColumns: {', '.join(table['column_names'])}"
        for table in database_schema_dict
    ]
)

functions = [
    {
        "name": "ask_database",
        "description": "Use this function to answer user questions about database, with google ads, keywords, etc. Output should be a fully formed SQL query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": f"""
                            SQL query extracting info to answer the user's question.
                            SQL should be written using this database schema:
                            {database_schema_string}
                            The query should be returned in plain text, not in JSON.
                            """,
                }
            },
            "required": ["query"],
        },
    }
]

def ask_database(conn, query):
    """Function to query PostgreSQL database with provided SQL query."""
    try:
        stmt = text(query)  # Convert string SQL query into SQLAlchemy text object
        # print(f"TESTTTTTTTTTTTTTTTTT: {stmt}")
        result = conn.execute(stmt)  # Execute text object
        results = [row for row in result]
        return results
    except Exception as e:
        raise Exception(f"SQL error: {e}")



def chat_completion_with_function_execution(messages, functions=None):
    """This function makes a ChatCompletion API call and if a function call is requested, executes the function"""
    try:
        response = chat_completion_request(messages, functions)
        # print(f"ChatCompletion response: {response.json()}")
        full_message = response.json()["choices"][0]
        if full_message["finish_reason"] == "function_call":
            print(f"Function generation requested, calling function")
            return call_function(messages, full_message)
        else:
            print(f"Function not required, responding to user")
            return response.json()
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return response


def call_function(messages, full_message):
    """Executes function calls using model generated function arguments."""

    # We'll add our one function here - this can be extended with any additional functions
    if full_message["message"]["function_call"]["name"] == "ask_database":
        query = eval(full_message["message"]["function_call"]["arguments"])
        print(f"Prepped query is {query}")
        try:
            # results = ask_database(conn, 'SELECT * FROM "Project"')
            results = ask_database(conn, query["query"])
        except Exception as e:
            print(e)

            # This following block tries to fix any issues in query generation with a subsequent call
            messages.append(
                {
                    "role": "system",
                    "content": f"""Query: {query['query']}
The previous query received the error {e}. 
Please return a fixed SQL query in plain text.
Your response should consist of ONLY the SQL query with the separator sql_start at the beginning and sql_end at the end""",
                }
            )
            response = chat_completion_request(messages, model="gpt-4-0613")

            # Retrying with the fixed SQL query. If it fails a second time we exit.
            try:
                cleaned_query = response.json()["choices"][0]["message"][
                    "content"
                ].split("sql_start")[1]
                cleaned_query = cleaned_query.split("sql_end")[0]
                print(cleaned_query)
                results = ask_database(conn, cleaned_query)
                print(results)
                print("Got on second try")

            except Exception as e:
                print("Second failure, exiting")

                print(f"Function execution failed")
                print(f"Error message: {e}")

        messages.append(
            {"role": "function", "name": "ask_database", "content": str(results)}
        )

        try:
            response = chat_completion_request(messages)
            return response.json()
        except Exception as e:
            print(type(e))
            print(e)
            raise Exception("Function chat request failed")
    else:
        raise Exception("Function does not exist and cannot be called")

agent_system_message = """You are GetGlobyGPT, a helpful assistant who gets answers to user questions from the GetGloby database.
Provide as many details as possible to your users
Begin!"""

parser = argparse.ArgumentParser(description="Enter the user message.")
parser.add_argument('user_message', type=str, help="User's message for the assistant.")
args = parser.parse_args()
user_message = args.user_message


sql_conversation = Conversation()
sql_conversation.add_message("system", agent_system_message)
sql_conversation.add_message("user", user_message)

# sql_conversation.add_message(
#     "user", "Hi. actualiza el project con id 53. edita su nombre a algo como HOLALOCOO. luego muestramelo"
# )

chat_response = chat_completion_with_function_execution(
    sql_conversation.conversation_history, functions=functions
)
try:
    assistant_message = chat_response["choices"][0]["message"]["content"]
    print(assistant_message)
except Exception as e:
    print(e)
    print(chat_response)


sql_conversation.add_message("assistant", assistant_message)
sql_conversation.display_conversation(detailed=True)
