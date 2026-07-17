from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

from langgraph.prebuilt import ToolNode, tools_condition
from ddgs import DDGS
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
import requests
import random
import os
import sqlite3

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    timeout=30,
    max_retries=2,
)

##Tools

@tool
def search_tool(query: str) -> str:
    """Search the web using DuckDuckGo"""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results = 5))
    return str(results) if results else "No results found."
@tool
def calculator(first_num : float, second_num : float, operation :str)->dict:
    """
    Perform a basic arithematic operation on two numbers.
    Supported operations : add, sub, mul, div
    """
    try :
        if operation =="add":
            result = first_num + second_num
        elif operation =="sub":
            result = first_num - second_num
        elif operation =="mul":
            result = first_num * second_num
        elif operation =="div":
            if second_num == 0:
                return {'error':"Division by zero is not allowed"}
            result = first_num / second_num
        else :
            return {"error": f"Unsupported operation: '{operation}'"}
        return {'first_num' : first_num, 'second_num' : second_num, 'operation':operation, 'result': result}
    except Exception as e:
        return {'error': str(e)}

@tool
def get_stock_price(symbol : str)-> dict:
    """
    Fetch the latest stock price for a given symbol (eg. 'AAPL', 'TSLA')
    using alpha vantage with the API hey in the url.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=JIBTVBIBFU1587L6"
    r = requests.get(url)
    return r.json()

tools = [search_tool, get_stock_price,calculator]

llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]

def Chat_node(state: ChatState) ->  ChatState:
    """LLM node that may answer or request a tool call"""
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    return {'messages' : [response]}

tool_node = ToolNode(tools)

### Creating the database and making a connection
conn = sqlite3.connect(database='chatbot.db',check_same_thread=False)
checkpointer = SqliteSaver(conn = conn)


graph = StateGraph(ChatState)
graph.add_node('chat_node', Chat_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, "chat_node")
# If the LLM asked for a tool, go to ToolNode else finish
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools','chat_node')

chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']["thread_id"])
    return list(all_threads)