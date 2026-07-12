from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
import os 
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3

load_dotenv()

llm = ChatOpenAI(
model="gpt-4o-mini",
api_key=os.environ["OPENROUTER_API_KEY"],
base_url="https://openrouter.ai/api/v1",
timeout=30,
max_retries=2,
)
CONFIG = {'configurable' : {'thread_id' : 'thread_1'}}

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  ## All types of messages inherit from the BaseMessage class


def chat_node(state: ChatState) -> ChatState:
    ## take user query from the state
    messages = state['messages']

    ## send to llm 
    response = llm.invoke(messages)

    ## response store in state
    return {'messages' : [response]}

### Creating the database and making a connection
conn = sqlite3.connect(database='chatbot.db',check_same_thread=False)
checkpointer = SqliteSaver(conn = conn)

graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_edge(START , 'chat_node')
graph.add_edge('chat_node', END)

workflow = graph.compile(checkpointer = checkpointer)
def retrieve_all_threads():

    all_threads = set()

    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])
    return list(all_threads)
