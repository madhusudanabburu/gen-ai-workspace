import streamlit as st
import os
import chromadb
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolExecutor
from constants import CHROMA_SETTINGS
import operator
import traceback
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolInvocation
from langgraph.graph import END, StateGraph, START

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "" # Insert the API Key for LangSmith

model = os.environ.get("MODEL", "local_Meta_Llama_LLM_8B")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "nomic-embed-text")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

### Setup the Tools - Build retriever tool ###
client = chromadb.PersistentClient(path=persist_directory, settings=CHROMA_SETTINGS)   
embeddings = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
db = Chroma(client=client, collection_name="website_docs", persist_directory=persist_directory, collection_metadata={"hnsw:space": "cosine"}, embedding_function=embeddings)
retriever = db.as_retriever(search_type="similarity_score_threshold",
        search_kwargs={
            "k": 20,
            "score_threshold": 0.5,
        })

tool = create_retriever_tool(
    retriever,
    "website_docs",
    "Searches and returns excerpts from the Autonomous Agents blog post.",
)

tools = [tool]
tool_executor = ToolExecutor(tools)

### Setup the Model ###
llm = OllamaFunctions(base_url="http://localhost:11434", model="local_Meta_Llama_LLM_8B", format="json")
llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if "tool_calls" not in str(last_message):
        return "end"
    # Otherwise if there is, we need to check what type of function call it is
    if "ToolMessage" in str(messages):
        return "end"
    # Otherwise we continue
    return "continue"

# Define the function that calls the model
def call_model(state):
    messages = state["messages"]
    response = llm.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the function to execute tools
def call_tool(state):
    messages = state["messages"]
    # We know the last message involves at least one tool call
    last_message = messages[-1]

    # We construct an ToolInvocation from the function_call
    tool_call = last_message.tool_calls[0]
    tool_name = tool_call["name"]
    arguments = tool_call["args"]
    action = ToolInvocation(
        tool=tool_name,
        tool_input=arguments,
    )
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)
    parsed_response1 = response.replace('\n', ' ').replace('\r', '')
    parsed_response2 = str(parsed_response1).replace('[','').replace(']','').replace('\'','').replace('\"','')
    parsed_response3 = str(parsed_response2).replace('{','').replace('}','')
    parsed_response = str(parsed_response3).replace('(','').replace(')','')
    # We use the response to create a ToolMessage
    tool_message = ToolMessage(
        content=str(parsed_response), name=action.tool, tool_call_id=tool_call["id"]
    )
    # We return a list, because this will get added to the existing list
    return {"messages": [tool_message]}

def create_agent():

    # Define a new graph
    # Initialize a new graph
    graph = StateGraph(AgentState)

    # Define the two Nodes we will cycle between
    graph.add_node("agent", call_model)
    graph.add_node("action", call_tool)

    # Set the Starting Edge
    graph.set_entry_point("agent")

    # Set our Contitional Edges
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "end": END,
        },
    )

    # Set the Normal Edges
    graph.add_edge("action", "agent")

    # Compile the workflow
    agent = graph.compile()

    return agent

def main():

    # Create the agent object
    agent = create_agent()

    # Create a side bar with information of the application - if necessary, we can collect information from the user here 
    with st.sidebar:
        st.title("📝 Conversational RAG with Website")

    st.subheader("Chatbot interaction with Local Llama3", divider="red", anchor=False)

    # Create a container to host the chat window
    message_container = st.container(height=600, border=True)

    # Declare a variable for storing the messages from the user and the agent in the session
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Change the icon for the messages depending on the role - user/model
    for message in st.session_state.messages:
        avatar = "🤖" if message["role"] == "assistant" else "😎"
        with message_container.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # This is where the chat becomes interactive 
    # The messages are appended to the session state and the processing information is displayed - 'model working'
    # For every query that the user types in, the response coming out of either the agent/tool is displayed here 
    if query := st.chat_input("Enter your query here..."):
        try:
            st.session_state.messages.append(
                {"role": "user", "content": query})

            message_container.chat_message("user", avatar="😎").markdown(query)

            with message_container.chat_message("assistant", avatar="🤖"):
                with st.spinner("model working..."):
                    inputs = {"messages": [HumanMessage(content=query)]}
                    for response in agent.stream(inputs):
                        for key, value in response.items():
                            print(f"Output from node '{key}':")
                            print("---")
                            print(value)
                            if key=='agent' or key=='action':
                                message = value["messages"][-1]
                        print("\n---\n")                        
                        # stream response
                        st.write(message.content)
                        st.session_state.messages.append({"role": "assistant", "content": message.content})

        except Exception as e:
            traceback.print_exc()
            st.error(e, icon="⛔️")

if __name__ == "__main__":
    main()
