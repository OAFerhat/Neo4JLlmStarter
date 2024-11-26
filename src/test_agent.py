import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import StrOutputParser
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from langchain_community.graphs import Neo4jGraph
from uuid import uuid4
from langchain_community.tools import YouTubeSearchTool


def test_env():
    # Get the absolute path to the .env file
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'
    
    # Force reload the .env file
    load_dotenv(dotenv_path=env_path, override=True)
    
    # Get environment variables
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    api_key = os.getenv("OPENAI_API_KEY")

    SESSION_ID = str(uuid4())
    print(f"Session ID: {SESSION_ID}")

    llm = ChatOpenAI(openai_api_key=api_key)

    graph = Neo4jGraph(
        url=neo4j_uri,
        username=neo4j_username,
        password=neo4j_password
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a movie expert. You find movies from a genre or plot.",
            ),
            ("human", "{input}"),
        ]
    )

    movie_chat = prompt | llm | StrOutputParser()

    def get_memory(session_id):
        return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

    youtube = YouTubeSearchTool()

    def call_trailer_search(input):
        input = input.replace(",", " ")
        return youtube.run(input)


    tools = [
        Tool.from_function(
            name="Movie Chat",
            description="For when you need to chat about movies. The question will be a string. Return a string.",
            func=movie_chat.invoke,
        ),
        Tool.from_function(
        name="Movie Trailer Search",
        description="Use when needing to find a movie trailer. The question will include the word trailer. Return a link to a YouTube video.",
        func=call_trailer_search,
        ),
    ]

    agent_prompt = hub.pull("hwchase17/react-chat")
    agent = create_react_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)

    chat_agent = RunnableWithMessageHistory(
        agent_executor,
        get_memory,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    while True:
        q = input("> ")

        response = chat_agent.invoke(
            {
                "input": q
            },
            {"configurable": {"session_id": SESSION_ID}},
        )
        
        print(response["output"])


if __name__ == "__main__":
    test_env()
