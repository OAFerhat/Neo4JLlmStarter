import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.graphs import Neo4jGraph
from langchain_community.chat_message_histories import Neo4jChatMessageHistory
from uuid import uuid4



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

    chat_llm = ChatOpenAI(openai_api_key=api_key)

    graph = Neo4jGraph(
        url=neo4j_uri,
        username=neo4j_username,
        password=neo4j_password
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an sustainability expert, and your task is to have a conversation with the user about sustainability and give them responses expected in a professional language. Be kind and adapt to the level of the user."),
        ("system", "{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    def get_memory(session_id):
        return Neo4jChatMessageHistory(session_id=session_id, graph=graph)


    chat_chain = prompt | chat_llm | StrOutputParser()

    chat_with_message_history = RunnableWithMessageHistory(
        chat_chain,
        get_memory,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    current_situation = """ {
        "overshooting": [
            {"name": "Carbon Dioxide", "status": "exceeded"},
            {"name": "Biodiversity", "status": "exceeded"},
            {"name": "Ozone", "status": "getting back under the limit"},
            {"name": "ocean acidification", "status": "at the limit"}
        ]
    }"""

    while True:
        question = input("> ")

        response = chat_with_message_history.invoke(
            {
                "context": current_situation,
                "question": question,
                
            }, 
            config={
                "configurable": {"session_id": SESSION_ID}
            }
        )
        
        print(response)


if __name__ == "__main__":
    test_env()
