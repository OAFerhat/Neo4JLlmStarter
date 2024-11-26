import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from langchain_community.graphs import Neo4jGraph
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

    graph = Neo4jGraph(
        url=neo4j_uri,
        username=neo4j_username,
        password=neo4j_password   
    )

    SESSION_ID = str(uuid4())
    print(f"Session ID: {SESSION_ID}")



if __name__ == "__main__":
    test_env()






