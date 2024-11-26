import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector


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

    embedding_provider = OpenAIEmbeddings( openai_api_key=api_key)

    movie_plot_vector = Neo4jVector.from_existing_index(
        embedding_provider,
        graph=graph,
        index_name="moviePlots",
        embedding_node_property="plotEmbedding",
        text_node_property="plot",
    )

    result = movie_plot_vector.similarity_search("A movie whith the plane of the US Presidents", k=4)
    for doc in result:
        print(doc.metadata["title"], "-", doc.page_content)


if __name__ == "__main__":
    test_env()
