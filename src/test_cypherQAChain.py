import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

from langchain_community.graphs import Neo4jGraph
from uuid import uuid4
from langchain_openai import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate


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

    # Get the schema from Neo4j
    schema = graph.schema
    print("Database Schema:")
    print(schema)

    llm = ChatOpenAI(
        openai_api_key=api_key
    )

    CYPHER_GENERATION_TEMPLATE = """
    You are an expert Neo4j Developer translating user questions into Cypher to answer questions about movies and provide recommendations.
    Convert the user's question based on the schema.
    Instructions:
    Use only the provided relationship types and properties in the schema.
    Do not use any other relationship types or properties that are not provided.
    For movie titles that begin with "The", move "the" to the end, For example "The 39 Steps" becomes "39 Steps, The" or "The Matrix" becomes "Matrix, The".

    If no data is returned, do not attempt to answer the question.
    Only respond to questions that require you to construct a Cypher statement.
    Do not include any explanations or apologies in your responses.

    Examples:
    Find movies and genres:
    MATCH (m:Movie)-[:IN_GENRE]->(g)
    RETURN m.title, g.name

    Schema: {schema}
    Question: {question}
    """

    cypher_generation_prompt = PromptTemplate(
        template=CYPHER_GENERATION_TEMPLATE,
        input_variables=["schema", "question"],
    )

    cypher_chain = GraphCypherQAChain.from_llm(
        llm,
        graph=graph,
        cypher_prompt=cypher_generation_prompt,
        verbose=True,
        allow_dangerous_requests=True  # Acknowledging that we understand the risks
    )

    while True:
        question = input("> ")
        if question.lower() in ['exit', 'quit']:
            break
        result = cypher_chain.invoke({
            "query": question,
            "schema": schema
        })
        print("\nAnswer:", result['result'])


if __name__ == "__main__":
    test_env()
