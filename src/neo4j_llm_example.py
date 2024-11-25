import os
from dotenv import load_dotenv
from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.llms import OpenAI

# Load environment variables
load_dotenv()

def connect_to_neo4j():
    """Establish connection to Neo4j database."""
    return Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD")
    )

def setup_qa_chain(graph):
    """Set up the question-answering chain with Neo4j and OpenAI."""
    llm = OpenAI(temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    return GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True
    )

def main():
    # Connect to Neo4j
    try:
        graph = connect_to_neo4j()
        print("Successfully connected to Neo4j!")
        
        # Create QA chain
        chain = setup_qa_chain(graph)
        print("Successfully set up LangChain integration!")
        
        # Example query
        question = "What are the nodes and relationships in the database?"
        response = chain.run(question)
        print(f"\nQuestion: {question}")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
