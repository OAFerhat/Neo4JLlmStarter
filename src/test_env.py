import os
from dotenv import load_dotenv
import sys
from pathlib import Path
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

def test_env_loading():
    """
    Test the loading of environment variables from a .env file located at the 
    project root directory. The function retrieves the OpenAI API key and 
    invokes a model to get a response for a specific query.

    Steps:
    - Determine the project root directory.
    - Print the current working directory and project root for debugging.
    - Check for the existence of the .env file in the project root.
    - Load the environment variables from the .env file.
    - Retrieve the 'OPENAI_API_KEY' from the environment variables.
    - If found, use the key to invoke an OpenAI model.
    - Print the API key prefix and the response from the model.
    - If the API key is not found, print an error message.
    """
    project_root = Path(__file__).parent.parent
    print(f"Current working directory: {os.getcwd()}")
    print(f"Project root directory: {project_root}")
    
    env_path = project_root / '.env'
    print(f"Looking for .env file at: {env_path}")
    
    if not env_path.exists():
        print(f"Error: .env file not found at {env_path}")
        sys.exit(1)
        
    load_dotenv(dotenv_path=env_path)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("API Key found!")
        print(f"API Key starts with: {api_key[:8]}...")
        llm = OpenAI(openai_api_key=api_key, model_name="gpt-4o", temperature=0)
        
        template = PromptTemplate(template="""
        You are an sustainability expert. Explain the concept of {concept}
        """, input_variables=["concept"])

        response = llm.invoke(template.format(concept="9 planetary boundaries"))

        print(response)
    else:
        print("Error: OPENAI_API_KEY not found in environment variables")

if __name__ == "__main__":
    test_env_loading()



