import os
import sys
from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

def test_lcel():
 
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'
    load_dotenv(dotenv_path=env_path)
    
    api_key = os.getenv("OPENAI_API_KEY")

    llm = OpenAI(openai_api_key=api_key)
    
    template = PromptTemplate.from_template(
        """You are an sustainability expert. Explain the concept of {concept} in bullet in details"""
        )

    llm_chain = template | llm

    response = llm_chain.invoke({"concept":"Ecosystem services"})
    
    print(response)

if __name__ == "__main__":
    test_lcel()
