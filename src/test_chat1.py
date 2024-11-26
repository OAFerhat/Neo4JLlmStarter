import os
import sys
from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage  



def test_chat1():
 
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'
    load_dotenv(dotenv_path=env_path)
    api_key = os.getenv("OPENAI_API_KEY")


    chat_llm = ChatOpenAI(openai_api_key=api_key)

    instructions = SystemMessage(content="You are an sustainability expert, and your task is to have a conversation with the user about sustainability and give them responses expected in a professional language. Be kind and adapt to the level of the user.")
    question = HumanMessage(content="what are the 9 planetary boundaries?")
    

    response = chat_llm.invoke([instructions, question])
    

    print(response.content)

if __name__ == "__main__":
    test_chat1()
