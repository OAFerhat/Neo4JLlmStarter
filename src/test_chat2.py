import os
import sys
from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser  



def test_chat1():
 
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'
    load_dotenv(dotenv_path=env_path)
    api_key = os.getenv("OPENAI_API_KEY")

    chat_llm = ChatOpenAI(openai_api_key=api_key)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an sustainability expert, and your task is to have a conversation with the user about sustainability and give them responses expected in a professional language. Be kind and adapt to the level of the user."),
        ("human", "{question}"),
        ("system", "{context}"),
    ])

    chat_chain = prompt | chat_llm | StrOutputParser()

    current_situation = """ {
        "overshooting": [
            {"name": "Carbon Dioxide", "status": "exceeded"},
            {"name": "Nitrogen Dioxide", "status": "exceeded"},
            {"name": "Ozone", "status": "exceeded"},
            {"name": "ocean acidification", "status": "at the limit"}
        ]
    }"""

    response = chat_chain.invoke(
        {
            "context": current_situation, 
            "question": "what are the 9 planetary boundaries?"
        }
    )  

    print(response)

if __name__ == "__main__":
    test_chat1()
