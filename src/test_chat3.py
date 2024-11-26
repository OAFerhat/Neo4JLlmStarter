import os
import sys
from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser  
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory



def test_chat():
 
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'
    load_dotenv(dotenv_path=env_path)
    api_key = os.getenv("OPENAI_API_KEY")

    chat_llm = ChatOpenAI(openai_api_key=api_key)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an sustainability expert, and your task is to have a conversation with the user about sustainability and give them responses expected in a professional language. Be kind and adapt to the level of the user."),
        ("system", "{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    memory = ChatMessageHistory()
    def get_memory(session_id):
        return memory

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
            {"name": "Nitrogen Dioxide", "status": "exceeded"},
            {"name": "Ozone", "status": "exceeded"},
            {"name": "ocean acidification", "status": "at the limit"}
        ]
    }"""

    while True:
        question = input("You: ")
        if question == "exit":
            break
        response = chat_with_message_history.invoke(
            {
                "context": current_situation, 
                "question": question
            },
            config={
                "configurable": {"session_id": "none"}
            }
        )  

        print(response)


if __name__ == "__main__":
    test_chat()
