from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
from langsmith import traceable, Client
import os

load_dotenv()

if os.getenv("LANGSMITH_API_KEY"):
    os.environ.setdefault(
        "LANGSMITH_TRACING",
        "true"
    )

    os.environ.setdefault(
        "LANGSMITH_PROJECT",
        "Smart Q&A Bot Project"
    )

    print("LangSmith is configured.")
else:
    print("LangSmith API key not found.")


class QAResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question")
    confidence: str = Field(description="Confidence level: high, medium or low")
    reasoning:str = Field(description="The reasoning behind the answer provided")
    follow_up_questions : List[str] = Field(description="A list of follow-up questions to the topic.", default_factory=list)
    sources_needed: bool = Field(description="Indicates whether sources are needed for the answer.", default=False)


class SmartQABot:
    def __init__(
      self,
      model_name : str = "gpt-4o-mini",
      temperature:float = 0.3  
    ):
        self.model = ChatOpenAI(model=model_name,temperature=temperature).with_structured_output(QAResponse)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a knowledgeable Q&A assistant. 
                    Your guidelines:
                    - Answer questions accurately and concisely.
                    - Be honest about uncertainty -  set confidence to low if unsure
                    - Provide clear reasoning for your answers
                    - Suggest relevant follow-up questions
                    Always respond with accurate, helpful information
                    """,
                ),
                ("human", "{question}"),
            ]
        )

        self.chain = self.prompt | self.model

    @traceable(name="ask_question", run_type="chain")
    def ask(self, question: str) -> QAResponse:
        try:
            response = self.chain.invoke({"question": question})
            return response
        except Exception as e:
            print(f"Error during bot response:{e}")
            raise

    @traceable(name="ask_batch", run_type="chain")
    def ask_batch(self, questions: List[str]) -> List[QAResponse]:
        """ask multiple questions in parallel"""
        inputs = [{"question":q} for q in questions]
        return self.chain.batch(inputs)


def demo_qa_bot():
    bot = SmartQABot()

    questions = [
        "What is the capital of France",
        "Explain the theory of relativity",
        "How does photosynthesis work?"
    ]

    for question in questions:
        response = bot.ask(question)
        print(response)


if __name__ == "__main__":
    try:
        demo_qa_bot()
    finally:
        Client().flush()









