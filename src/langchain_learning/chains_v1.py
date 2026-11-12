"""
Understanding chains in Langchain V.1 
LCEL patterns, compositions and debugging
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()

model = init_chat_model(model="gpt-4o-mini", temperature=0)

def demo_basic_chain():
    prompt = ChatPromptTemplate.from_template("Summarize the following text in one sentence: {text}")
    parser = StrOutputParser()
    chain = prompt | model | parser

    result = chain.invoke({"text":"Langchain is a framework for developing applications powered by language models"})

    print(f"Summary: {result}")


def demo_parallel_chain():
    summarize_prompt = ChatPromptTemplate.from_template("Summarize in two sentences: {text}")
    keywords_prompt = ChatPromptTemplate.from_template("Extract 5 keywords in the following text: {text}\n Return as a comma separated list")
    sentiment_prompt = ChatPromptTemplate.from_template("What is the sentiment of the following text? {text}")

    parser = StrOutputParser()

    analysis_chain = RunnableParallel(
        summary = summarize_prompt | model | parser,
        keywords = keywords_prompt | model | parser,
        sentiment = sentiment_prompt | model | parser
    )

    text = """
            The new AI features are absolutely incredible
            Technology is evolving faster than ever. AI is transforming the way we work and learn. 
            What once seemed impossible is now becoming reality. 
            The possibilities are truly limitless. Excited to see what the future holds
          """

    result = analysis_chain.invoke({"text":text})
    print(result)

def demo_passthrough_chain():
    prompt = ChatPromptTemplate.from_template(
        "Original question:{question} \n"
        "Context: {context} \n \n"
        "Answer the question based on the context."
    )

    def fake_retriver(input_dict):
        return "Langchain was created by Harrison Chase in 2022"

    chain =(RunnableParallel(
        context = RunnableLambda(fake_retriver),
        question = RunnablePassthrough(),
    ) | prompt | model) | StrOutputParser()

    result = chain.invoke({"question:": "Who created and lanchain and future of langchain"})
    print(result);


if __name__== "__main__":
    demo_passthrough_chain()
    

