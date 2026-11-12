
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

def exercise_json_output_parser():
    parser = JsonOutputParser()
    llm = init_chat_model(model="gpt-4o-mini", temperature=0.7)
    prompt = ChatPromptTemplate.from_template("Return a JSON object with name and age for: {description}")

    chain = prompt | llm | parser

    response = chain.invoke({"description" : "a person name alex aged 45 is a javascript developer"})
    print(response)



def exercise_chat_prompt():
    prompt = ChatPromptTemplate.from_template("Tell me a {adjective} joke about {topic}")
    messages = prompt.format_messages(adjective="funny", topic="cow")
    print(messages)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    response = model.invoke(messages)
    print(response.content)

    multi = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a translator, be concise"),
            ("human", "Translate {text} to {language}")
        ]
    )

    messages = multi.format_messages(text="Good Morning", language="Japanese")
    print("\n Multi-message template:")

    for msg in messages:
        print(f"{type(msg).__name__}: {msg.content}")


    print("\n")

    response = model.invoke(messages)
    print(response.content)



def exercise_multi_model():
    def get_responses(question: str, model_names: list[str]) -> dict[str, str]:
        responses = {}    
        for model_name in model_names:
            model = init_chat_model(
                model=model_name,
                temperature = 0.7,
                streaming = False
            )
            response = model.invoke(question)
            responses[model_name] = response.content
        return responses

    results = get_responses("What is AI?", ["gpt-4o-mini", "gpt-4o"])
    for model, answer in results.items():
        print(f"Response from {model}: {answer} \n")
     
def demo_basic_chain():
    """Demonstrate a basic chain using LCEL and Runnables."""

    # Step 1: Create prompt template
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. "
        "Answer in one sentence: {question}"
    )

    # Step 2: Initialize model
    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7
    )

    # Step 3: Initialize output parser
    parser = StrOutputParser()

    # Step 4: Compose LCEL chain
    chain = prompt | model | parser

    # Step 5: Invoke chain
    result = chain.invoke({
        "question": "What is LangChain?"
    })

    print(result)

def demo_batch_execution():
    """Demonstrate batch execution for multiple inputs"""

    prompt = ChatPromptTemplate.from_template("Transate to French: {text}")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    inputs = [
        {"text":"Hello, how are you"}, 
        {"text": "What is your name"},
        {"text": "Where is the nearest restaurant"}
    ]

    results = chain.batch(inputs);
    for text in zip(inputs, results):
        print(f"Input: {text[0] ['text']} => Output: {text[1]}")

def demo_streaming():
    """Demonstrate steaming for real-time output"""
    prompt = ChatPromptTemplate.from_template("Write a haiku about: {topic}")
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    print("Streaming output: ")

    for chunk in chain.stream({"topic": "AI"}):
        print(chunk, end="", flush=True)

def exercise_first_chain():
    """
    1. Takes a product name and target audience
    2. Generate a marketing tagline
    3. Returns just the tagline as a string

    Test with product="AI Course", audience="developers"
    """
    prompt = ChatPromptTemplate.from_template(
        "Create a marketing tagline for a product named '{product}' targeting '{audience}'."
    )

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    result = chain.invoke(
        {"product": "AI Course", "audience" : "developers"}
    )

    print(result)

def demo_init_chat_model():
    chat_model = init_chat_model(
        model="gpt-4o-mini",
        model_provider="openai",
        temperature=0.7,
        streaming=True
    )
    response = chat_model.invoke("What is the capital of the France?Anwser in one word")
    print(response.content)

def demo_message():
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    messages = [
        SystemMessage(content="You are a pirate. Always answer like a pirate"),
        HumanMessage(content="What's the weather like today")        
    ]

    response = model.invoke(messages)
    print(f"Response from the pirate: {response.content}");

def demo_message_types():
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    messages = [
        SystemMessage(content="You are a math tutor, Be brief"),
        HumanMessage(content="What's is 5*5?"),
        AIMessage(content="30"),
        HumanMessage(content="And if I add 10?")
    ]

    response = model.invoke(messages)
    print(f"Conversion result:{response.content}")



if __name__ == "__main__":
    exercise_json_output_parser()