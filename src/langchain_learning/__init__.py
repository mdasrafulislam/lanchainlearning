from dotenv import load_dotenv
load_dotenv()

from importlib.metadata import version
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
openai_version = version("langchain-openai")

print(f"langchain-openai version: {openai_version}")

def main() -> None:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke("Hello world")
    print(response)


if __name__ == "__main__":
    main()
