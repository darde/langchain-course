from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from typing import List
from pydantic import BaseModel, Field
# from langchain_tavily import TavilySearch

load_dotenv()

tavily = TavilyClient()

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url:str = Field(description="The url of the source")

class AgentResponse(BaseModel):
    """Schema for the agent response with answer and sources"""

    answer:str = Field(description="The agent's answer to the query")
    sources:List[Source] = Field(default_factory=list, description="The list of sources to generate the answer.")



@tool
def search(query: str) -> str:
    """
    Tool that searches over the internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """

    print(f"Searching for query: {query}")
    return tavily.search(query=query)

llm = ChatOpenAI(model="gpt-5")
tools = [search]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main() -> None:
    print("Hello from langchain-course!")
    result = agent.invoke({ "messages": HumanMessage(content="Search for three job postings for an ai engineer using langchain in USA hiring latin american engineers to work remotely.")})
    print(result)

if __name__ == "__main__":
    main()
