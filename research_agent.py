from langchain.agents import initialize_agent, Tool, AgentType
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
import os

def create_research_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    search = TavilySearchResults()
    tools = [
        Tool(
            name="TavilySearch",
            func=search.run,
            description="Search the web for real-time information to augment video transcription context."
        )
    ]
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    return agent
