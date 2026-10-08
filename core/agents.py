from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

try:
    from core.tools import web_search, scrape_url
except ImportError:
    from tools import web_search, scrape_url

# ReAct prompt template
react_prompt_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


def build_search_agent(llm):
    prompt = PromptTemplate.from_template(react_prompt_template)
    agent = create_react_agent(llm, [web_search], prompt)
    return AgentExecutor(
        agent=agent,
        tools=[web_search],
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,
    )


def build_reader_agent(llm):
    prompt = PromptTemplate.from_template(react_prompt_template)
    agent = create_react_agent(llm, [scrape_url], prompt)
    return AgentExecutor(
        agent=agent,
        tools=[scrape_url],
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,
    )


def build_writer_chain(llm):
    prompt = PromptTemplate.from_template(
        "You are an expert Writer. Based on the following research facts, "
        "write a comprehensive professional summary report.\n\n"
        "Research Facts:\n{facts}\n\nReport:"
    )
    return prompt | llm | StrOutputParser()
