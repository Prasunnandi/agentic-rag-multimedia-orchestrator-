from langchain.prompts import PromptTemplate

class Extractor:
    def __init__(self, llm):
        self.llm = llm

    def extract_action_items(self, transcript):
        prompt = PromptTemplate.from_template("Extract a bulleted list of action items from the following transcript:\n{transcript}")
        chain = prompt | self.llm
        return chain.invoke({"transcript": transcript})

    def extract_decisions(self, transcript):
        prompt = PromptTemplate.from_template("Extract a bulleted list of key decisions made in the following transcript:\n{transcript}")
        chain = prompt | self.llm
        return chain.invoke({"transcript": transcript})
