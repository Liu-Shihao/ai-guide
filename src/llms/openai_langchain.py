from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
import os

os.environ["OPENAI_API_KEY"] = 'OPENAI_API_KEY'

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

llm = OpenAI()

# llm = OpenAI(openai_api_key="YOUR_API_KEY", openai_organization="YOUR_ORGANIZATION_ID")

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

llm_chain.run(question)