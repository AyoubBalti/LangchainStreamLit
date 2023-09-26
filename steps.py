from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.agents import Tool, AgentOutputParser
from langchain.schema import AgentAction, AgentFinish
from commonUtils import configParams
import json
from typing import List, Union
import re

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

# Custom JSON Encoder for AgentAction objects
class AgentActionEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, AgentAction):
            return {
                "tool": obj.tool,
                "tool_input": obj.tool_input,
                "log": obj.log
            }
        return super().default(obj)

# Load environment variables
load_dotenv(find_dotenv())

# Initialize LLM
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings()

# Load configuration
config = configParams()
params = config.loadConfig()

# Load FAISS vector store
productInfoDB = FAISS.load_local(params['VectorStore'], embeddings)

# Initialize RetrievalQA
productInfo = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=productInfoDB.as_retriever()
)

# Define tools
tools = [
    Tool(
        name="Information QA System",
        func=productInfo.run,
        description="useful to provide the most accurate and detailed answers to fully formed questions about Pixartprinting's products, services and general information.",
    ),
]
custom_output_parser = CustomOutputParser()

# Initialize agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    output_parser=custom_output_parser,
    verbose=True,
    return_intermediate_steps=True,
)

# Run agent and get response
response = agent(
    {
        "input": "What's a file and what are the available services?"
    }
)

# Print intermediate steps with custom JSON encoding
intermediate_steps = response["intermediate_steps"]
serialized_steps = []
for step in intermediate_steps:
    serialized_step = (json.dumps(step[0], cls=AgentActionEncoder), step[1])
    serialized_steps.append(serialized_step)

print(json.dumps(serialized_steps, indent=2))
