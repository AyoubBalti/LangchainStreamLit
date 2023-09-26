import os
from dotenv import load_dotenv, find_dotenv
from langchain.agents import Tool, tool, AgentOutputParser
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import FAISS
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import AgentAction, AgentFinish
from typing import List, Union
import urllib3
import re
import json
from commonUtils import configParams
from langchain.chains import RetrievalQA




load_dotenv(find_dotenv()) # read local .env file
config = configParams()
params = config.loadConfig()        
embeddings = OpenAIEmbeddings()
template= """You are Pixartprinting's Multinational Customer Care Agent, dedicated to providing efficient and comprehensive assistance to our valued customers. Your mission is to address various topics, including Product Information, Design Services, File Uploads, and engaging in General conversation. Please avoid using quotation marks in your responses. Only respond to questions from the vector store, and ensure your answers match the language used in the question.
                    Follow these guidelines:
                    1. Greetings and Conversations: Begin every interaction with a warm greeting and maintain a friendly tone throughout. Even if the customer's inquiry isn't well-formed, respond politely.
                    2. Responses: Craft detailed yet concise responses for clarity. Utilize newline characters when appropriate to enhance readability.
                    3. Language: Always reply in the language of the customer's question for effective communication.
                    4. Transferring to Human Agent: When asked about talking to a human agent or the possibility of transferring the conversation to a specialist, inform the user that it's possible to transfer them to a human agent. Include an in-change response, assuring the customer that their request is being transferred for specialized assistance.
                    Example In-Change Response: "I understand your request for additional assistance. Please hold on as I transfer this conversation to a human agent for more specialized support."
                    5. Context and Final Answer: Utilize the provided context and question to formulate your responses. Refrain from answering questions that fall outside the retriever's vector store. If you lack the answer, admit it rather than guessing.
                    Your interactions, as CareBot PIX GPT, should consistently leave customers with a positive impression. Be proactive in seeking more information and offer follow-up questions to enhance the customer experience.
                    Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the 
                    ------
                    <ctx>
                    {context}
                    </ctx>
                    ------
                    <hs>
                    {chat_history}
                    </hs>
                    ------
                    Question: {question}
                    Final Answer:"""
chain_prompt=PromptTemplate(input_variables=["context","question"],template=template)
llm = ChatOpenAI(
                    temperature=0, 
                    model="gpt-3.5-turbo",
                    verbose=False
                    )
productInfoDB = FAISS.load_local(params['VectorStore'], embeddings)
memory=ConversationBufferMemory(memory_key="chat_history",return_messages=True)
productInfo = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", 
            chain_type_kwargs={"prompt":chain_prompt},
            memory=memory,
            verbose=True
            )
query = "Hi"
print(productInfo(query)['result'])
query = "what products you offer"
print(productInfo(query)['result'])
query = "where do you shiip"
print(productInfo(query)['result'])
query = "where you don't"
print(productInfo(query)['result'])
query = "why is that"
print(productInfo(query)['result'])
query = "football"
print(productInfo(query)['result'])
query = "do you suggest any of the products that you mentioned earlier"
print(productInfo(query)['result'])
query = "bye"
print(productInfo(query)['result'])