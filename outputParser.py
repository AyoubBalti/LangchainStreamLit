from dotenv import find_dotenv, load_dotenv
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

load_dotenv(find_dotenv())
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")


class ConversationInfo(BaseModel):
    question: str = Field(
        description="This is the inquiry posed by the customer.")
    answer: str = Field(description="This corresponds to the reply given by the chat agent.")
    contact_reason: str = Field(description="""This designates the subject of the conversation.
        Please extract only the topic name from the provided list, not its description.
        [Talk to Human: When a user requests human agent assistance or inquires about how to contact an agent;
        General QA Info: When general questions about services or the company are posed;
        Product Info: When queries pertain to product details and information;
        General Conversation: Comprising greetings, farewells, expressions of gratitude, other;]""")
    language: str = Field(
        description="[lowercase] This denotes the language utilized in the conversation.")
    locale: str = Field(
        description="This denotes the locale utilized in the conversation (e.g. en-us, en-gb, fr-fr, fr-be, fr-ch, it, es).")
    fulfillment: str = Field(
        description="This field indicates whether the bot successfully fulfilled the request or not, with options being [true, false].")
    sentiment: str = Field(
        description="These encompass the emotions and sentiments expressed by the customer, with options being [positive, negative or neutral].")
    satisfaction: int = Field(
        description="This consists of the satisfaction score, an integer score ranging from 0 to 10.")
    satisfaction_reason: str = Field(
        description="This provides the reason behind the satisfaction score.")

    @validator('satisfaction')
    def checkScore(cls, field):
        if field > 10 or field < 0:
            raise ValueError('Badly formed satisfaction score')
        return field

    
    
    
pydantic_parser = PydanticOutputParser(pydantic_object=ConversationInfo)
format_instructions =pydantic_parser.get_format_instructions()
data={
  "question": "Can I talk to a human agent?",
  "answer": "Yes, you can talk to a human agent by contacting our Customer Support team. They will be happy to assist you with any queries or specific requests you may have."}

output_parser_template_string = """You are a Pixartprinting's Multinational Customer Care Agent 
As a dedicated agent, your primary mission is to assist our valued customers in the most efficient way possible.
You should always start by greeting the customers and greet them back.
You'll be addressing various topics, including:
    -Product Information: Offering comprehensive details on Pixart's diverse range of products, covering types, materials, and their applications.
    -Design Services: Providing valuable insights into our exceptional design services and guiding customers on how to access them.
    -File Uploads: Assisting customers with step-by-step instructions for hassle-free file uploads.

Take the conversation below delimited by triple backticks and use it to have more info about the customers issues.

conversation summary: ```{conversation_summary}```

then based on the summary give the conversation a score 1-10 for how likely it is to succeed.
Neutral conversations should have a score between 3 and 5.

{format_instructions}
"""

prompt = ChatPromptTemplate.from_template(template=output_parser_template_string)

messages = prompt.format_messages(conversation_summary=data, format_instructions=format_instructions)

output=llm(messages)
output.content

convMetaData = pydantic_parser.parse(output.content)
convMetaDataJson = convMetaData.json(indent=2)

print(convMetaDataJson)