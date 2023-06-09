import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()


st.set_page_config(
   page_title="AI Scribe",
#    page_icon="🧊",
   layout="wide",
   initial_sidebar_state="expanded",
)



col1, col2, col3 = st.columns(3)

with col1:
    st.subheader('Parameters')
    openAI_api_key = st.text_input('OpenAI API Key', os.environ["OPENAI_API_KEY"])


with col2:
    st.subheader('Input')




with col3:
    st.subheader('Output')



from langchain.output_parsers import StructuredOutputParser, ResponseSchema


from langchain.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

# How you would like your reponse structured. This is basically a fancy prompt template
response_schemas = [
    ResponseSchema(name="cc", description="chief complaint, usually a single label"),
    ResponseSchema(name="hpi", description="history of present illness"),
    ResponseSchema(name="pmhx", description="past medical history, as a list of comma separated values"),
    ResponseSchema(name="meds", description="medications, as a list of comma separated values"),
    ResponseSchema(name="all", description="allergies, as a list of comma separated values"),
    ResponseSchema(name="pe", description="physical exam")
]

# How you would like to parse your output
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()

default_format_instructions = """
The output should be a markdown code snippet formatted in the following schema:

```json
{
    "cc": string  // chief complaint, usually a single label
    "hpi": string  // history of present illness
    "pmhx": list  // past medical history, as a list of comma separated values
    "meds": list  // medications, as a list of comma separated values
    "all": list  // allergies, as a list of comma separated values
    "pe": string  // physical exam
    }
```
"""

with col1:
    format_instructions = st.text_area("Format Instructions", default_format_instructions)

prompt=PromptTemplate(
    template="You will be given a details about a patient's medical issue. Reformat it using the following instructions:\n{format_instructions}",
    input_variables=["format_instructions"],
)
system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)

human_template="{narrative}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

default_narrative = """
This is a 20 year-old male with asthma who comes to the ER today because he's feeling more short of breath. He has had a cough and fever for the past 3 days after recently being exposed to COVID-19. He takes Ventolin and Flovent for his asthma, and has no allergies. On exam, he has expiratory wheezing bilaterally, with a prolonged expiratory phase, accessory muscle use.
"""

with col2:
    narrative_input = st.text_area('Narrative', default_narrative)


# get a chat completion from the formatted messages
messages = chat_prompt.format_prompt(narrative=narrative_input, format_instructions=format_instructions).to_messages()

from langchain.chat_models import ChatOpenAI
chat = ChatOpenAI(temperature=0)

result = chat(messages)
with col3:
    st.write(output_parser.parse(result.content))

st.write(messages)