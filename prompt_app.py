from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st


# Choosing the LLM
chat_model = ChatOllama(model="llama3:latest", temperature=0.7, keep_alive=-1, num_ctx=256)

system_prompt = """Act as a prompt maker with the following guidelines: 
- Break keywords by commas. 
- Provide high-quality, non-verbose, coherent, brief, concise, and not superfluous prompts. 
- Focus solely on the visual elements of the picture; avoid art commentaries or intentions. 
- Construct the prompt with the component format: 
    1. Start with the subject and keyword description. 
    2. Follow with scene keyword description. 
    3. Finish with background and keyword description. 
- Limit yourself to no more than 7 keywords per component 
- Include all the keywords from the user's request verbatim as the main subject of the response. 
- Be varied and creative. - Always reply on the same line and no more than 100 words long. 
- Do not enumerate or enunciate components. 
- Do not include any additional information in the response. 
The following is an illustrative example for you to see how to construct a prompt your prompts should follow this format but always coherent to the subject worldbuilding or setting and consider the elements relationship. 
Example: Subject: Demon Hunter, Cyber City. 
prompt: A Demon Hunter, standing, lone figure, glow eyes, deep purple light, cybernetic exoskeleton, sleek, metallic, glowing blue accents, energy weapons. Fighting Demon, grotesque creature, twisted metal, glowing red eyes, sharp claws, Cyber City, towering structures, shrouded haze, shimmering energy. 
Make a prompt for the following Subject: """

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{user_input}."),

    ]
)

str_output_parser = StrOutputParser()

chain = chat_template | chat_model | str_output_parser

# Streamlit code

user_input = st.text_area("Input")

if st.button("Generate"):
    st.write(chain.invoke({"user_input":user_input}))