from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import os
import json



st.header("IdeaForge")

model_list = ["llama3:latest", "mistral:latest", "llava:7b", "gemma2:latest"]
selected_model = st.selectbox("Choose the LLM", model_list)

# Choosing the LLM
chat_model = ChatOllama(model=selected_model, temperature=0.7, keep_alive=-1, num_ctx=256)

system_prompt = """Act as a prompt maker with the following guidelines: 
- Break keywords by commas. 
- Provide high-quality, non-verbose, coherent, brief, concise, and not superfluous prompts. 
- Focus solely on the visual elements of the picture; avoid art commentaries or intentions. 
- Construct the prompt with the component format: 
    1. Start with the subject and description. 
    2. Follow with scene description. 
    3. Finish with background and description. 
- Limit yourself to no more than 7 keywords per component 
- Include all the keywords from the user's request verbatim as the main subject of the response. 
- Be varied and creative. 
- Always reply on the same line and no more than 100 words long. 
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

#Loading prompt style presets

def style_loader(file_path):
    with open(file_path, 'r') as f:
        style_preset = json.load(f)
    return style_preset

# sdxl_styles = style_loader(os.path.join(".\presets", "sdxl_styles.json")) #json file content
style_prompts = style_loader(os.path.join(".\presets", "styles.json")) #json file content


# Streamlit code

user_input = st.text_area("Input")

def key_extractor(data):
    keys_list = []
    for index in range(len(data)):
        keys_list.append(data[index]["name"])
    return keys_list

def name_search(name, data):
    for item in data:
        if name.lower()==item["name"].lower():
            return item["Keywords"]

# style_choices = st.multiselect(
#     "Choose style",
#     key_extractor(sdxl_styles)
# )

style_choices_2 = st.multiselect(
    "Choose fashion/photography styles",
    key_extractor(style_prompts)
)
styles = ""
for choice in style_choices_2:
        styles = styles + name_search(choice, style_prompts) + ", "
if st.button("Generate"):
    result_prompt = chain.invoke({"user_input":user_input})
    st.write(f"{result_prompt}" + f"\n {styles}" )
    
    # for choice in style_choices:
    #     st.write(name_search(choice, sdxl_styles))