from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import os
import json

st.set_page_config(
    page_icon="💡",
    page_title="IdeaForge"
)
st.header("IdeaForge")


# Choosing the LLM
model_list = ["llama3.1:latest", "mistral:latest", "llava:7b", "gemma2:latest"]
selected_model = st.selectbox("Choose the LLM", model_list)

chat_model = Ollama(model=selected_model, temperature=0.7, keep_alive=0, num_ctx=256)

system_prompt = """You are a helpful AI bot. You are to act as a prompt maker for stable diffusion with the following guidelines: 
- Break keywords by commas. 
- Provide high-quality, non-verbose, coherent, brief, concise, and not superfluous prompts. 
- Construct the prompt with the component format: 
    1. Start with the subject and description. 
    2. Follow with scene description. 
    3. Finish with background and description. 
- Limit yourself to no more than 7 keywords per component 
- Include all the keywords from the user's request verbatim as the main subject of the response. 
- Be varied and creative. 
- Limit yourself to 100-150 words.
- Do not enumerate or enunciate components. 
- Do not include any additional information in the response. 
"""

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{user_input}."),

    ]
)

str_output_parser = StrOutputParser()

chain = chat_template | chat_model | str_output_parser


#Loading prompt style presets from json
def style_loader(file_path : str) -> list:
    with open(file_path, 'r') as f:
        style_preset = json.load(f)
    return style_preset

# sdxl_styles = style_loader(os.path.join(".\presets", "sdxl_styles.json")) #json file content
style_prompts_file = style_loader(os.path.join("presets", "styles.json")) #json file content


# UI Code
user_input = st.text_area("Input")

def key_extractor(data : list) -> list:
    keys_list = []
    for index in range(len(data)):
        keys_list.append(data[index]["name"])
    return keys_list

def style_search(name : str, data : list) -> str:
    for item in data:
        if name.lower()==item["name"].lower():
            return item["Keywords"]

# style_choices = st.multiselect(
#     "Choose style",
#     key_extractor(sdxl_styles)
# )

style_choices = st.multiselect(
    "Choose fashion/photography styles",
    key_extractor(style_prompts_file)
)
styles = ""
for choice in style_choices:
        styles = styles + style_search(choice, style_prompts_file) + ", "

if st.button("Generate"):
    result_prompt = chain.invoke({"user_input":user_input})
    st.write(f"{result_prompt}" + f"\n {styles}" )
    # for choice in style_choices:
    #     st.write(name_search(choice, sdxl_styles))



