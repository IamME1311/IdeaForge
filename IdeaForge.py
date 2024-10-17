from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import os
from .pages.utils import *

st.set_page_config(
    page_icon="💡",
    page_title="IdeaForge"
)
st.header("IdeaForge")


# Choosing the LLM
model_list = ["llama3.1:latest", "mistral:latest", "llava:7b", "gemma2:latest"]
selected_model = st.selectbox("Choose the LLM", model_list)

chat_model = Ollama(model=selected_model, temperature=0.7, keep_alive=0, num_ctx=256)

system_prompt = "You are a helpful AI assistant who helps user to make prompts for generating images using stable diffusion. The prompt must contain information about the surroundings which is in sync with the subject of the image. Try to describe the subject of the image within the first few words of the prompt. Don't use imperatives like 'generate, create', just specify things that can used as prompt for the image. Try to avoid words that are NSFW. The prompt must follow this structure : image composition, major details and finally all the fine details."

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{user_input}."),

    ]
)

str_output_parser = StrOutputParser()

chain = chat_template | chat_model | str_output_parser



# UI Code
user_input = st.text_area("Input")

style_prompts_file = style_loader(os.path.join("presets", "styles.json")) #json file content
style_choices = st.multiselect(
    "Choose fashion/photography styles",
    key_extractor(style_prompts_file)
)
styles = ""
for choice in style_choices:
        styles = styles + style_search(choice, style_prompts_file) + ", "



if st.button("Generate"):
    result_prompt = chain.invoke({"user_input":user_input})
    result_prompt = result_prompt + f"\n {styles}"
    st.write_stream(stream_response(result_prompt))



