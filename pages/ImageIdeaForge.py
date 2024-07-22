from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import base64
from io import BytesIO


def image_to_base64(image : BytesIO) -> base64:
    img_base64 = base64.b64encode(image.getvalue()).decode("utf-8")
    return img_base64
st.set_page_config(
    page_icon="📸",
    page_title="ImageIdeaForge"
)

# UI Code
st.header("ImageIdeaForge")
    
# Choosing the LLM
model_list = ["llava:7b", "bakllava:latest"]
selected_model = st.selectbox("Choose the LLM", model_list)
user_input = st.text_area("Input")
uploaded_image = st.file_uploader("Choose Image", type=['png', 'jpg', 'jpeg', 'jfif'])

if uploaded_image: # Image Preview
    st.image(uploaded_image, width=256)

    if st.button("Generate"):
        while True:
            with st.spinner("Generating..."):
                img_b64 = image_to_base64(uploaded_image)
                
                llm = Ollama(model=selected_model, temperature=0.7, keep_alive=0, num_ctx=256)

                llm_with_image_context = llm.bind(images=[img_b64])

                system_prompt = """You are an assistant who describes the content and composition of images. Describe only what you see in the image, not what you think the image is about.Be factual and literal. Do not use metaphors or similes. Be concise, Create a image generation prompt that fits the image, don't use "", don't use imperatives like "generate, create" etc., just describe what you see in the image in a way that it could be used as a prompt for that specific image. If there are words or something that looks like a copyright symbol, make no mention of it."""

                chat_template = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt),
                        ("human", "{user_input}."),

                    ]
                )

                str_output_parser = StrOutputParser()

                chain = chat_template | llm_with_image_context | str_output_parser
                result_prompt = chain.invoke({"user_input":user_input})
                if result_prompt:
                    st.sidebar.markdown("## Output")
                    st.sidebar.write(result_prompt)
                    break
else:
    st.warning("Image not Uploaded!!", icon="⚠️")



