from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate
import google.generativeai as genai

import streamlit as st

import os
from pathlib import Path

from typing import Tuple
from dotenv import load_dotenv
from utils import *
import logging


# Logging setup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Configuration:

    def __init__(self):

        try:
            load_dotenv("D:\\Software_Projects\\Assets\\.env")
            self.api_key = os.getenv("GOOGLE_API_KEY")
            if not self.api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
        except Exception as e:
            logger.error(f"Failed to initialize api key : {e}")
            raise
    
    @staticmethod
    def configure_app() -> None:
        try:
            st.set_page_config(
                    page_icon="📸",
                    page_title="ImageIdeaForge"
                )
        except Exception as e:
            logger.error(f"Failed to configure app : {e}")
            raise

    def configure_models(self)-> Tuple[genai.GenerativeModel | Ollama, str]:

        try:
            # Choosing the LLM
            model_list = ["llava:7b", "bakllava:latest", "llava-llama3", "gemini-1.5-flash"]
            selected_model = st.selectbox("Choose the LLM", model_list)

            if selected_model == "gemini-1.5-flash":
                genai.configure(api_key=self.api_key)
                llm = genai.GenerativeModel(model_name=selected_model, generation_config=genai.GenerationConfig(temperature=0.6))
            else:
                llm = Ollama(model=selected_model, temperature=0.7, keep_alive=0, num_ctx=256)
            
            return llm, selected_model
        
        except Exception as e:
            logger.error(f"Failed to configure the models : {e}")
            raise



def main():

    # Configuration.configure_app()
    # UI Code
    st.header("ImageIdeaForge")
        
    llm, selected_model = Configuration().configure_models()

    # Choosing prompt
    prompt_options = ["Default", "Overall SDXL", "Background Description", "Product Description", "Custom"]
    selected_prompt = st.selectbox("Choose Prompt", prompt_options)

    if selected_prompt=="Custom":
        selected_prompt = st.text_area("Input", value="describe the composition,dress, person and background")
    elif selected_prompt=="Default":
        default_prompt = """
                            The model will describe the content and composition of images in a factual and literal manner, focusing on realistic depictions of people. The description should be detailed yet concise, capturing only what is visible in the image without making assumptions or interpretations. The descriptions should be suitable for generating images of realistic people, with a focus on Indian features and settings. Fantasy, anime, and sketch-like elements should be omitted.

                            Structure and Focus:

                                - Subject Description: Start by identifying the main subject in the image, specifying gender, age range (e.g., 20-40 years), body posture, and any visible gestures or actions. Mention any distinctive physical attributes that stand out.

                                - Facial Features: Provide a detailed description of the facial features, including eye shape, size, and color; hairstyle, length, and texture; eyebrow thickness, shape, and alignment; complexion; jawline and chin structure; nose shape; and lip shape, size, and fullness. Emphasize Indian features, reflecting common characteristics.

                                - Clothing: Describe the type of clothing the person is wearing, focusing on style, fit, and any visible patterns or textures. Avoid emphasizing specific colors, instead describing the clothing in a way that reflects the cultural and situational context (e.g., casual, formal, traditional).

                                - Background and Setting: Describe the background or setting, noting whether the scene is indoors or outdoors, and providing details about the environment. Mention any notable objects, architecture, or natural elements that are visible in the background.

                                - Lighting and Mood: Mention the lighting conditions, such as the direction and intensity of light, and describe any visible shadows or reflections. Convey the overall mood or atmosphere of the scene, whether it's calm, energetic, formal, or casual.

                            Example Description:
                                A 35-year-old Indian man standing in an office environment. He has short, neatly trimmed black hair with a slight wave, and a well-groomed beard that outlines his square jawline. His eyes are almond-shaped, dark brown, and framed by thick, slightly arched eyebrows. He has a medium complexion with a slight tan. His lips are medium-sized, with a defined shape, and he has a prominent nose with a straight bridge.

                                He is dressed in a fitted, long-sleeve button-down shirt with subtle pinstripes, tucked into tailored trousers. The shirt collar is open, and he is not wearing a tie. He is holding a tablet in one hand and gesturing with the other, as if explaining something.

                                The background shows a modern office with large windows, through which a cityscape with tall buildings and a clear sky is visible. The room is well-lit with natural light, creating soft shadows on the man's face and adding depth to the scene. The mood is professional yet approachable, suggesting a collaborative work environment.
                            """
        selected_prompt = default_prompt
    else:
        prompt_dump = yaml_extractor()
        selected_prompt = prompt_dump[selected_prompt]


    try:
        # path or file
        choice = st.toggle("upload via path or uploader")
        uploaded_image = None
        if choice is True:
            uploaded_image = st.file_uploader("Choose Image", type=['png', 'jpg', 'jpeg', 'jfif'],)
        elif choice is False:
            path = Path(st.text_input("Enter path", placeholder="Enter path here"))
            if path.exists() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".jfif"}:
                uploaded_image = to_pil_image(path)
            elif path is None or not path.exists():
                raise ValueError("Incorrect image path !!")
    except Exception as e:
        logger.error(f"Error in getting image : {e}")
        raise

    if uploaded_image: # Image Preview
        st.image(uploaded_image, width=256)

        if st.button("Generate"):
            with st.spinner("Generating..."): 
                # Gemini Block
                if selected_model.startswith("gemini"):
                    response = llm.generate_content([selected_prompt, uploaded_image])
                    st.sidebar.markdown("## Output")
                    st.sidebar.write(response.text)

                else:  # Ollama Block
                    img_b64 = image_to_base64(uploaded_image)
                    llm_with_image_context = llm.bind(images=[img_b64])       
                    chat_template = ChatPromptTemplate.from_messages(
                        [
                            ("human", "{user_input}."),
                        ]
                    )
                    chain = chat_template | llm_with_image_context
                    result_prompt = chain.invoke({"user_input":selected_prompt})
                    st.sidebar.markdown("## Output")
                    st.sidebar.write_stream(stream_response(result_prompt))
    else:
        st.warning("Image not Uploaded!!", icon="⚠️")


if __name__ == "__main__":
    main()

