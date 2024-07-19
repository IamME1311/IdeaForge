import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import google.generativeai as genai
import time
from dotenv import load_dotenv
import os
import json
import base64
from io import BytesIO

st.set_page_config(
        page_icon="💡",
        page_title="IdeaForge"
    )

st.header("Welcome to IdeaForge!!")
IdeaForge, ImageIdeaForge, VideoIdeaForge = st.tabs(["IdeaForge", "ImageIdeaForge", "VideoIdeaForge"])

with IdeaForge:
    st.header("IdeaForge")


    # Choosing the LLM
    t1_model_list = ["llama3:latest", "mistral:latest", "llava:7b", "gemma2:latest"]
    t1_selected_model = st.selectbox("Choose the LLM", t1_model_list, key="ideaforge_model")

    t1_chat_model = ChatOllama(model=t1_selected_model, temperature=0.7, keep_alive=0, num_ctx=256)

    t1_system_prompt = """Act as a prompt maker with the following guidelines: 
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

    t1_chat_template = ChatPromptTemplate.from_messages(
        [
            ("system", t1_system_prompt),
            ("human", "{user_input}."),

        ]
    )

    t1_str_output_parser = StrOutputParser()

    t1_chain = t1_chat_template | t1_chat_model | t1_str_output_parser


    #Loading prompt style presets from json
    def style_loader(file_path : str) -> list:
        with open(file_path, 'r') as f:
            style_preset = json.load(f)
        return style_preset

    # sdxl_styles = style_loader(os.path.join(".\presets", "sdxl_styles.json")) #json file content
    t1_style_prompts_file = style_loader(os.path.join(".\presets", "styles.json")) #json file content


    # UI Code
    t1_user_input = st.text_area("Input", key="ideaforge_prompt")

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

    t1_style_choices = st.multiselect(
        "Choose fashion/photography styles",
        key_extractor(t1_style_prompts_file), key="ideaforge_style"
    )
    t1_styles = ""
    for choice in t1_style_choices:
            t1_styles = t1_styles + style_search(choice, t1_style_prompts_file) + ", "

    if st.button("Generate", key="ideaforge_button"):
        t1_result_prompt = t1_chain.invoke({"user_input":t1_user_input})
        st.write(f"{t1_result_prompt}" + f"\n {t1_styles}")
        # for choice in style_choices:
        #     st.write(name_search(choice, sdxl_styles))


with ImageIdeaForge:
    def image_to_base64(image : BytesIO) -> base64:
        img_base64 = base64.b64encode(image.getvalue()).decode("utf-8")
        return img_base64
    # st.set_page_config(
    #     page_icon="📸",
    #     page_title="ImageIdeaForge"
    # )

    # UI Code
    st.header("ImageIdeaForge")
        
    # Choosing the LLM
    t2_model_list = ["llava:7b", "bakllava:latest"]
    t2_selected_model = st.selectbox("Choose the LLM", t2_model_list, key="imageideaforge_model")
    t2_user_input = st.text_area("Input", key="imageideaforge_prompt")
    uploaded_image = st.file_uploader("Choose Image", type=['png', 'jpg', 'jpeg'])

    if uploaded_image: # Image Preview
        st.image(uploaded_image, width=256)

        if st.button("Generate", key="imageideaforge_button"):
            while True:
                with st.spinner("Generating..."):
                    img_b64 = image_to_base64(uploaded_image)
                    
                    llm = Ollama(model=t2_selected_model, temperature=0.7, keep_alive=0, num_ctx=256)

                    llm_with_image_context = llm.bind(images=[img_b64])

                    t2_system_prompt = """You are an assistant who describes the content and composition of images. Describe only what you see in the image, not what you think the image is about.Be factual and literal. Do not use metaphors or similes. Be concise, Create a image generation prompt that fits the image, don't use "", don't use imperatives like "generate, create" etc., just describe what you see in the image in a way that it could be used as a prompt for that specific image. If there are words or something that looks like a copyright symbol, make no mention of it."""

                    t2_chat_template = ChatPromptTemplate.from_messages(
                        [
                            ("system", t2_system_prompt),
                            ("human", "{user_input}."),

                        ]
                    )

                    t2_str_output_parser = StrOutputParser()

                    t2_chain = t2_chat_template | llm_with_image_context | t2_str_output_parser
                    t2_result_prompt = t2_chain.invoke({"user_input":t2_user_input})
                    if t2_result_prompt:
                        st.sidebar.write(t2_result_prompt)
                        break
    else:
        st.warning("Image not Uploaded!!", icon="⚠️")


with VideoIdeaForge:
    # st.set_page_config(
    #     page_icon="📹",
    #     page_title="VideoIdeaForge"
    # )

    # UI Code
    st.header("VideoIdeaForge")
        
    def stream_data(data):
        for word in data.split(" "):
            yield word + " "
            time.sleep(0.02)

    # Configuring the LLM
    load_dotenv()
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    t3_system_prompt = "You are a helpful AI bot. You are supposed to describe and analyze the input video as best as you can."
    t3_model_list = [ "Gemini 1.5 Flash", "Gemini 1.5 Pro"]
    t3_selected_model = st.selectbox("Choose Gemini model", t3_model_list, key="videoideaforge_model")
    t3_prompt = st.text_area("Input Prompt", placeholder="Enter Prompt here", key="videoideaforge_prompt")
    video_file_st = st.file_uploader("Upload Video File", type=["mp4", "mpeg", "mov", "avi", "x-flv", "mpg", "webm", "wmv", "3gpp"])

    if st.button("Generate", key="videoideaforge_button"):
                
        if video_file_st:
            st.video(video_file_st)
            with open(video_file_st.name, "wb") as f:
                f.write(video_file_st.read())

            # Google File API, file upload
            progress_text = "Analyzing video..."
            progress_bar = st.progress(0, text=progress_text)
            video_file = genai.upload_file(video_file_st.name)
            percent_complete = 25
            while video_file.state.name=="PROCESSING":
                percent_complete += 15
                if percent_complete >= 100:
                    percent_complete=95
                print('.', end='')
                time.sleep(1)
                video_file = genai.get_file(video_file.name)
                progress_bar.progress(percent_complete, text=progress_text)
            progress_bar.empty()

            st.success("Analyzed Successfully!!" , icon="✅")

            if video_file.state.name=="FAILED":
                raise ValueError(video_file.state.name)
            
            # Text generation
            while True:
                with st.spinner("Generating..."):
                    if t3_selected_model=="Gemini 1.5 Pro":
                        t3_selected_model="gemini-1.5-pro"
                    else:
                        t3_selected_model="gemini-1.5-flash"
                    t3_model = genai.GenerativeModel(model_name=t3_selected_model, generation_config=genai.GenerationConfig(temperature=0.6), system_instruction=t3_system_prompt)
                    t3_response = t3_model.generate_content([t3_prompt, video_file])
                    if t3_response:
                        t3_response.text
                        break
                    
            os.remove(video_file_st.name)         # Remove file from local system
            genai.delete_file(video_file.name)    # Remove file from google file server

        else:
            st.warning("Video not Uploaded!!", icon="⚠️")


