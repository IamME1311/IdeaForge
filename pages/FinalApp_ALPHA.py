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

    t1_system_prompt = "You are a helpful AI assistant who helps user to make prompts for generating images using stable diffusion. The prompt must contain information about the surroundings which is in sync with the subject of the image. Try to describe the subject of the image within the first few words of the prompt. Don't use imperatives like 'generate, create', just specify things that can used as prompt for the image. Try to avoid words that are NSFW. The prompt must follow this structure : image composition, major details and finally all the fine details."

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
    # import socket


    # def fromPhotoshop():
        
    #     return

    # def toComfyUI(data:bytes) -> None:
    #     HOST = "127.0.0.1"
    #     PORT = 9000
    #     server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #     server_socket.bind((HOST, PORT))
    #     server_socket.listen(1)
    #     conn, addr = server_socket.accept()
    #     send_data = data.encode('utf-8')
    #     conn.sendall(send_data)
    def image_to_base64(image : BytesIO) -> base64:
        img_base64 = base64.b64encode(image.getvalue()).decode("utf-8")
        return img_base64
    # UI Code
    st.header("ImageIdeaForge")
        
    # Choosing the LLM
    t2_model_list = ["llava:7b", "bakllava:latest", "llava-llama3"]
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

                    # t2_system_prompt = """
                    # The model will describe the content and composition of images in a factual and literal manner, focusing on realistic depictions of people. The description should be concise, detailing only what is visible in the image without making assumptions or interpretations. The descriptions should be suitable for generating images of realistic people, with a focus on Indian features and settings. Fantasy, anime, and sketch-like elements should be omitted.

                    # Structure and Focus:

                    #     - Subject Description: Begin by identifying the main subject in the image, specifying gender, age range (e.g., 20-40 years), and any distinguishing characteristics such as body posture or gestures.
                        
                    #     - Facial Features: Describe the facial features with precision, including details like eye shape, size, and color; hairstyle and length; eyebrow thickness and shape; complexion; jawline definition; and lip shape and fullness. Aim for an accurate portrayal of Indian features.
                        
                    #     - Clothing: Describe the type of clothing the person is wearing (e.g., t-shirt, sari, jeans) without emphasizing colors. Focus on the style and fit rather than specific details that may not be consistent across similar images.

                    #     - Background and Setting: Provide a brief description of the background or setting, focusing on whether it is indoors or outdoors, and noting any relevant environmental elements (e.g., urban street, office, natural landscape).

                    #     - Lighting and Mood: Mention the lighting conditions if they are evident (e.g., natural light, artificial light, shadows) and the overall mood or atmosphere conveyed by the scene.

                    # Example Description:
                    #     A 30-year-old Indian woman standing outdoors in a relaxed pose. She has medium-length, wavy black hair parted to the side, large brown eyes with thick eyebrows, a medium complexion, and full lips. She is wearing a casual kurti and jeans. The background shows a busy urban street with a few trees visible. The lighting is natural, suggesting it's daytime, with soft shadows.
                    # """


                    ## alternate system prompt

                    t2_system_prompt = """ 
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


