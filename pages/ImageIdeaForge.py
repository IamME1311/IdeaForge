from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import google.generativeai as genai
from seaborn import load_dataset
import streamlit as st
import base64
from io import BytesIO
from PIL import Image
import os
from dotenv import load_dotenv
import time
import yaml

########################WIP###################################
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
##########################WIP####################################



############ Helper functions ############
def image_to_base64(image : BytesIO) -> base64:
    img_base64 = base64.b64encode(image.getvalue()).decode("utf-8")
    return img_base64

def yaml_extractor():
    with open('./presets/prompts.yaml', 'r') as f:
        yaml_data = yaml.safe_load(f)
    return yaml_data



st.set_page_config(
    page_icon="📸",
    page_title="ImageIdeaForge"
)

# UI Code
st.header("ImageIdeaForge")
    
# Choosing the LLM
model_list = ["llava:7b", "bakllava:latest", "llava-llama3", "gemini-1.5-flash"]
selected_model = st.selectbox("Choose the LLM", model_list)

# Choosing prompt
prompt_options = ["Default", "Overall SDXL", "Background Description", "Product Description", "Custom"]
selected_prompt = st.selectbox("Choose Prompt", prompt_options)

if selected_prompt=="Custom":
    user_input = st.text_area("Input", value="describe the composition,dress, person and background")
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




choice = st.toggle("upload via path or uploader")

if choice:
    uploaded_image = st.file_uploader("Choose Image", type=['png', 'jpg', 'jpeg', 'jfif'])
else:
    path = st.text_input("Enter path", placeholder="Enter path here")
    path = path.replace('"', '')
    path = path.replace('\\', '/')
    img = Image.open(path)
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    uploaded_image = img_byte_arr


if uploaded_image: # Image Preview
    st.image(uploaded_image, width=256)

    if st.button("Generate"):
        while True:
            with st.spinner("Generating..."):
                
                ### Gemini API 
                if selected_model == "gemini-1.5-flash":
                    with open(uploaded_image.name, "wb") as f:
                        f.write(uploaded_image.read())

                    #Google File API, file upload
                    progress_text = "Analyzing Image..."
                    my_bar = st.progress(0, text=progress_text)
                    image_file = genai.upload_file(uploaded_image.name)
                    percent_complete = 25
                    while image_file.state.name=="PROCESSING":
                        percent_complete += 15
                        if percent_complete >= 100:
                            percent_complete=95
                        print('.', end='')
                        time.sleep(1)
                        image_file = genai.get_file(image_file.name)
                        my_bar.progress(percent_complete, text=progress_text)
                    my_bar.empty()
                    st.success("Analyzed Successfully!!" , icon="✅")
                    if image_file.state.name=="FAILED":
                        raise ValueError(image_file.state.name)
                    
                    # system_prompt = "describe the image"
                    load_dotenv()
                    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                    llm = genai.GenerativeModel(model_name=selected_model, generation_config=genai.GenerationConfig(temperature=0.6))
                    response = llm.generate_content([selected_prompt, image_file])
                    if response:
                        st.sidebar.write(response.text)
                    os.remove(uploaded_image.name)    
                    genai.delete_file(image_file.name)
                    break


                else: 
                    img_b64 = image_to_base64(uploaded_image)
                    
                    llm = Ollama(model=selected_model, temperature=0.7, keep_alive=0, num_ctx=256)

                    llm_with_image_context = llm.bind(images=[img_b64])

                    
                    chat_template = ChatPromptTemplate.from_messages(
                        [
                            # ("system", system_prompt),
                            ("human", "{user_input}."),

                        ]
                    )

                    str_output_parser = StrOutputParser()

                    chain = chat_template | llm_with_image_context | str_output_parser
                    result_prompt = chain.invoke({"user_input":selected_prompt})
                    if result_prompt:
                        st.sidebar.markdown("## Output")
                        st.sidebar.write(result_prompt)
                        break
else:
    st.warning("Image not Uploaded!!", icon="⚠️")



