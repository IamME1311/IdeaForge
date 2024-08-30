from langchain_community.llms.ollama import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import base64
from io import BytesIO
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
st.set_page_config(
    page_icon="📸",
    page_title="ImageIdeaForge"
)

# UI Code
st.header("ImageIdeaForge")
    
# Choosing the LLM
model_list = ["llava:7b", "bakllava:latest", "llava-llama3"]
selected_model = st.selectbox("Choose the LLM", model_list)
user_input = st.text_area("Input", value="describe the composition,dress, person and background")
uploaded_image = st.file_uploader("Choose Image", type=['png', 'jpg', 'jpeg', 'jfif'])

if uploaded_image: # Image Preview
    st.image(uploaded_image, width=256)

    if st.button("Generate"):
        while True:
            with st.spinner("Generating..."):
                img_b64 = image_to_base64(uploaded_image)
                
                llm = Ollama(model=selected_model, temperature=0.7, keep_alive=0, num_ctx=256)

                llm_with_image_context = llm.bind(images=[img_b64])

                # system_prompt = """
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

                system_prompt = """
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



