import streamlit as st
import google.generativeai as genai
import os
import time
from dotenv import load_dotenv


st.set_page_config(
    page_icon="📹",
    page_title="VideoIdeaForge"
)

# UI Code
st.header("VideoIdeaForge")
    
def stream_data(data):
    for word in data.split(" "):
        yield word + " "
        time.sleep(0.02)

# Configuring the LLM
load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
system_prompt = "You are a helpful AI bot. You are supposed to describe and analyze the input video as best as you can."
model_list = [ "Gemini 1.5 Flash", "Gemini 1.5 Pro"]
selected_model = st.selectbox("Choose Gemini model", model_list)
prompt = st.text_area("Input Prompt", placeholder="Enter Prompt here")
video_file_st = st.file_uploader("Upload Video File", type=["mp4", "mpeg", "mov", "avi", "x-flv", "mpg", "webm", "wmv", "3gpp"])

if st.button("Generate"):
            
    if video_file_st:
        st.video(video_file_st)
        with open(video_file_st.name, "wb") as f:
            f.write(video_file_st.read())

        # Google File API, file upload
        progress_text = "Analyzing video..."
        my_bar = st.progress(0, text=progress_text)
        video_file = genai.upload_file(video_file_st.name)
        percent_complete = 25
        while video_file.state.name=="PROCESSING":
            percent_complete += 15
            if percent_complete >= 100:
                percent_complete=95
            print('.', end='')
            time.sleep(1)
            video_file = genai.get_file(video_file.name)
            my_bar.progress(percent_complete, text=progress_text)
        my_bar.empty()
        st.success("Analyzed Successfully!!" , icon="✅")
        if video_file.state.name=="FAILED":
            raise ValueError(video_file.state.name)
        while True:
            with st.spinner("Generating..."):
                if selected_model=="Gemini 1.5 Pro":
                    selected_model="gemini-1.5-pro"
                else:
                    selected_model="gemini-1.5-flash"
                model = genai.GenerativeModel(model_name=selected_model, generation_config=genai.GenerationConfig(temperature=0.6), system_instruction=system_prompt)
                response = model.generate_content([prompt, video_file])
                if response:
                    response.text
                    break
        os.remove(video_file_st.name)         # Remove file from local system
        genai.delete_file(video_file.name)    # Remove file from google file server
    else:
        st.warning("Video not Uploaded!!", icon="⚠️")
