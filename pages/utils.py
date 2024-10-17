import time
import json
from io import BytesIO
import base64
from PIL import Image
import yaml


############ Helper functions ############
def image_to_base64(image : BytesIO) -> base64:
    img_base64 = base64.b64encode(image.getvalue()).decode("utf-8")
    return img_base64

def yaml_extractor() -> dict:
    with open('./presets/prompts.yaml', 'r') as f:
        yaml_data = yaml.safe_load(f)
    return yaml_data

def to_pil_image(data)->Image:
    img = Image.open(data)
    bytes_arr = BytesIO()
    img.save(bytes_arr, format="PNG")
    return img

def key_extractor(data : list) -> list:
    keys_list = []
    for index in range(len(data)):
        keys_list.append(data[index]["name"])
    return keys_list

def style_search(name : str, data : list) -> str:
    for item in data:
        if name.lower()==item["name"].lower():
            return item["Keywords"]
        
def stream_response(data):
    for word in data.split():
        yield word + " "
        time.sleep(0.05)

#Loading prompt style presets from json
def style_loader(file_path : str) -> list:
    with open(file_path, 'r') as f:
        style_preset = json.load(f)
    return style_preset
        
