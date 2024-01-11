import requests
import base64
import numpy as np



BASE = "http://127.0.0.1:6000/predict"
img_path = "Alzheimer_s Dataset/test/VeryMildDemented/26 (50).jpg"

with open(img_path, "rb") as img_file:
    b64_string = base64.b64encode(img_file.read()).decode('utf-8')

headers = {"Content-type": "application/json"}
data = {"base64_image": b64_string}

response = requests.post(BASE, headers=headers, json=data)

print(response)
print(response.json())
