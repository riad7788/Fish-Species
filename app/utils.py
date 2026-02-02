import requests

API_URL = "http://127.0.0.1:8000/predict"

def send_image(image):
    files = {"file": image}
    response = requests.post(API_URL, files=files)
    return response.json()
