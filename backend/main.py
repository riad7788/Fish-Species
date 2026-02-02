from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

from model_loader import FishModel
from predict import predict_image

app = FastAPI(title="Fish Species Detection API")

model = FishModel()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    result = predict_image(model, image)
    return result

