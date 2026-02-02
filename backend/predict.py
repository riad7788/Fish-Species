import torch
import numpy as np
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def predict_image(model, image: Image.Image):
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        feat = model.encoder(x)
        logits = model.classifier(feat)
        probs = torch.softmax(logits, dim=1).numpy()[0]

    top3_idx = probs.argsort()[-3:][::-1]

    result = {
        "predicted_species": model.class_names[top3_idx[0]],
        "confidence": float(probs[top3_idx[0]]),
        "top3": [
            (model.class_names[i], float(probs[i]))
            for i in top3_idx
        ]
    }
    return result
