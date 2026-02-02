import torch
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict(image: Image.Image, encoder, classifier, class_names):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = encoder(img_tensor)
        output = classifier(features)
        probs = torch.softmax(output, dim=1).numpy()[0]
        idx = probs.argmax()
        return class_names[idx], probs[idx]
