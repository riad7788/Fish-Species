import os
import torch
from backend.model_loader import load_models
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Load encoder
encoder = torch.load(os.path.join(BASE_DIR, "models", "encoder.pt"), map_location="cpu")
encoder.eval()

# Load classifier & class names
classifier, class_names = load_models(BASE_DIR)
