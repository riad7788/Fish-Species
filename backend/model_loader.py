# backend/model_loader.py
import os
import json
import torch

def load_models(base_dir):
    """
    Load classifier and class names
    """
    # Paths
    classifier_path = os.path.join(base_dir, "models", "classifier.pt")
    class_names_path = os.path.join(base_dir, "models", "class_names.json")

    # Load classifier
    classifier = torch.load(classifier_path, map_location="cpu")
    classifier.eval()

    # Load class names
    with open(class_names_path, "r") as f:
        class_names = json.load(f)

    return classifier, class_names
