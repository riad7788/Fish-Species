import torch
import os
import json

def load_models(base_dir):
    classifier_path = os.path.join(base_dir, "models", "classifier.pt")
    class_names_path = os.path.join(base_dir, "models", "class_names.json")

    classifier = torch.load(classifier_path, map_location="cpu")
    classifier.eval()

    with open(class_names_path, "r") as f:
        class_names = json.load(f)

    return classifier, class_names
