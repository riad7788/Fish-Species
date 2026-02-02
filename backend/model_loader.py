import json
import torch
import torch.nn as nn
from torchvision import models

DEVICE = "cpu"

class FishModel:
    def __init__(self):
        # Encoder (DINO / SimCLR / BYOL)
        self.encoder = models.resnet50(weights=None)
        self.encoder.fc = nn.Identity()
        self.encoder.load_state_dict(
            torch.load("../models/encoder_dino.pt", map_location=DEVICE)
        )
        self.encoder.eval()

        # Classifier
        self.classifier = nn.Linear(2048, self._num_classes())
        self.classifier.load_state_dict(
            torch.load("../models/classifier.pt", map_location=DEVICE)
        )
        self.classifier.eval()

        with open("../models/class_names.json") as f:
            self.class_names = json.load(f)

    def _num_classes(self):
        with open("../models/class_names.json") as f:
            return len(json.load(f))
