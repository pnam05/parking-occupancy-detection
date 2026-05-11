import torch
import torch.nn as nn
from torchvision import transforms, models
import config

def load_parking_model():
    model = models.mobilenet_v3_small(weights=None)
    num_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_features, len(config.CLASS_NAMES))
    
    model.load_state_dict(torch.load(config.MODEL_WEIGHTS, map_location=config.DEVICE))
    model = model.to(config.DEVICE)
    model.eval()
    return model

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])