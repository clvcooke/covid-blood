from torchvision import transforms
from utils import setup_torch, load_model
from dataloader import load_all_patients
from models.imagenet import get_model


def predict():
    """
    Process predictions for wbc classification
    :return:
    """
    setup_torch(0, 1, 3)
    image_size = 224
    