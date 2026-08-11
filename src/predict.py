import argparse

import torch
from PIL import Image

from dataset import get_test_transform
from model import CNNModel


CLASS_NAMES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck",
]


def load_model(model_path, device):  
    model = CNNModel().to(device) 

    checkpoint = torch.load(model_path, map_location=device) 

    model.load_state_dict(checkpoint["model_state_dict"]) 
 
    model.eval() 

    return model


def predict_image(model, image_path, device):

    transform = get_test_transform()

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image) 
    image_tensor = image_tensor.unsqueeze(0).to(device) 

    with torch.no_grad():
        outputs = model(image_tensor) 

        probabilities = torch.softmax(outputs, dim=1) 

        predicted_class = probabilities.argmax(dim=1).item() 
        confidence = probabilities[0, predicted_class].item() 

    return CLASS_NAMES[predicted_class], confidence


def main():
    parser = argparse.ArgumentParser(
        description="Predict the CIFAR-10 class of an image."
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="models/best_model.pth",
        help="Path to the trained model."
    )

    args = parser.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    model = load_model(
        args.model,
        device
    )

    predicted_class, confidence = predict_image(
        model,
        args.image,
        device
    )

    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
    