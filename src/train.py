import os

import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import get_dataloaders
from model import CNNModel


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train() 

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images) 
        loss = criterion(outputs, labels) 

        loss.backward() 
        optimizer.step() 

        total_loss += loss.item()

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return average_loss, accuracy


def validate(model, dataloader, criterion, device):
    model.eval() 

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): 
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return average_loss, accuracy


def main():

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders()  

    model = CNNModel().to(device) 

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    num_epochs = 10

    train_losses = [] 
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    best_val_accuracy = 0.0

    os.makedirs("models", exist_ok=True) 
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_accuracy = validate(
            model,
            test_loader,
            criterion,
            device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)

        val_accuracies.append(val_accuracy)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.4f} | "
            f"Validation Loss: {val_loss:.4f} | "
            f"Validation Accuracy: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy: 
            best_val_accuracy = val_accuracy

            torch.save( 
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "validation_accuracy": val_accuracy, 
                    "class_names": [
                        "Airplane",
                        "Automobile",
                        "Bird",
                        "Cat",
                        "Deer",
                        "Dog",
                        "Frog",
                        "Horse",
                        "Ship",
                        "Truck",
                    ],
                },
                "models/best_model.pth"
            )

    print("\nTraining completed.")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print("Model saved to: models/best_model.pth")


if __name__== "__main__":
    main()
