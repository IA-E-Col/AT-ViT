import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import os
from utils import load_config

def get_transforms(config):
    """Return transforms for training and testing."""
    train_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, test_transform

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, config, results_dir):
    """
    Train the model and save metrics and the final model.
    
    Args:
        model: The model to train.
        train_loader: DataLoader for training data.
        test_loader: DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        config: Configuration dictionary.
        results_dir: Directory to save results.
    
    Returns:
        str: Path to the saved model.
        dict: Training and validation metrics.
    """
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    final_model_path = os.path.join(results_dir, "final_model.pth")

    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
            images = images.to(config['device'])
            labels = labels.to(config['device']).long()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = 100 * train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels, _ in test_loader:
                images = images.to(config['device'])
                labels = labels.to(config['device']).long()
                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_loss /= len(test_loader.dataset)
        test_acc = 100 * test_correct / test_total
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        scheduler.step()

        print(f"Epoch [{epoch+1}/{config['num_epochs']}], Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {test_loss:.4f}, Val Acc: {test_acc:.2f}%")

    torch.save(model.state_dict(), final_model_path)

    metrics = {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "test_losses": test_losses,
        "test_accuracies": test_accuracies
    }
    return final_model_path, metrics