import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
from PIL import ImageFile
from dataset import PlantTraitDataset
from model import CrossViT
from train import train_model, get_transforms
from test import evaluate_model, evaluate_noisy_datasets
from visualize import visualize_small_branch_attention, summarize_attention_patterns, generate_gradcam_small_branch, run_attention_visualization
from utils import load_config, setup_environment

def main():
    # Set up environment
    setup_environment(seed=42)

    # Allow loading of truncated images
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # Load configuration
    config = load_config()
    print(f"Using device: {config['device']}")

    # Load the CSV
    df = pd.read_csv(config['csv_path'])

    # Verify the dataset
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("train_test_set value counts:\n", df["train_test_set"].value_counts())
    print(f"{config['target_variable']} distribution in each set:")
    print("Train:\n", df[df["train_test_set"] == "train"][config['target_variable']].value_counts(normalize=True))
    print("Test:\n", df[df["train_test_set"] == "test"][config['target_variable']].value_counts(normalize=True))

    # Create train and test DataFrames
    train_df = df[df["train_test_set"] == "train"].copy()
    test_df = df[df["train_test_set"] == "test"].copy()

    print("\nTrain size:", len(train_df))
    print("Test size:", len(test_df))

    # Get transforms
    train_transform, test_transform = get_transforms(config)

    # Create datasets
    train_dataset = PlantTraitDataset(train_df, config['original_img_dir'], transform=train_transform, subset='train')
    test_dataset = PlantTraitDataset(test_df, config['original_img_dir'], transform=test_transform, subset='test')

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Print dataset sizes
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")

    # Initialize model
    num_classes = len(train_dataset.classes)
    print(f"Training model for {num_classes} classes: {train_dataset.classes}")
    model = CrossViT(num_classes=num_classes).to(config['device'])
    print(f"Model architecture:\n{model}")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Train the model
    final_model_path, metrics = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, config, config['results_dir'])
    print(f"Final model saved to: {final_model_path}")

    # Evaluate on test dataset
    test_metrics, confusion_matrix_data, predictions_df = evaluate_model(model, test_loader, criterion, config)
    print(f"Test Loss: {test_metrics['test_loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F1 Score: {test_metrics['f1_score']:.4f}")

    # Evaluate on noisy datasets
    evaluate_noisy_datasets(model, test_df, criterion, config)

    # Generate visualizations
    attention_save_dir = os.path.join(config['results_dir'], "attention_visualizations")
    print("Creating CrossViT attention visualizations and summary...")
    run_attention_visualization(
        model=model,
        test_loader=test_loader,
        num_samples=453,
        save_dir=attention_save_dir,
        layer_idx=-1
    )

    print("Creating CrossViT attention visualizations with IoU for small branch...")
    visualize_small_branch_attention(
        model=model,
        test_loader=test_loader,
        num_samples=453,
        save_dir=os.path.join(config['results_dir'], "crossvit_iou_attention_visualizations"),
        layer_idx=-1,
        segmented_img_dir=config['segmented_img_dir']
    )

    print("Generating Grad-CAM visualizations for small branch...")
    gradcam_dir = generate_gradcam_small_branch(
        model=model,
        test_loader=test_loader,
        results_dir=config['results_dir'],
        device=config['device'],
        num_images=453,
        output_size=(240, 240)
    )
    print(f"Grad-CAM visualizations saved to: {gradcam_dir}")

if __name__ == "__main__":
    main()