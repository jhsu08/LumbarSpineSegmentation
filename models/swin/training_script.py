import os
import shutil
import tempfile
import multiprocessing as mp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.data import (
    ImageDataset,
)

from monai.transforms import Compose, EnsureChannelFirst, RandFlip, RandRotate, ScaleIntensity, Spacing, DivisiblePad, Activations, AsDiscrete

from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.inferers import sliding_window_inference
from tqdm import tqdm
from monai.transforms import MapTransform

class LabelMapper:
    def __init__(self, label_mapping):
        """
        A simplified transform to map labels based on the provided dictionary.
        Args:
            label_mapping (dict): Dictionary for label mapping.
        """
        self.label_mapping = label_mapping

    def __call__(self, mask):
        """
        Args:
            mask (torch.Tensor or np.ndarray): Input mask to be mapped.
        Returns:
            torch.Tensor or np.ndarray: Mask with labels mapped.
        """
        # Apply mapping to the mask
        return mask.apply_(lambda x: self.label_mapping.get(x, 0))  # Default to 0 for unknown labels

def train_epoch(model, train_loader, optimizer, loss_function, scaler, device, roi_size, sw_batch_size, overlap=0.25):
    """
    Perform one epoch of training with sliding window inference for validation.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        loss_function (torch.nn.Module): Loss function to optimize.
        scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision training.
        device (torch.device): The device to train on.
        roi_size (tuple): The region of interest size for sliding window inference.
        sw_batch_size (int): The number of windows to process in parallel during sliding inference.
        overlap (float): The fraction of overlap between patches in sliding window inference.

    Returns:
        train_loss (float): Average training loss.
    """
    # Set the model to training mode
    model.train()
    train_loss = 0.0
    
    for pairs in tqdm(train_loader, desc="Training"):
    
        # Load the data and move to device
        inputs, labels = pairs[0].to(device), pairs[1].to(device)

        # Forward pass with mixed precision
        with torch.amp.autocast('cuda'):
            # Perform sliding window inference
            outputs = sliding_window_inference(
                inputs,                # Input image
                roi_size=roi_size,     # Size of each sliding window (e.g., [64, 64, 64])
                sw_batch_size=sw_batch_size,  # Number of windows processed at once
                predictor=model,       # Model for inference
                overlap=overlap        # Amount of overlap between patches
            )
            loss = loss_function(outputs, labels)

        # Backpropagation and optimizer step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()

    # Average training loss
    train_loss /= len(train_loader)

    print("Training Loss:", train_loss)

    return train_loss


def validate_epoch(model, val_loader, loss_function, dice_metric, device, roi_size, sw_batch_size, overlap=0.25):
    """
    Perform validation with sliding window inference.

    Args:
        model (torch.nn.Module): The model to validate.
        val_loader (DataLoader): DataLoader for the validation dataset.
        loss_function (torch.nn.Module): Loss function to evaluate.
        device (torch.device): The device to run validation on.
        roi_size (tuple): The region of interest size for sliding window inference.
        sw_batch_size (int): The number of windows to process in parallel during sliding inference.
        overlap (float): The fraction of overlap between patches in sliding window inference.

    Returns:
        val_loss (float): Average validation loss.
        val_dice (float): Mean Dice score on the validation set.
    """
    # Set the model to evaluation mode
    model.eval()
    val_loss = 0.0
    
    # Postprocessing transforms
    post_transforms = Compose([
        Activations(softmax=True),  # Convert logits to probabilities
        AsDiscrete(argmax=True, to_onhot=20)    # Get class predictions
    ])

    with torch.no_grad():  # No gradients needed for validation
        for val_data in tqdm(val_loader, desc="Validating"):
            # Load the data and move to device
            val_inputs = val_data[0].to(device)
            val_labels = val_data[1].to(device)

            # Perform sliding window inference
            val_outputs = sliding_window_inference(
                inputs=val_inputs,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap
            )

            # Postprocess outputs
            val_outputs_post = post_transforms(val_outputs.squeeze(0))

            # Compute loss
            loss = loss_function(val_outputs, val_labels)
            val_loss += loss.item()

            # Compute Dice metric
            dice_metric(val_outputs_post, val_labels)

    # Average validation loss
    val_loss /= len(val_loader)

    # Mean Dice score
    val_dice = dice_metric.aggregate().item()
    dice_metric.reset()  # Reset metric for the next epoch

    print("Validation Loss:", val_loss)
    print("Validation Dice:", val_dice)
    return val_loss, val_dice

def train_model(
    model, train_loader, val_loader, loss_function, optimizer, scaler, roi_size, sw_batch_size, overlap, dice_metric,
    num_epochs, val_interval, patience, device, save_dir="model_checkpoints"
):
    """
    Train model with early stopping and save metrics.

    Args:
        model: PyTorch model.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        loss_function: Loss function.
        optimizer: Optimizer.
        scaler: Gradient scaler for mixed precision training.
        roi_size: ROI size for sliding window inference.
        sw_batch_size: Sliding window batch size.
        overlap: Overlap for sliding window inference.
        dice_metric: MONAI DiceMetric.
        num_epochs: Total number of epochs.
        val_interval: Number of epochs between validations.
        patience: Number of epochs to wait for improvement before stopping.
        device: Device to run training on.
        save_dir: Directory to save checkpoints and metrics.
    """
    # Create directory to save checkpoints
    os.makedirs(save_dir, exist_ok=True)

    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    val_dice_scores = []

    # Early stopping parameters
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training step
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            scaler=scaler,
            device=device,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
        )
        train_losses.append(train_loss)

        # Validation step
        if (epoch + 1) % val_interval == 0:
            val_loss, val_dice = validate_epoch(
                model=model,
                val_loader=val_loader,
                loss_function=loss_function,
                dice_metric=dice_metric,
                device=device,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                overlap=overlap,
            )
            val_losses.append(val_loss)
            val_dice_scores.append(val_dice)

            print(f"Validation Loss: {val_loss:.4f}, Validation Dice: {val_dice:.4f}")

            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                # Save the best model checkpoint
                checkpoint_path = os.path.join(save_dir, f"best_model.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Best model saved to {checkpoint_path}")
            else:
                epochs_without_improvement += 1
                print(f"Validation loss did not improve for {epochs_without_improvement}/{patience} epochs")

            # Stop training if patience is exceeded
            if epochs_without_improvement >= patience:
                print("Early stopping triggered. Stopping training.")
                break

        # Save model checkpoint for each epoch
        checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

    # Save metrics to an .npz file
    metrics_file = os.path.join(save_dir, "training_metrics.npz")
    np.savez(metrics_file,
             train_losses=train_losses,
             val_losses=val_losses,
             val_dice_scores=val_dice_scores)
    print(f"Metrics saved to {metrics_file}")

    return train_losses, val_losses, val_dice_scores


if __name__ == "__main__":
    image_files = []
    mask_files = []

    # Base directories for images and masks
    image_base_dir = "../../../data/images"
    mask_base_dir = "../../../data/masks"

    # Find all the files in the extracted directory
    for root, _, files in os.walk(image_base_dir):
        for file in files:
            if file.endswith('.mha'):
                image_files.append(os.path.join(root, file))

    # Find all mask files in the extracted directory
    for root, _, files in os.walk(mask_base_dir):
        for file in files:
            if file.endswith('.mha'):
                mask_files.append(os.path.join(root, file))

    image_mask_pairs = list(zip(image_files, mask_files))
    for image_file, mask_file in image_mask_pairs:
        if os.path.basename(image_file) == os.path.basename(mask_file):
            continue
        else:
            print("mismatched files")

    # Update the paths in image_mask_pairs
    image_mask_pairs = [
        (os.path.join(image_base_dir, os.path.basename(pair[0])),
        os.path.join(mask_base_dir, os.path.basename(pair[1])))
        for pair in image_mask_pairs
    ]

    # Define patch size and stride
    patch_size = (64, 64, 64)
    stride = (32, 32, 32)
    spacing = (2.0, 2.0, 2.0) #isotropic spacing for downsampling
    roi_size = (64, 64, 64)
    sw_batch_size = 20
    overlap = 0.5
    
    label_mapping = {
        0: 0,  # Background
        1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9,  # Vertebrae
        100: 10,  # Spinal Canal
        201: 11, 202: 12, 203: 13, 204: 14, 205: 15, 206: 16, 207: 17, 208: 18, 209: 19  # IVDs
    }
    num_classes = 20
    
    # Initialize model, loss, optimizer, scaler, and metrics
    torch.backends.cudnn.benchmark = True

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # Training parameters
    num_epochs = 50
    val_interval = 2

    # K-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    
    for train_idx, val_idx in kf.split(image_mask_pairs):
        print(f"Processing Fold {fold}...")
    
        # Split into train and validation datasets
        train_pairs = [image_mask_pairs[i] for i in train_idx]
        val_pairs = [image_mask_pairs[i] for i in val_idx]
    
        train_images = [image for image, _ in train_pairs]
        train_masks = [mask for _, mask in train_pairs]
        val_images = [image for image, _ in val_pairs]
        val_masks = [mask for _, mask in val_pairs]

        # Image transformations
        img_transforms = Compose([
            EnsureChannelFirst(),  # Ensure image is in channel-first format
            ScaleIntensity(minv=0.0, maxv=1.0),  # Normalize intensity to [0, 1]
            Spacing(pixdim=spacing, mode="bilinear"),  # Resample to target spacing
            RandFlip(spatial_axis=0, prob=0.5),  # Randomly flip along axis 0
            RandRotate(range_x=0.2, range_y=0.2, range_z=0.2, prob=0.5),  # Randomly rotate
            DivisiblePad(roi_size[0], mode="constant")
        ])
        
        mask_transforms = Compose([
            EnsureChannelFirst(),  # Ensure mask is in channel-first format
            LabelMapper(label_mapping=label_mapping),
            Spacing(pixdim=spacing, mode="nearest"),  # Resample using nearest neighbor
            RandFlip(spatial_axis=0, prob=0.5),  # Randomly flip
            RandRotate(range_x=0.2, range_y=0.2, range_z=0.2, prob=0.5),  # Randomly rotate
            DivisiblePad(roi_size[0], mode="constant")
        ])
    
        train_ds = ImageDataset(
            image_files=train_images,
            seg_files=train_masks,
            transform=img_transforms,
            seg_transform=mask_transforms,
            image_only=False,
            transform_with_metadata=False,
        )
        val_ds = ImageDataset(
            image_files=val_images,
            seg_files=val_masks,
            transform=img_transforms,
            seg_transform=mask_transforms,
            image_only=False,
            transform_with_metadata=False,
        )
    
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=10)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=10)

        # Initialize a new model, optimizer, and scaler for each fold
        model = SwinUNETR(
            img_size=patch_size,
            in_channels=1,
            out_channels=num_classes,
            feature_size=48,
            use_checkpoint=True,
        ).to(device)
    
        model.load_from(weights=torch.load("./model_swinvit.pt", map_location=device))
        # state_dict = torch.load("model_checkpoints_dice_loss/2/fold_1/model_epoch_25.pth", map_location=device, weights_only=True)
        # model.load_state_dict(state_dict)
        
        loss_function = DiceLoss(to_onehot_y=True, softmax=True, include_background=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scaler = torch.amp.GradScaler('cuda')
        dice_metric = DiceMetric(include_background=True, reduction="mean", num_classes=-1)
        patience = 3
        
        # Train the model
        train_losses, val_losses, val_dice_scores = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            scaler=scaler,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            dice_metric=dice_metric,
            num_epochs=num_epochs,
            val_interval=val_interval,
            patience=patience,
            device=device,
            save_dir=f"model_checkpoints/fold_{fold}"
        )
    
        
        fold += 1