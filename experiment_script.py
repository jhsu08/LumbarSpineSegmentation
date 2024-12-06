import SimpleITK as sitk
import numpy as np
import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from monai.networks.nets import SwinUNETR
from tqdm import tqdm
import os

image_files = []

# Find all the files in the extracted directory
for root, _, files in os.walk('extracted_images/images'):
    for file in files:
        if file.endswith('.mha'):
            image_files.append(os.path.join(root, file))

mask_files = []

# Find all mask files in the extracted directory
for root, _, files in os.walk('extracted_images/masks'):
    for file in files:
        if file.endswith('.mha'):  # Assuming masks have "_mask" in their filename
            mask_files.append(os.path.join(root, file))

image_mask_pairs = list(zip(image_files, mask_files))
for image_file, mask_file in image_mask_pairs:
    if os.path.basename(image_file) == os.path.basename(mask_file):
        continue
    else:
        print("mismatched files")

class MRIDataset(Dataset):
    def __init__(self, image_mask_pairs, target_size=(128, 128, 128), transform=None):
        """
        Initializes the MRIDataset.

        Args:
            image_mask_pairs (list of tuples): List of (image_path, mask_path) pairs.
            target_size (tuple): Desired size for the resampled images and masks (depth, height, width).
            transform (callable, optional): Transformations to apply to the data.
        """
        self.image_mask_pairs = image_mask_pairs
        self.target_size = target_size
        self.transform = transform

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        image_file, mask_file = self.image_mask_pairs[idx]

        # Load image and mask
        image = sitk.ReadImage(image_file)
        mask = sitk.ReadImage(mask_file)

        # Resample image and mask
        image = self.resample_image(image, self.target_size, sitk.sitkLinear)
        mask = self.resample_image(mask, self.target_size, sitk.sitkNearestNeighbor)

        # Convert to numpy arrays
        image = sitk.GetArrayFromImage(image)
        mask = sitk.GetArrayFromImage(mask)

        # Normalize the image
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        mask = torch.tensor(mask, dtype=torch.long)  # Long type for classification tasks

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

    @staticmethod
    def resample_image(image, target_size, interpolator):
        """
        Resamples a 3D image to the target size.

        Args:
            image (SimpleITK.Image): The input image.
            target_size (tuple): Desired size (depth, height, width).
            interpolator (SimpleITK.Interpolator): Interpolation method (e.g., sitk.sitkLinear or sitk.sitkNearestNeighbor).

        Returns:
            SimpleITK.Image: Resampled image.
        """
        original_size = image.GetSize()
        original_spacing = image.GetSpacing()

        # Calculate target spacing
        target_spacing = [
            (original_size[i] * original_spacing[i]) / target_size[i]
            for i in range(3)
        ]

        # Resample the image
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(target_spacing)
        resample.SetSize(target_size)
        resample.SetInterpolator(interpolator)
        resample.SetOutputOrigin(image.GetOrigin())
        resample.SetOutputDirection(image.GetDirection())
        resample.SetDefaultPixelValue(0)

        return resample.Execute(image)

dataset = MRIDataset(image_mask_pairs=image_mask_pairs)

class Swin3DSegmentation(nn.Module):
    def __init__(self, input_channels=1, num_classes=3, img_size=(128, 128, 128), embed_dim=96, patch_size=4):
        """
        3D segmentation model using SwinUNETR backbone.
        Args:
            input_channels (int): Number of input channels (e.g., 1 for grayscale MRI images).
            num_classes (int): Number of output classes for segmentation.
            img_size (tuple): Size of the 3D input image (D, H, W).
            embed_dim (int): Embedding dimension of the Swin Transformer.
            patch_size (int): Patch size for splitting the input volume.
        """
        super(Swin3DSegmentation, self).__init__()
        
        # SwinUNETR backbone
        self.swin_unetr = SwinUNETR(
            img_size=img_size,  # 3D image size
            in_channels=input_channels,
            out_channels=num_classes,
            feature_size=embed_dim,
            use_checkpoint=True  # Enables gradient checkpointing for memory efficiency
        )

    def forward(self, x):
        """
        Forward pass for 3D segmentation.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, H, W).
        Returns:
            torch.Tensor: Output tensor of shape (B, num_classes, D, H, W).
        """
        return self.swin_unetr(x)

class DiceLoss(nn.Module):
    def forward(self, preds, targets):
        smooth = 1e-5
        intersection = torch.sum(preds * targets)
        union = torch.sum(preds) + torch.sum(targets)
        return 1 - (2.0 * intersection + smooth) / (union + smooth)

# Split image-mask pairs into training and validation
train_pairs, val_pairs = train_test_split(image_mask_pairs, test_size=0.2, random_state=55)

# Create datasets
train_dataset = MRIDataset(image_mask_pairs=train_pairs)
val_dataset = MRIDataset(image_mask_pairs=val_pairs)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# Check device compatibility
# device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Initialize the model and move it to the appropriate device
model = Swin3DSegmentation(input_channels=1, num_classes=3).to(device)

# Define the training function with progress tracking
def train(model, train_loader, val_loader, epochs, device):
    """
    Trains the model on the given data loaders.
    Args:
        model: The neural network model to be trained.
        train_loader: DataLoader for the training data.
        val_loader: DataLoader for the validation data.
        epochs: Number of epochs to train.
        device: Device for training (e.g., 'cpu', 'cuda', 'mps').
    """
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        epoch_loss = 0.0
        print(f"Epoch {epoch + 1}/{epochs}")

        # Training loop with progress bar
        for images, masks in tqdm(train_loader, desc="Training", leave=False):
            images, masks = images.to(device), masks.to(device)  # Move data to device

            optimizer.zero_grad()  # Zero the gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, masks)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Training Loss: {avg_train_loss:.4f}")

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # Disable gradient calculations for validation
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

# Call the train function
train(model, train_loader, val_loader, epochs=10, device=device)

def dice_coefficient(preds, targets):
    smooth = 1e-5
    intersection = torch.sum(preds * targets)
    union = torch.sum(preds) + torch.sum(targets)
    return (2.0 * intersection + smooth) / (union + smooth)