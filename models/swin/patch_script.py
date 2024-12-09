import MRIdataset as md
import Models as m
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.multiprocessing as mp
import os

import warnings

# Suppress specific warning message
warnings.filterwarnings(
    action='ignore',
    category=FutureWarning,
    message=".*Argument `img_size` has been deprecated.*"
)

image_files = []

# Find all the files in the extracted directory
for root, _, files in os.walk('../../extracted_images/images'):
    for file in files:
        if file.endswith('.mha'):
            image_files.append(os.path.join(root, file))

mask_files = []

# Find all mask files in the extracted directory
for root, _, files in os.walk('../../extracted_images/masks'):
    for file in files:
        if file.endswith('.mha'):
            mask_files.append(os.path.join(root, file))

image_mask_pairs = list(zip(image_files, mask_files))
for image_file, mask_file in image_mask_pairs:
    if os.path.basename(image_file) == os.path.basename(mask_file):
        continue
    else:
        print("mismatched files")

# Split image-mask pairs into training and validation
train_pairs, val_pairs = train_test_split(image_mask_pairs, test_size=0.2, random_state=55)

# Create datasets
train_dataset = md.MRIDatasetWithPatches(image_mask_pairs=train_pairs)
val_dataset = md.MRIDatasetWithPatches(image_mask_pairs=val_pairs)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # Ensure proper start method for multiprocessing
    # Check device compatibility
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize the model and move it to the appropriate device
    model = m.Swin3DSegmentation(input_channels=1, num_classes=210).to(device)
    images, masks = next(iter(train_loader))
    images = images.to(device)
    m.train(model, train_loader, val_loader, epochs=10, device=device, save_path="patch_model.pth")
