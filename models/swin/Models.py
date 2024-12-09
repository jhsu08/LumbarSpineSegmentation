from monai.networks.nets import SwinUNETR
from tqdm import tqdm
import torch.nn as nn
import torch.optim

class Swin3DSegmentation(nn.Module):
    def __init__(self, input_channels=1, num_classes=210, img_size=(64, 64, 64), embed_dim=96, patch_size=4):
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
                              - B: Batch size
                              - C: Number of input channels
                              - D: Depth of the input image
                              - H: Height of the input image
                              - W: Width of the input image
        Returns:
            torch.Tensor: Output tensor of shape (B, num_classes, D, H, W).
                          - B: Batch size
                          - num_classes: Number of segmentation classes
                          - D: Depth of the output image
                          - H: Height of the output image
                          - W: Width of the output image
        """
        return self.swin_unetr(x)

def save_model(model, path):
    """
    Saves the model's state dictionary to the specified path.

    Args:
        model: The trained model to be saved.
        path: The file path where the model will be saved.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def train(model, train_loader, val_loader, epochs, device, save_path="trained_model.pth"):
    """
    Trains the model on the given data loaders and saves it after training.
    Args:
        model: The neural network model to be trained.
        train_loader: DataLoader for the training data.
        val_loader: DataLoader for the validation data.
        epochs: Number of epochs to train.
        device: Device for training (e.g., 'cpu', 'cuda', 'mps').
        save_path: Path to save the trained model.
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
        
    # Save the model after training
    save_model(model, save_path)