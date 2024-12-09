import SimpleITK as sitk
import torch
import numpy as np
from torch.utils.data import Dataset

def extract_padded_patches(volume, patch_size=(64, 64, 64), stride=(32, 32, 32)):
    """
    Extract overlapping patches from a 3D volume with padding for patches that exceed the volume boundary.
    Args:
        volume (numpy array): Input 3D volume (depth, height, width).
        patch_size (tuple): Size of the patch (depth, height, width).
        stride (tuple): Stride for sliding window extraction.
    Returns:
        list: List of padded 3D patches.
    """
    depth, height, width = volume.shape
    patch_depth, patch_height, patch_width = patch_size
    stride_d, stride_h, stride_w = stride

    patches = []

    for z in range(0, depth, stride_d):
        for y in range(0, height, stride_h):
            for x in range(0, width, stride_w):
                # Extract patch and pad if necessary
                patch = volume[z:z + patch_depth, y:y + patch_height, x:x + patch_width]
                padded_patch = pad_volume(patch, patch_size)
                patches.append(padded_patch)

    return patches


class MRIDatasetWithPatches(Dataset):
    def __init__(self, image_mask_pairs, target_size=(64, 64, 64), stride=(32, 32, 32), transform=None):
        """
        Dataset class for extracting padded patches.
        Args:
            image_mask_pairs (list): List of (image_path, mask_path) pairs.
            target_size (tuple): Patch size for extraction.
            stride (tuple): Stride for overlapping patches.
            transform (callable, optional): Transformations to apply to the data.
        """
        self.image_mask_pairs = image_mask_pairs
        self.target_size = target_size
        self.stride = stride
        self.transform = transform

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        image_file, mask_file = self.image_mask_pairs[idx]

        # Load image and mask
        image = sitk.ReadImage(image_file)
        mask = sitk.ReadImage(mask_file)

        # Convert to numpy arrays
        image = sitk.GetArrayFromImage(image)
        mask = sitk.GetArrayFromImage(mask)

        # Extract padded patches
        image_patches = extract_padded_patches(image, self.target_size, self.stride)
        mask_patches = extract_padded_patches(mask, self.target_size, self.stride)

        # Select a specific patch based on idx
        patch_idx = idx % len(image_patches)  # Cycle through available patches
        img_patch = image_patches[patch_idx]
        mask_patch = mask_patches[patch_idx]

        # Apply transformations (optional)
        if self.transform:
            img_patch, mask_patch = self.transform(img_patch, mask_patch)

        # Convert to PyTorch tensors
        img_tensor = torch.tensor(img_patch, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
        mask_tensor = torch.tensor(mask_patch, dtype=torch.long)

        return img_tensor, mask_tensor


def pad_volume(volume, target_size):
    """
    Pads a 3D volume to the target size.
    Args:
        volume (numpy array): Input 3D volume.
        target_size (tuple): Target size (depth, height, width).
    Returns:
        numpy array: Padded volume.
    """
    depth, height, width = volume.shape
    target_depth, target_height, target_width = target_size

    pad_depth = max(0, target_depth - depth)
    pad_height = max(0, target_height - height)
    pad_width = max(0, target_width - width)

    padding = [
        (0, pad_depth),  # Pad at the end only
        (0, pad_height),
        (0, pad_width),
    ]

    return np.pad(volume, padding, mode="constant", constant_values=0)
    
class MRIDataset(Dataset):
    def __init__(self, image_mask_pairs, target_size=(64, 64, 64), transform=None):
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

        # Resample the imageå
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(target_spacing)
        resample.SetSize(target_size)
        resample.SetInterpolator(interpolator)
        resample.SetOutputOrigin(image.GetOrigin())
        resample.SetOutputDirection(image.GetDirection())
        resample.SetDefaultPixelValue(0)

        return resample.Execute(image)