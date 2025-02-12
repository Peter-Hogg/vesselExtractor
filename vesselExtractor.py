import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import napari

# Importing the necessary libraries for augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tifffile import imread
import torch.nn.functional as F
from tqdm.notebook import trange, tqdm

# Define the same validation transformations
def get_val_transform():
    return A.Compose([
        A.Resize(128, 128),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # RGB means/stds
        ToTensorV2()
    ])
# Define the batch inference function

# Function to load a trained model
def load_model(model_path, device, in_channels=1, classes=1):
    # Rebuild the U-Net model architecture
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",  # Choose encoder
        encoder_depth=5,
        encoder_weights="imagenet",
        in_channels=3,  # RGB input
        classes=1,  # Binary segmentation (single output channel for masks)
        decoder_channels=(128, 128, 32, 32, 16)
    )

    # Load the trained weights into the model
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # Move the model to GPU if available
    model = model.to(device)
    
    print(f"Model loaded from {model_path}")
    
    return model



def batch_inference(model, device, crops, batch_size=16):
    
    model.eval()
    results = []
    with torch.no_grad():
        for i in range(0, len(crops), batch_size):
            batch = torch.stack(crops[i:i + batch_size]).to(device)
            preds = model(batch)
            results.append(preds.cpu())
    return torch.cat(results, dim=0)

def sliding_window_inference_2d(model, image, device, window_size=128, stride=64, batch_size=16, threshold=0.5):
    """
    Perform sliding window inference on a 2D image with single-channel (grayscale) or RGB images.

    :param model: Trained PyTorch model
    :param image: 2D NumPy array (H, W) for grayscale or (H, W, C) for RGB
    :param window_size: Size of the sliding window (assumes square window)
    :param stride: Stride of the sliding window (how far to move the window after each step)
    :param batch_size: Number of windows to process in parallel
    :param threshold: Threshold for converting probability maps to binary masks
    :return: 2D NumPy array of predicted mask (H, W)
    """
    model.eval()  # Set model to evaluation mode
    height, width = image.shape[:2]  # Get dimensions of the 2D image
    predicted_mask = np.zeros((height, width), dtype=np.float32)
    count_mask = np.zeros((height, width), dtype=np.float32)

    # Initialize the validation transform
    transform = get_val_transform()

    # Collect windows and positions for batch inference
    windows = []
    positions = []

    # Generate sliding windows
    for y in range(0, height - window_size + 1, stride):
        for x in range(0, width - window_size + 1, stride):
            # Extract the window
            window = image[y:y+window_size, x:x+window_size]
            
            # Expand dimensions for grayscale images (H, W -> H, W, 1)
            if len(window.shape) == 2:
                window = np.expand_dims(window, axis=-1)

            # Apply the validation transform
            transformed = transform(image=window)
            transformed_window = transformed['image']  # Transformed tensor
            
            windows.append(transformed_window)  # Append transformed window
            positions.append((y, x))  # Store top-left position of the window

    # Perform batch inference on all collected windows
    predictions = batch_inference(model, device, windows, batch_size=batch_size)

    # Place the predictions back into the full-size mask
    for (y, x), pred_window in zip(positions, predictions):
        pred_window = torch.sigmoid(pred_window).squeeze().cpu().numpy()  # Apply sigmoid

        # Accumulate predictions and update count mask
        predicted_mask[y:y+window_size, x:x+window_size] += pred_window
        count_mask[y:y+window_size, x:x+window_size] += 1

    # Normalize the predicted mask by the count mask to handle overlaps
    predicted_mask /= np.maximum(count_mask, 1)
    predicted_mask = (predicted_mask > threshold).astype(np.int8)  # Binarize the output

    return predicted_mask

