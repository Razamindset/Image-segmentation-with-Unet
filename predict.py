#! This file is AI generated
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

from model import UNet
from config import *


def find_default_weights():
    """Search common locations for trained weights."""
    candidates = [
        BEST_MODEL_PATH,
        CHECKPOINT_PATH,
        os.path.join(CHECKPOINT_DIR, "unet_pet_segmentation_final.pth"),
        "./checkpoints/best_model.pth",
        "./checkpoints/last_checkpoint.pth",
        "./checkpoints/unet_pet_segmentation_final.pth",
        "./unet_pet_segmentation.pth",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def load_model(weights_path, device):
    """Instantiate UNet and load trained weights."""
    model = UNet(in_channels=3, out_channels=1).to(device)

    print(f"Loading weights from: '{weights_path}'")
    checkpoint = torch.load(weights_path, map_location=device)

    # Support both full checkpoint dict and direct state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "unknown")
        val_loss = checkpoint.get("val_loss", None)
        loss_info = f" (Val Loss: {val_loss:.4f})" if val_loss is not None else ""
        print(f"Loaded checkpoint saved at epoch {epoch}{loss_info}")
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
        print("Loaded model state_dict successfully.")
    else:
        raise ValueError(f"Unrecognized checkpoint format at {weights_path}")

    model.eval()
    return model


def preprocess_image(image_path, target_size=(256, 256)):
    """Open and preprocess a single image file for inference."""
    raw_img = Image.open(image_path).convert("RGB")
    resized_img = raw_img.resize(target_size, Image.BILINEAR)
    tensor_img = T.ToTensor()(resized_img).unsqueeze(0)  # [1, 3, H, W]
    return raw_img, resized_img, tensor_img


def predict(model, tensor_img, device, threshold=0.5):
    """Run model forward pass and return probability map and binary mask."""
    tensor_img = tensor_img.to(device)

    with torch.no_grad():
        logits = model(tensor_img)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()  # [H, W]

    binary_mask = (probs >= threshold).astype(np.float32)
    return probs, binary_mask


def create_overlay(resized_img, binary_mask, alpha=0.45, color=(1.0, 0.0, 0.0)):
    """Create a colored mask overlay on top of the original resized image."""
    img_np = np.array(resized_img).astype(np.float32) / 255.0
    overlay = img_np.copy()

    # Blend foreground mask with specified color
    fg = binary_mask == 1.0
    for c in range(3):
        overlay[fg, c] = (1 - alpha) * img_np[fg, c] + alpha * color[c]

    return np.clip(overlay, 0.0, 1.0)


def visualize_and_save(resized_img, probs, binary_mask, overlay, output_path, gt_mask=None):
    """Save multi-panel comparison figure."""
    num_cols = 5 if gt_mask is not None else 4
    fig, axes = plt.subplots(1, num_cols, figsize=(4 * num_cols, 4))

    # 1. Original Image
    axes[0].imshow(resized_img)
    axes[0].set_title("Input Image (256x256)")
    axes[0].axis("off")

    # 2. Ground Truth Mask (if available)
    col_idx = 1
    if gt_mask is not None:
        axes[col_idx].imshow(gt_mask, cmap="gray")
        axes[col_idx].set_title("Ground Truth Mask")
        axes[col_idx].axis("off")
        col_idx += 1

    # 3. Probability Map
    prob_im = axes[col_idx].imshow(probs, cmap="magma", vmin=0.0, vmax=1.0)
    axes[col_idx].set_title("Predicted Probabilities")
    axes[col_idx].axis("off")
    fig.colorbar(prob_im, ax=axes[col_idx], fraction=0.046, pad=0.04)
    col_idx += 1

    # 4. Binary Mask
    axes[col_idx].imshow(binary_mask, cmap="gray")
    axes[col_idx].set_title("Binary Mask (Thresh >= 0.5)")
    axes[col_idx].axis("off")
    col_idx += 1

    # 5. Colored Overlay
    axes[col_idx].imshow(overlay)
    axes[col_idx].set_title("Segmentation Overlay")
    axes[col_idx].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Prediction saved to: '{output_path}'")


def main():
    parser = argparse.ArgumentParser(description="Inference script for U-Net pet segmentation.")
    parser.add_argument("--image", type=str, default=None, help="Path to input image file.")
    parser.add_argument("--weights", type=str, default=None, help="Path to trained model weights / checkpoint.")
    parser.add_argument("--output", type=str, default="prediction_result.png", help="Path to save output visualization.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization probability threshold.")
    parser.add_argument("--sample_index", type=int, default=0, help="Dataset index to sample if --image is not provided.")
    parser.add_argument("--device", type=str, default=None, help="Force device ('cuda' or 'cpu').")
    args = parser.parse_args()

    # Determine Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine Weights Path
    weights_path = args.weights or find_default_weights()
    if weights_path is None or not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"No weights file found. Please train the model first with 'python train.py' "
            f"or specify an explicit weights path using '--weights <path>'."
        )

    # Load Model
    model = load_model(weights_path, device)

    gt_mask = None

    # Case 1: An explicit image path is provided
    if args.image and os.path.exists(args.image):
        print(f"Inferring on input image: '{args.image}'")
        raw_img, resized_img, tensor_img = preprocess_image(args.image)

    # Case 2: No image provided -> load a sample from PetDataset
    else:
        if args.image:
            print(f"[WARNING] Image path '{args.image}' not found. Falling back to PetDataset sample.")
        else:
            print("No --image specified. Loading sample from dataset...")

        from dataset import PetDataset
        dataset = PetDataset(root=DATA_DIR, split="trainval")
        tensor_img, mask_tensor = dataset[args.sample_index]

        # Invert normalized tensor [3, H, W] to PIL Image for display
        img_np = (tensor_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        resized_img = Image.fromarray(img_np)
        tensor_img = tensor_img.unsqueeze(0)  # [1, 3, H, W]
        gt_mask = mask_tensor.squeeze().numpy()

    # Run inference
    probs, binary_mask = predict(model, tensor_img, device, threshold=args.threshold)

    # Create overlay
    overlay = create_overlay(resized_img, binary_mask, alpha=0.45, color=(1.0, 0.0, 0.0))

    # Save visual comparison
    visualize_and_save(resized_img, probs, binary_mask, overlay, args.output, gt_mask=gt_mask)


if __name__ == "__main__":
    main()
