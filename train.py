import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model import UNet
from dataset import PetDataset
from config import *

def train_model():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # Ensure checkpoint directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Load Dataset and perform Train/Val Split
    full_dataset = PetDataset(root=DATA_DIR, split="trainval")
    
    # Split into 80% training and 20% validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed=SEED)
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

    # Initialize Model, Loss, and Optimizer
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Checkpoint Recoverys
    start_epoch = 0
    best_val_loss = float("inf")

    if RESUME_TRAINING and os.path.exists(CHECKPOINT_PATH):

        print(f"\n[INFO] Found checkpoint at '{CHECKPOINT_PATH}'. Resuming training...")

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

        model.load_state_dict(checkpoint["model_state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"]

        best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        print(f"[INFO] Resumed from epoch {start_epoch} (Previous Best Val Loss: {best_val_loss:.4f})\n")
    else:
        print("\n[INFO] Starting fresh training from epoch 0\n")

    # Training Loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        running_train_loss = 0.0
        train_samples = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            batch_size = images.size(0)

            # Forward pass
            predictions = model(images)
            loss = criterion(predictions, masks)

            # Backward pass 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Weight loss by batch size to accurately account for the last partial batch
            running_train_loss += loss.item() * batch_size
            train_samples += batch_size

        avg_train_loss = running_train_loss / train_samples

        # Validation Loop
        model.eval()
        running_val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                batch_size = images.size(0)

                predictions = model(images)
                loss = criterion(predictions, masks)
                
                running_val_loss += loss.item() * batch_size
                val_samples += batch_size

        avg_val_loss = running_val_loss / val_samples

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save last checkpoint for crash/disconnection recovery
        checkpoint_data = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "best_val_loss": best_val_loss,
        }
        torch.save(checkpoint_data, CHECKPOINT_PATH)

        # Save best model when validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            checkpoint_data["best_val_loss"] = best_val_loss

            torch.save(checkpoint_data, BEST_MODEL_PATH)

            print(f"  ⭐ Best model updated and saved to '{BEST_MODEL_PATH}' (Val Loss: {best_val_loss:.4f})")

    # Save final model weights
    final_model_path = os.path.join(CHECKPOINT_DIR, "unet_pet_segmentation_final.pth")

    torch.save(model.state_dict(), final_model_path)

    print(f"\nTraining completed.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Final weights saved to: '{final_model_path}'")
    print(f"Best checkpoint saved to: '{BEST_MODEL_PATH}'")

if __name__ == "__main__":
    train_model()