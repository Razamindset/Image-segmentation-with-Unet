import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model import UNet
from dataset import PetDataset
from metrics import dice_score, iou_score
import matplotlib.pyplot as plt
from config import *


def plot_history(history, out_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], marker="o", label="train")
    ax1.plot(epochs, history["val_loss"], marker="o", label="val")
    ax1.set_title("Loss (BCE)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history["val_dice"], marker="o", label="Dice")
    ax2.plot(epochs, history["val_iou"], marker="o", label="IoU")
    ax2.set_title("Validation metrics")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to: '{path}'")


def train_model():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Dataset + 80/20 split
    full_dataset = PetDataset(root=DATA_DIR, split="trainval")
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True,
    )
    print(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

    # Model
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    raw_model = model
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Resume logic
    start_epoch = 0
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}

    if RESUME_TRAINING and os.path.exists(CHECKPOINT_PATH):
        print(f"\n[INFO] Found checkpoint at '{CHECKPOINT_PATH}'. Resuming training...")

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

        raw_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        history = checkpoint.get("history", history)

        print(f"[INFO] Resumed from epoch {start_epoch} (Best Val Loss: {best_val_loss:.4f})\n")
    else:
        print("\n[INFO] Starting fresh training from epoch 0\n")

    # Training loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        # train 
        model.train()
        running_train_loss = 0.0
        train_samples = 0

        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            batch_size = images.size(0)

            predictions = model(images)
            loss = criterion(predictions, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * batch_size
            train_samples += batch_size

        avg_train_loss = running_train_loss / train_samples

        # validation 
        model.eval()
        running_val_loss = 0.0
        running_val_dice = 0.0
        running_val_iou = 0.0
        val_samples = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                batch_size = images.size(0)

                predictions = model(images)
                loss = criterion(predictions, masks)

                running_val_loss += loss.item() * batch_size
                running_val_dice += dice_score(predictions, masks).item() * batch_size
                running_val_iou += iou_score(predictions, masks).item() * batch_size
                val_samples += batch_size

        avg_val_loss = running_val_loss / val_samples
        avg_val_dice = running_val_dice / val_samples
        avg_val_iou = running_val_iou / val_samples

        # record history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_dice"].append(avg_val_dice)
        history["val_iou"].append(avg_val_iou)

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Val Dice: {avg_val_dice:.4f} | Val IoU: {avg_val_iou:.4f}"
        )

        # checkpointing
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss  

        checkpoint_data = {
            "epoch": epoch + 1,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "best_val_loss": best_val_loss,
            "history": history,
        }

        if is_best:
            torch.save(checkpoint_data, BEST_MODEL_PATH)
            print(f"⭐ Best model saved to '{BEST_MODEL_PATH}' (Val Loss: {best_val_loss:.4f})")

        torch.save(checkpoint_data, CHECKPOINT_PATH)

    # Final weights
    final_model_path = os.path.join(CHECKPOINT_DIR, "unet_pet_segmentation_final.pth")
    torch.save(raw_model.state_dict(), final_model_path)

    print("\nTraining completed.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Final weights saved to: '{final_model_path}'")
    print(f"Best checkpoint saved to: '{BEST_MODEL_PATH}'")

    plot_history(history, CHECKPOINT_DIR)


if __name__ == "__main__":
    train_model()