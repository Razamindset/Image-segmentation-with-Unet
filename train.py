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

    # Load Dataset and perform Train/Val Split
    full_dataset = PetDataset(root="./data", split="trainval")
    
    # Split into 80% training and 20% validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")

    # Initialize Model, Loss, and Optimizer
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            # Forward pass
            predictions = model(images)
            loss = criterion(predictions, masks)

            # Backward pass & Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation Loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                predictions = model(images)
                loss = criterion(predictions, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save the trained model weights
    torch.save(model.state_dict(), "unet_pet_segmentation.pth")
    print("\nModel saved successfully as 'unet_pet_segmentation.pth'")

if __name__ == "__main__":
    train_model()