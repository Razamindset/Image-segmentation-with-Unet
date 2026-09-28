import os

# Environment & Directory Configuration

CHECKPOINT_DIR = "/kaggle/working/checkpoints"
DATA_DIR = "/kaggle/working/data"

# Checkpoint Settings
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
RESUME_TRAINING = True  # Automatically resume if last_checkpoint.pth exists

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 5
SEED=69