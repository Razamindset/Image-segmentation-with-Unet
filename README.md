# Image Segmentation with U-Net
A modular, clean PyTorch implementation of the classic **U-Net** architecture for semantic image segmentation, trained on the Oxford-IIIT Pet dataset for binary foreground/background pet segmentation.

---

## Architecture Overview

The U-Net architecture follows an encoder-decoder "U"-shaped structure comprising a contracting path (encoder), a bottleneck, and an expanding path (decoder), linked via long skip connections that preserve high-resolution spatial details.

```
Input Image (3 x 256 x 256)
      │
   [Enc 1] ─── (64 x 256 x 256 Skip 1) ──────────────────────────► [Dec 1] ───► Conv 1x1 ───► Output (1 x 256 x 256)
      │ MaxPool                                                       ▲
   [Enc 2] ─── (128 x 128 x 128 Skip 2) ───────────► [Dec 2] ─────────┘
      │ MaxPool                                         ▲
   [Enc 3] ─── (256 x 64 x 64 Skip 3) ──► [Dec 3] ──────┘
      │ MaxPool                             ▲
   [Enc 4] ─── (512 x 32 x 32 Skip 4) ─► [Dec 4]
      │ MaxPool                            ▲
 [Bottleneck] (1024 x 16 x 16) ────────────┘
```

### Architectural Components

The model is modularized into distinct building blocks across dedicated files:

#### 1. Double Convolution Block (`double_conv.py`)
- **Class**: `DoubleConv(in_channels, out_channels)`
- **Structure**:
  - `Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)`
  - `BatchNorm2d(out_channels)`
  - `ReLU(inplace=True)`
  - `Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)`
  - `BatchNorm2d(out_channels)`
  - `ReLU(inplace=True)`
- **Purpose**: Cascading two $3\times3$ convolutions achieves an effective receptive field of $5\times5$ with fewer parameters and richer non-linearities. Using `padding=1` keeps the spatial dimensions intact throughout each block, avoiding the cropping needed in the original 2015 U-Net paper. `bias=False` is used because Batch Normalization absorbs the bias shift.

#### 2. Contracting / Encoder Block (`encoder.py`)
- **Class**: `EncoderBlock(in_channels, out_channels)`
- **Structure**:
  - `conv = DoubleConv(in_channels, out_channels)`
  - `pool = MaxPool2d(kernel_size=2, stride=2)`
- **Outputs**:
  - `skip_connection`: Feature map after `DoubleConv` (retained for the corresponding decoder level).
  - `out`: Spatial downsampled feature map via $2\times2$ max pooling ($H/2, W/2$).

#### 3. Bottleneck (`model.py`)
- **Component**: `DoubleConv(512, 1024)`
- **Purpose**: Connects the deepest encoder stage to the first decoder stage at the lowest spatial resolution ($16\times16$ for $256\times256$ inputs) with the highest feature representation (1024 channels).

#### 4. Expanding / Decoder Block (`decoder.py`)
- **Class**: `DecoderBlock(in_channels, out_channels)`
- **Structure**:
  - `up = ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)`: Upsamples spatial dimensions $2\times$ and halves channel count.
  - `torch.concat([x, skip], dim=1)`: Concatenates upsampled features with the corresponding encoder skip connection along the channel axis. Resulting channels = `out_channels * 2`.
  - `conv = DoubleConv(out_channels * 2, out_channels)`: Fuses and refines concatenated semantic and spatial features back to `out_channels`.

#### 5. Output Head (`model.py`)
- **Component**: `Conv2d(64, out_channels=1, kernel_size=1)`
- **Purpose**: A point-wise $1\times1$ convolution that maps the final 64 feature channels down to the target class count (default 1 for binary segmentation logits).

---

## Detailed Tensor Dimension Progression

Assuming a standard batch input of shape `[B, 3, 256, 256]`:

| Stage | Sub-module | Input Shape | Output Shape | Notes |
|---|---|---|---|---|
| **Input** | - | - | `(B, 3, 256, 256)` | Normalized RGB batch |
| **Encoder 1** | `EncoderBlock(3, 64)` | `(B, 3, 256, 256)` | `out: (B, 64, 128, 128)`<br>`skip1: (B, 64, 256, 256)` | Pool reduces resolution by $2\times$ |
| **Encoder 2** | `EncoderBlock(64, 128)` | `(B, 64, 128, 128)` | `out: (B, 128, 64, 64)`<br>`skip2: (B, 128, 128, 128)` | Feature channels double |
| **Encoder 3** | `EncoderBlock(128, 256)` | `(B, 128, 64, 64)` | `out: (B, 256, 32, 32)`<br>`skip3: (B, 256, 64, 64)` | Spatial dimension: $64\times64$ |
| **Encoder 4** | `EncoderBlock(256, 512)` | `(B, 256, 32, 32)` | `out: (B, 512, 16, 16)`<br>`skip4: (B, 512, 32, 32)` | Spatial dimension: $32\times32$ |
| **Bottleneck** | `DoubleConv(512, 1024)` | `(B, 512, 16, 16)` | `(B, 1024, 16, 16)` | Deepest layer |
| **Decoder 4** | `DecoderBlock(1024, 512)` | `x: (B, 1024, 16, 16)`<br>`skip4: (B, 512, 32, 32)` | `(B, 512, 32, 32)` | Up: $32\times32$ (512 ch) + skip (512 ch) -> Cat 1024 ch -> Conv 512 ch |
| **Decoder 3** | `DecoderBlock(512, 256)` | `x: (B, 512, 32, 32)`<br>`skip3: (B, 256, 64, 64)` | `(B, 256, 64, 64)` | Up: $64\times64$ (256 ch) + skip (256 ch) -> Cat 512 ch -> Conv 256 ch |
| **Decoder 2** | `DecoderBlock(256, 128)` | `x: (B, 256, 64, 64)`<br>`skip2: (B, 128, 128, 128)` | `(B, 128, 128, 128)` | Up: $128\times128$ (128 ch) + skip (128 ch) -> Cat 256 ch -> Conv 128 ch |
| **Decoder 1** | `DecoderBlock(128, 64)` | `x: (B, 128, 128, 128)`<br>`skip1: (B, 64, 256, 256)` | `(B, 64, 256, 256)` | Up: $256\times256$ (64 ch) + skip (64 ch) -> Cat 128 ch -> Conv 64 ch |
| **Final Conv**| `Conv2d(64, 1, 1)` | `(B, 64, 256, 256)` | `(B, 1, 256, 256)` | Raw output logits |

---

## Dataset & Training Pipeline

### Dataset (`dataset.py`)
- **Dataset**: [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) (`torchvision.datasets.OxfordIIITPet`).
- **Input Resolution**: Resized to $256\times256$ pixels (Bilinear for RGB image, Nearest Neighbor for segmentation mask).
- **Mask Binarization**: Original trimap classes $\{1: \text{Pet}, 2: \text{Background}, 3: \text{Border}\}$ are converted to a binary target:
  $$\text{Mask}(i, j) = \begin{cases} 1.0 & \text{if trimap} = 1 \\ 0.0 & \text{otherwise} \end{cases}$$
- **Output Tensor Shapes**:
  - Image: `torch.FloatTensor` of shape `[3, 256, 256]` scaled to $[0.0, 1.0]$.
  - Mask: `torch.FloatTensor` of shape `[1, 256, 256]` with values in $\{0.0, 1.0\}$.

### Training Configuration (`config.py` & `train.py`)
- **Loss Function**: `nn.BCEWithLogitsLoss()` (combines Sigmoid layer and binary cross-entropy for numerical stability).
- **Optimizer**: `torch.optim.AdamW` with learning rate $\eta = 10^{-4}$.
- **Batch Size**: 16.
- **Epochs**: 5.
- **Train/Val Split**: 87% train / 13% validation.
- **Checkpointing**: Weights saved to `unet_pet_segmentation.pth` upon completion.

---

## Repository Structure

```
├── config.py          # Training hyperparameters (batch size, learning rate, epochs)
├── dataset.py         # PyTorch Dataset implementation for Oxford-IIIT Pet & mask binarization
├── double_conv.py     # Double Conv2D + BatchNorm2D + ReLU module
├── encoder.py         # Encoder block (DoubleConv + MaxPool2d)
├── decoder.py         # Decoder block (ConvTranspose2d + concat + DoubleConv)
├── model.py           # Full U-Net architecture assembly & forward pass
├── train.py           # Dataset split, training & validation loops, model checkpointing
└── README.md          # Architectural specifications and documentation
```

---

## Quick Start

### 1. Verify Architecture & Shapes
Run the forward pass test on dummy input:
```bash
python model.py
```
*Expected Output:*
```
Input shape:  torch.Size([1, 3, 256, 256])
Output shape: torch.Size([1, 1, 256, 256])
```

### 2. Verify Dataset Loader
```bash
python dataset.py
```

### 3. Train the Model
```bash
python train.py
```