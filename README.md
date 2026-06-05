# 🧠 Brain Tumor Segmentation using U-Net

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Colab](https://img.shields.io/badge/Google%20Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

**A deep learning solution for automated brain tumor detection and segmentation using semantic segmentation with U-Net architecture**

[📖 Overview](#-overview) •
[🏗️ Architecture](#️-model-architecture) •
[📊 Results](#-results) •
[🚀 Quick Start](#-quick-start) •
[📁 Dataset](#-dataset-structure)

---

</div>

## 📖 Overview

This project implements a **U-Net convolutional neural network** for binary semantic segmentation of brain tumors from MRI scans. The model accurately identifies and segments tumor regions, providing pixel-level precision that is crucial for medical diagnosis and treatment planning.

### ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🎯 **Binary Segmentation** | Precise tumor vs. healthy tissue classification |
| 📈 **IoU Metric** | Intersection over Union for accurate evaluation |
| 🔄 **Data Pipeline** | Efficient TensorFlow data loading with prefetching |
| 🎛️ **Dropout Regularization** | Prevents overfitting with 30% dropout |
| ⚡ **GPU Accelerated** | Optimized for NVIDIA T4 GPU on Google Colab |

---

## 🏗️ Model Architecture

The implementation follows the classic **U-Net encoder-decoder architecture** with skip connections, specifically designed for biomedical image segmentation.

```
                           U-Net Architecture
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │   INPUT (128×128×3)                    OUTPUT (128×128×1)   │
    │         │                                      ▲            │
    │         ▼                                      │            │
    │   ┌─────────┐                            ┌─────────┐        │
    │   │ Conv 64 │ ─────────────────────────► │ Conv 64 │        │
    │   └────┬────┘      Skip Connection       └────▲────┘        │
    │        │ MaxPool                              │ UpSample    │
    │        ▼                                      │             │
    │   ┌──────────┐                          ┌──────────┐        │
    │   │ Conv 128 │ ────────────────────────►│ Conv 128 │        │
    │   └────┬─────┘     Skip Connection      └────▲─────┘        │
    │        │ MaxPool                              │ UpSample    │
    │        ▼                                      │             │
    │   ┌──────────┐                          ┌──────────┐        │
    │   │ Conv 256 │ ────────────────────────►│ Conv 256 │        │
    │   └────┬─────┘     Skip Connection      └────▲─────┘        │
    │        │ MaxPool                              │ UpSample    │
    │        ▼                                      │             │
    │   ┌──────────┐                          ┌──────────┐        │
    │   │ Conv 512 │ ────────────────────────►│ Conv 512 │        │
    │   └────┬─────┘     Skip Connection      └────▲─────┘        │
    │        │ MaxPool                              │ UpSample    │
    │        ▼                                      │             │
    │   ┌────────────────────────────────────────────┐            │
    │   │            BOTTLENECK (Conv 1024)          │            │
    │   └────────────────────────────────────────────┘            │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
```

### 🔧 Architecture Details

| Component | Configuration |
|-----------|--------------|
| **Input Shape** | 128 × 128 × 3 (RGB) |
| **Encoder Blocks** | 4 (64 → 128 → 256 → 512 filters) |
| **Bottleneck** | 1024 filters |
| **Decoder Blocks** | 4 (512 → 256 → 128 → 64 filters) |
| **Convolution** | 3×3 kernels, 'same' padding, ReLU activation |
| **Pooling** | 2×2 MaxPooling |
| **Upsampling** | Conv2DTranspose (3×3, stride 2) |
| **Output** | Sigmoid activation (binary mask) |
| **Total Parameters** | ~31 Million |

---

## 📊 Results

### Training Performance (15 Epochs)

<table>
<tr>
<td>

| Metric | Training | Validation |
|--------|----------|------------|
| **Accuracy** | 99.36% | 99.10% |
| **IoU Score** | 0.607 | 0.564 |
| **Loss** | 0.018 | 0.029 |

</td>
<td>

```
Training Progress:
━━━━━━━━━━━━━━━━━ 99.36% Accuracy
━━━━━━━━━━━━━━━━━ 60.7% IoU
━━━━━━━━━━━━━━━━━ 0.018 Loss
```

</td>
</tr>
</table>

### 📈 Training Curves

The model shows consistent improvement across all metrics with minimal overfitting:

- **IoU**: Progressive improvement from 0.006 → 0.607
- **Accuracy**: Stable high accuracy from 96.32% → 99.36%
- **Loss**: Smooth convergence from 0.438 → 0.018

---

## 🎯 Why IoU over Accuracy?

> **Critical Insight**: In medical imaging, accuracy can be misleading!

```
┌──────────────────────────────────────────────────────────────────┐
│                     THE CLASS IMBALANCE PROBLEM                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Brain MRI Image:    ████████████████████████████████████████   │
│   (98% Background)    ████████████████████████████████████████   │
│                       ██████████████████████  ← 2% Tumor        │
│                                                                  │
│   ❌ A model predicting "all background" = 98% Accuracy          │
│   ✅ IoU catches this: IoU = 0 (no tumor overlap!)               │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### IoU Formula

$$IoU = \frac{|P \cap G|}{|P \cup G|} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives} + \text{False Negatives}}$$

Where:
- **P** = Predicted mask
- **G** = Ground truth mask

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install tensorflow numpy matplotlib
```

### Option 1: Run on Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

1. Upload the notebook to Google Colab
2. Enable GPU: `Runtime → Change runtime type → T4 GPU`
3. Mount Google Drive and update data paths
4. Run all cells!

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/ZeyadArafa/Brain-Tumor-Segmentation-UNet.git
cd Brain-Tumor-Segmentation-UNet

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook UNET_Brain_Tumor_Quiz.ipynb
```

---

## 📁 Dataset Structure

```
BrainTumor/
├── images/
│   ├── brain_001.png
│   ├── brain_002.png
│   ├── brain_003.png
│   └── ... (3064 total images)
│
└── masks/
    ├── brain_001_mask.png
    ├── brain_002_mask.png
    ├── brain_003_mask.png
    └── ... (3064 corresponding masks)
```

### Dataset Statistics

| Property | Value |
|----------|-------|
| **Total Samples** | 3,064 |
| **Training Set** | 2,452 (80%) |
| **Validation Set** | 612 (20%) |
| **Image Size** | 128 × 128 pixels |
| **Image Channels** | RGB (3 channels) |
| **Mask Type** | Binary (0 = background, 1 = tumor) |

---

## 🔧 Configuration

```python
# Hyperparameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
CHANNELS = 3
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3
VAL_SPLIT = 0.2

# Optimizer & Loss
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss = 'binary_crossentropy'
metrics = ['accuracy', BinaryIoU(target_class_ids=[1])]
```

---

## 📂 Project Structure

```
Brain-Tumor-Segmentation-UNet/
│
├── 📓 UNET_Brain_Tumor_Quiz.ipynb    # Main training notebook
├── 📄 README.md                       # Project documentation
├── 📋 requirements.txt                # Python dependencies
├── 📜 LICENSE                         # MIT License
│
├── 📁 data/                           # Dataset directory
│   ├── images/                        # MRI scan images
│   └── masks/                         # Segmentation masks
│
└── 📁 models/                         # Saved models
    └── unet_brain_tumor.h5            # Trained model weights
```

---

## 🛠️ Tech Stack

<div align="center">

| Technology | Purpose |
|------------|---------|
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) | Deep Learning Framework |
| ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white) | High-level Neural Network API |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | Numerical Computing |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) | Visualization |
| ![Google Colab](https://img.shields.io/badge/Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white) | Cloud Computing |

</div>

---

## 🔮 Future Improvements

- [ ] 🎨 **Data Augmentation** — Rotation, flipping, elastic deformations
- [ ] 📐 **Higher Resolution** — 256×256 or 512×512 input images
- [ ] 🏋️ **Transfer Learning** — Pre-trained encoder (ResNet, EfficientNet)
- [ ] 📉 **Dice Loss** — Combined BCE + Dice loss function
- [ ] 🔄 **Attention U-Net** — Add attention gates for better localization
- [ ] 📱 **Model Deployment** — TensorFlow Lite / ONNX export
- [ ] 🌐 **Web Interface** — Streamlit or Gradio demo

---

## 📚 References

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) — Ronneberger et al., 2015
- [TensorFlow Image Segmentation Tutorial](https://www.tensorflow.org/tutorials/images/segmentation)
- [Keras Documentation](https://keras.io/api/)

---

## 👨‍💻 Author

<div align="center">

### **Zeyad Ayman**

[![GitHub](https://img.shields.io/badge/GitHub-ZeyadArafa-181717?style=for-the-badge&logo=github)](https://github.com/ZeyadArafa)

</div>

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

### ⭐ Star this repository if you found it helpful!

Made with ❤️ and 🧠 for medical AI research

</div>
