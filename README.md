
# 🧠 Brain Tumor Segmentation using U-Net

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

<p align="center">
  <a href="https://github.com/ZeyadArafa/Brain-Tumor-Segmentation-UNet/stargazers">
    <img src="https://img.shields.io/github/stars/ZeyadArafa/Brain-Tumor-Segmentation-UNet?style=social" alt="Stars">
  </a>
  <a href="https://github.com/ZeyadArafa/Brain-Tumor-Segmentation-UNet/network/members">
    <img src="https://img.shields.io/github/forks/ZeyadArafa/Brain-Tumor-Segmentation-UNet?style=social" alt="Forks">
  </a>
</p>

<p align="center">
  <b>A Deep Learning approach to automate the detection of brain tumors from MRI scans using semantic segmentation.</b>
  <br>
  <i>Pixel-level accuracy. Medical-grade precision.</i>
</p>

---

## 📋 Table of Contents

1. [📖 Project Overview](#-project-overview)
2. [🏗️ Methodology & Architecture](#%EF%B8%8F-methodology--architecture)
3. [📂 Dataset Structure](#-dataset-structure)
4. [📊 Performance Metrics](#-performance-metrics)
5. [🚀 Installation & Usage](#-installation--usage)
6. [🖼️ Visualizations](#%EF%B8%8F-visualizations)
7. [🛣️ Roadmap](#%EF%B8%8F-roadmap)
8. [👨‍💻 Author](#%EF%B8%8F-author)

---

## 📖 Project Overview

Diagnosing brain tumors manually from MRI scans is time-consuming and prone to human error. This project leverages **Deep Learning** to automate this process.

Using a custom **U-Net architecture**, the model learns to identify tumor regions pixel-by-pixel, outputting a binary mask where:
* **White (1):** Tumor Region
* **Black (0):** Healthy Tissue

### Key Features
* ✅ **End-to-End Pipeline:** From raw image loading to binary mask prediction.
* ✅ **Custom U-Net:** Built from scratch with Dropout regularization to prevent overfitting.
* ✅ **IoU Metric:** Optimization using Intersection over Union (Jaccard Index) for imbalanced data.
* ✅ **Optimized Training:** Implements `tf.data` for efficient GPU utilization.

---

## 🏗️ Methodology & Architecture 

The model follows the standard **U-Net** encoder-decoder design, specifically tuned for 128x128 inputs.

| Component | Description |
| :--- | :--- |
| **Encoder** | Captures context using convolutional layers and Max Pooling (Downsampling). |
| **Bottleneck** | The bridge between encoder and decoder, extracting high-level features. |
| **Decoder** | Enables precise localization using Transposed Convolutions (Upsampling). |
| **Skip Connections** | Concatenates encoder features with decoder layers to recover spatial details lost during pooling. |

**Hyperparameters:**
* **Input Size:** `128 x 128 x 3`
* **Batch Size:** `16`
* **Optimizer:** `Adam (lr=0.001)`
* **Loss Function:** `Binary Crossentropy`

---

## 📂 Dataset Structure

The project expects the data to be organized in the following directory tree:

```text
Brain-Tumor-Segmentation-UNet/
│
├── dataset/
│   ├── images/          # Original MRI Scans (RGB)
│   │   ├── image_001.png
│   │   └── ...
│   └── masks/           # Binary Ground Truth Masks
│       ├── mask_001.png
│       └── ...
│
├── UNET_Brain_Tumor_Quiz.ipynb   # Main Training Notebook
├── README.md                     # Project Documentation
└── requirements.txt              # Dependencies

```

---

## 📊 Performance Metrics

The model is evaluated using **Accuracy** and **IoU (Intersection over Union)**. IoU is critical in medical segmentation because background pixels often vastly outnumber tumor pixels.

| Metric | Training Score | Validation Score |
| --- | --- | --- |
| **Accuracy** | ~99.4% | ~99.1% |
| **IoU Score** | ~0.60 | ~0.56 |

> **Note:** An IoU score > 0.5 is generally considered a good overlap for complex medical segmentation tasks without extensive pre-training.

---

## 🚀 Installation & Usage

### 1. Clone the Repository

```bash
git clone [https://github.com/ZeyadArafa/Brain-Tumor-Segmentation-UNet.git](https://github.com/ZeyadArafa/Brain-Tumor-Segmentation-UNet.git)
cd Brain-Tumor-Segmentation-UNet

```

### 2. Install Dependencies

Ensure you have Python installed, then run:

```bash
pip install -r requirements.txt

```

*Alternatively, you can install the main packages directly:*

```bash
pip install tensorflow numpy matplotlib

```

### 3. Run the Training

You can open the notebook `UNET_Brain_Tumor_Quiz.ipynb` in **Google Colab** or **Jupyter Notebook**.

**If using Google Colab:**

1. Upload your `dataset` folder to Google Drive.
2. Mount Drive in the notebook:

```python
from google.colab import drive
drive.mount('/content/drive')

```

3. Update the `DATA_DIR` path in the code to point to your folder.

---

## 🖼️ Visualizations

The notebook includes a visualization block that compares the **Original Image**, the **Ground Truth Mask**, and the **Predicted Mask**.

| MRI Scan | Ground Truth | Prediction |
| --- | --- | --- |
| 🖼️ | 🏁 | 🤖 |
| *(Original)* | *(Actual Tumor)* | *(AI Output)* |

---

## 🛣️ Roadmap

* [x] Implement U-Net Architecture
* [x] Integrate IoU Metric
* [ ] Add Data Augmentation (Rotation, Flip) to improve generalization
* [ ] Experiment with Dice Loss function
* [ ] Deploy model as a web app using Streamlit

---

## 👨‍💻 Author

**Zeyad Ayman** ([@ZeyadArafa](https://github.com/ZeyadArafa))

* 🎓 **Computer Engineering Student**
* 💻 **Interests:** AI, Computer Vision, Cybersecurity, DevOps, Software Engineering

---

<p align="center">
<i>"Deep learning can save lives — one pixel at a time."</i>
</p>
