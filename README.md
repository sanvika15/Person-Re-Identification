# 👣 Person Re-Identification using Deep Learning

A deep learning project for **Person Re-Identification (Re-ID)** using the **Market-1501** dataset and **Triplet Loss with EfficientNet-B0**. The goal is to learn robust embeddings to match individuals across different camera views.

---

## 🧠 Project Objective

This project aims to build a **deep feature extractor** that transforms images of individuals into a high-dimensional embedding space, where:

- Images of the **same person** (from different cameras) are mapped close together.
- Images of **different people** are mapped far apart.

This is achieved using **Triplet Loss**, which trains the model on samples consisting of:

- **Anchor Image**: A reference image of a person.
- **Positive Image**: An image of the same person from a different camera.
- **Negative Image**: An image of a different person.

---

## 📁 Dataset: Market-1501

- 📦 **Source**: [Kaggle - Market-1501](https://www.kaggle.com/pengcw1/market-1501)
- 📸 32,668 annotated bounding boxes
- 👤 1,501 unique identities
- 📷 Captured from 6 different cameras

### 📌 Sample Triplet:

<p align="center">
  <img src="images/sample_triplet.png" alt="Triplet Sample" width="600"/>
</p>

Each triplet helps the model **learn by comparison**:  
👉 "Anchor and Positive should be close. Anchor and Negative should be far."

---

## 🔄 Working Pipeline

<p align="center">
  <img src="images/pipeline.png" alt="Pipeline Flow" width="800"/>
</p>

### 🔍 Step-by-Step Explanation:

1. **Image Loader**  
   Loads anchor, positive, and negative images using custom `TripletDataset`.

2. **Preprocessing**  
   Images are resized, normalized, and augmented using `Albumentations`.

3. **Feature Extraction**  
   The preprocessed images are passed through **EfficientNet-B0**, outputting a 512-dim embedding vector for each.

4. **Triplet Loss**  
   The **TripletMarginLoss** ensures:
   - The anchor is **closer** to the positive than to the negative by a margin.
   - This is done in the **embedding space**, not pixel space.

5. **Training Loop**  
   The model is optimized using Adam + LR scheduling to learn meaningful embeddings.

6. **Evaluation**  
   Embeddings are compared using **cosine or Euclidean distance** to identify matches.

---

## 🔧 Tech Stack

- **Language**: Python
- **Framework**: PyTorch
- **Model**: EfficientNet-B0 (via `timm`)
- **Image Processing**: OpenCV, Albumentations
- **Evaluation**: Scikit-image, Matplotlib

---

## 🧩 Model Architecture

- 📌 **Backbone**: Pretrained EfficientNet-B0
- 🔗 **Embedding Layer**: Outputs 512-dim vector
- 🎯 **Loss Function**: TripletMarginLoss

<p align="center">
  <img src="images/model_architecture.png" alt="Model Architecture" width="600"/>
</p>

---
