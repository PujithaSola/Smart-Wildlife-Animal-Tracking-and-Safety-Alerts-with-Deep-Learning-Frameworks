
# Smart Wildlife Animal Tracking and Safety Alerts

## 📌 Project Overview

This project aims to minimize human-wildlife conflict by detecting and classifying wild animals in real-time using deep learning techniques. It combines object detection (YOLOv8n) with image classification (ResNet18) to provide alerts for potentially dangerous wildlife near human settlements.

The project uses datasets collected from Kaggle and web scraping tools, and applies augmentation techniques to improve the model generalization. The trained models produce real-time detections and classifications, accompanied by result metrics and visualizations.

---

## 📂 Repository Contents

* `YOLOv8n_Training.ipynb` — YOLOv8n object detection training code
* `YOLOv8n_Testing.ipynb` — YOLOv8n testing & evaluation
* `ResNet18_Training.ipynb` — ResNet18 image classification training code
* `ResNet18_Testing.ipynb` — ResNet18 testing & evaluation
* `Graphs/` — Folder containing result graphs (training loss, accuracy curves, confusion matrices)
* `Test_Code/` — Code snippets to test models on sample images

---

## 📝 Dataset

> ⚠ **Note:** The dataset is not included in this repository due to size limitations.

* **Sources:**

  * Kaggle (Animal Image Datasets)
  * Web scraping using `iCrawler.io`

* **Dataset Summary:**

  * Total \~1500 images
  * 10 classes for detection model
  * 5 classes for classification model
  * Data augmentation applied: flipping, rotation, brightness adjustments, etc.

---

## 🏗️ Models Used

### 1️⃣ YOLOv8n (Detection Model)

* Lightweight real-time object detection model
* Divides input image into grid cells and predicts bounding boxes & class probabilities.
* Post-processed using Non-Maximum Suppression (NMS)
* Selected for real-time inference with limited hardware resources.

### 2️⃣ ResNet18 (Classification Model)

* Convolutional Neural Network with Residual Blocks
* Prevents vanishing gradient problem in deeper networks.
* Fine-grained classification into 5 wild animal classes.

---

## 📊 Model Results

### YOLOv8n Detection

| Metric    | Value |
| --------- | ----- |
| mAP50     | 71.8% |
| Precision | 68%   |
| Recall    | 70.4% |

### ResNet18 Classification

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 93.33% |
| Precision | 94%    |
| Recall    | 93%    |
| F1 Score  | 93%    |

---

## ⚙️ How to Run

1️⃣ Clone this repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2️⃣ Setup virtual environment (optional but recommended)

3️⃣ Install dependencies (requirements.txt file can be generated if needed)

```bash
pip install -r requirements.txt
```

4️⃣ Download dataset (not provided here).
Structure your dataset as per the training code expectations.

5️⃣ Open the respective `.ipynb` files using **Jupyter Notebook** or **Google Colab**.

6️⃣ Run the training and testing notebooks.

---

## 📌 Results Visualization

* The repository contains training graphs:

  * Loss vs Epochs
  * Accuracy vs Epochs
  * Confusion Matrices
  * Precision-Recall curves
  * F1-confidence curves

You can explore these in the `Graphs/` folder.

---

## 🚀 Future Work

* Deploy trained models on surveillance cameras or IoT-based embedded systems.
* Extend datasets for more animal species.
* Improve detection accuracy with transfer learning and data expansion.
* Real-time field deployment in forests or protected areas.
