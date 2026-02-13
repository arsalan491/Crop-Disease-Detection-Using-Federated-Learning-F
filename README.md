# Crop Disease Detection Using Federated Learning

## Group Members
- ARSALAN KHAN (2024-MS-AIE-17)

---

## Project Overview

Crop diseases can cause significant losses in agriculture. Detecting diseases early is crucial for improving crop yield and reducing financial loss.  

This project implements **Federated Learning** with a **pre-trained ResNet18** to detect crop diseases from leaf images while preserving data privacy across multiple farms (clients).  

**Key features:**  
- Federated Learning with multiple clients  
- Pre-trained ResNet18 for high accuracy  
- Data augmentation for better generalization  
- Single image inference  
- Save and load trained models  

---

## Dataset

We use the **Crop Disease Detection Dataset** available on Kaggle:  
[Crop Disease Detection Dataset](https://www.kaggle.com/snikhilrao/crop-disease-detection-dataset)

### Dataset Structure

Crop-Disease-Detection-Dataset/
├── Train/ # Training images organized by class
├── Val/ # Validation images organized by class
└── Test/ # Test images

The dataset contains **multiple classes of crop diseases**, including healthy leaves. Each class has its own folder containing leaf images.

---
## Architecture / Component Diagram

<img width="1536" height="1024" alt="ChatGPT Image Feb 13, 2026, 03_37_27 PM" src="https://github.com/user-attachments/assets/460ddc9d-1022-4a47-81f4-6fb9a60d028e" />


## Environment & Dependencies

Tested on **Python 3.10+** and requires:

torch>=2.0
torchvision>=0.18
numpy
Pillow


Install dependencies with:

```bash
pip install -r requirements.txt

## Project Structure

crop-disease-federated/
├── data/                     # Optional: small sample images or instructions
├── models/                   # Trained models (.pth)
├── src/
│   ├── dataset_loader.py     # Dataset loading & transformations
│   ├── model.py              # Pre-trained ResNet18 definition
│   ├── federated_train.py    # Federated training loop
│   └── inference.py          # Single image inference
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── LICENSE                   # License file

## Usage Instructions
1. Download Dataset

Download the Kaggle dataset from:
https://www.kaggle.com/snikhilrao/crop-disease-detection-dataset
data/Train/
 Val/Test/   # optional

2. Training the Federated Model

Run federated training:
python src/federated_train.py

Features:

Multiple clients simulate separate farms

Local training on each client dataset

Aggregation of models via FedAvg

Validation after each federated round

Saves trained global model to models/global_resnet_federated.pth


3. Single Image Inference

Predict the disease class of a leaf image using:

python src/inference.py


Inside the script:

from inference import predict_image

class_names = [...]  # Replace with train_dataset.classes
image_path = "path_to_leaf.jpg"
predicted_class = predict_image(image_path, class_names)
print(f"The predicted disease class is: {predicted_class}")

4. Model Loading for Custom Use

Load the trained model in any script:

from model import get_resnet_model
import torch

NUM_CLASSES = 38  # Replace with your dataset's number of classes
MODEL_PATH = "models/global_resnet_federated.pth"

model = get_resnet_model(NUM_CLASSES, pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

## Performance & Accuracy

Using pre-trained ResNet18 + 224×224 input + data augmentation, this pipeline can achieve 90%+ validation accuracy.

Federated learning simulates multiple clients while preserving privacy.

Increase LOCAL_EPOCHS and ROUNDS to improve accuracy further.

Reduce NUM_CLIENTS if dataset per client is too small to avoid underfitting.
