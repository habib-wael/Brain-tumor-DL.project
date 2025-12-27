# 🧠 Brain Tumor MRI Classification System
### **Deep Learning Project 

This repository presents the complete lifecycle of a **Deep Learning–based medical imaging system** for **brain tumor classification using MRI scans**.  
The project covers **data preprocessing, model design, training, evaluation, and deployment** through an interactive graphical interface, following a real-world AI project workflow.

---

## 1. Project Overview

Brain tumor detection from MRI images is a critical medical task that requires high accuracy and reliability.  
The goal of this project is to design and implement a **robust and scalable deep learning system** capable of classifying brain MRI images into multiple tumor categories.

The project simulates a **real-world AI pipeline**, starting from raw medical data and ending with a deployable application that can be used by non-technical users.

---

## 2. Objectives

- Apply deep learning techniques to medical imaging data  
- Compare custom CNN models with transfer learning architectures  
- Evaluate model performance using standard classification metrics  
- Deploy the best-performing model through a graphical user interface  
- Follow a structured and documented project workflow  

---

## 3. Dataset Description

- **Dataset Name:** Brain Tumor MRI Dataset  
- **Data Type:** MRI Images  
- **Number of Classes:** 4  
- **Dataset Structure:**
  - Training set (with validation split)
  - Independent testing set

The dataset represents a realistic medical imaging problem and is suitable for deep learning applications beyond basic benchmark datasets.

---

## 4. Data Preprocessing and Analysis

To ensure effective training and generalization, the following preprocessing steps were applied:

- Image resizing to **256 × 256**
- Normalization using ImageNet statistics
- Data augmentation techniques:
  - Random horizontal flipping
  - Random rotation
  - Color jitter (brightness and contrast)
- Training and validation split (80% / 20%)

These steps help reduce overfitting and improve model robustness.

---

## 5. Model Architecture and Design

Three different deep learning models were implemented and evaluated:

### 5.1 Custom CNN
- Designed from scratch
- Convolutional layers for feature extraction
- Max-pooling layers for spatial reduction
- Fully connected layers for classification
- Dropout for regularization

### 5.2 VGG16 (Transfer Learning)
- Pre-trained on ImageNet
- Early layers frozen to preserve learned features
- Custom classifier head added
- Partial fine-tuning for domain adaptation

### 5.3 ResNet50 (Transfer Learning)
- Residual learning architecture
- Early layers frozen
- Fine-tuning applied to higher layers
- Improved convergence and performance stability

Transfer learning was used to leverage pre-trained knowledge and enhance performance on limited medical datasets.

---

## 6. Model Training Strategy

- **Loss Function:** CrossEntropyLoss  
- **Optimizer:** Adam  
- **Learning Rate:** 0.0001  
- **Batch Size:** 32  

Training was performed using a validation-based approach, where the best model was saved based on validation accuracy.

---

## 7. Testing and Evaluation

The trained models were evaluated using unseen test data.

### Evaluation Metrics:
- Accuracy
- Confusion Matrix
- Precision, Recall, and F1-score

The best-performing model was selected based on both validation and test performance and used for final deployment.

---

## 8. Deployment and GUI Implementation

A **Graphical User Interface (GUI)** was developed using **Streamlit** to enable easy interaction with the trained model.

### GUI Features:
- Upload MRI image
- Run real-time inference
- Display predicted tumor class
- Visualize the input image

The GUI allows the system to be used without direct interaction with the source code.

---
```
## 9. Project Structure
Brain-Tumor-MRI-Classification/
│
├── README.md
│
├── src/
│ ├── data_preprocessing.py
│ ├── cnn_model.py
│ ├── transfer_learning.py
│ ├── train.py
│ └── test_and_evaluation.py
│
├── gui/
│ └── app.py
│
├── saved_models/
│ ├── cnn_best.pth
│ ├── vgg16_best.pth
│ └── resnet50_best.pth
│
├── notebooks/
│ └── experiments.ipynb
│
└── requirements.txt

---

## 10. Team Members and Responsibilities

| Member | Role | Responsibilities |
|------|------|----------------|
| Member 1 | **Team Leader & System Architect** | Project planning, system integration, model testing, evaluation, results analysis |
| Member 2 | **Data Engineer** | Dataset analysis, preprocessing, data augmentation |
| Member 3 | **CNN Model Developer** | Custom CNN design and training |
| Member 4 | **Transfer Learning Engineer** | VGG16 & ResNet50 fine-tuning and optimization |
| Member 5 | **GUI Developer** | Streamlit GUI development and deployment |

The project workload was distributed evenly to ensure fair contribution among all team members.
```
---


12. Technologies Used

Python

PyTorch

Streamlit

Scikit-learn

Matplotlib

Seaborn

13. License

This project is developed for educational and academic purposes and demonstrates the practical application of deep learning techniques in medical imaging.
