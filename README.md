# Brain-tumor-DL.project

🧠 Brain Tumor MRI Classification System
Deep Learning Project – Cyber Security / AI Track

This repository contains the full design, implementation, training, evaluation, and deployment of a Deep Learning–based medical imaging system for Brain Tumor Classification using MRI scans.
The project integrates data preprocessing, custom CNN modeling, transfer learning (VGG16 & ResNet50), model evaluation, and a user-friendly GUI, all implemented using PyTorch.

1. Project Overview

The objective of this project is to build a robust and accurate deep learning system capable of classifying brain MRI images into multiple tumor categories.

The system simulates a real-world medical AI pipeline, starting from raw data ingestion to final deployment through a graphical user interface.

🎯 2. Key Features
Deep Learning & AI

Custom CNN architecture built from scratch

Transfer Learning using VGG16 and ResNet50

Fine-tuning of pre-trained networks

Regularization using Dropout & Batch Normalization

Automatic feature extraction from MRI images

Data Processing

Image resizing and normalization

Extensive data augmentation

Train / Validation / Test splitting

Robust preprocessing pipeline

Evaluation & Validation

Accuracy measurement

Confusion Matrix visualization

Precision, Recall, and F1-score

Model comparison and best-model selection

Inference on unseen MRI images

Deployment

Interactive GUI using Streamlit

Image upload and real-time prediction

Visualization of prediction results

3. Dataset Description

Dataset Name: Brain Tumor MRI Dataset

Data Type: MRI Images

Number of Classes: 4

Dataset Structure:

Training set (with validation split)

Independent testing set

The dataset represents a real medical imaging problem suitable for deep learning applications.

4. System Architecture

The system follows a modular deep learning pipeline:

Data Loading & Preprocessing

Data Augmentation

Model Design

Model Training

Model Validation

Model Testing

Model Deployment (GUI)

Each module is developed and tested independently, then integrated into a complete system.

5. Models Implemented
5.1 Custom CNN

Convolutional layers for feature extraction

MaxPooling layers for spatial reduction

Fully connected layers for classification

Dropout layers for regularization

5.2 VGG16 – Transfer Learning

Pre-trained on ImageNet

Partial freezing of convolutional layers

Custom classifier head

Fine-tuning for medical imaging domain

5.3 ResNet50 – Transfer Learning

Residual learning architecture

Freezing early layers

Fine-tuning higher-level features

Improved convergence and accuracy

6. Model Training Strategy

Loss Function: CrossEntropyLoss

Optimizer: Adam

Learning Rate: 0.0001

Batch Size: 32

Training Approach:

Epoch-based training

Validation after each epoch

Best model saved based on validation accuracy

7. Testing & Evaluation Performed

✔ Validation accuracy comparison
✔ Final test accuracy measurement
✔ Confusion matrix analysis
✔ Classification report generation
✔ Inference on real unseen MRI images
✔ Best model selection and deployment

🖥️ 8. GUI Implementation

A Graphical User Interface (GUI) was implemented using Streamlit to allow non-technical users to interact with the trained model.

GUI Capabilities:

Upload MRI image

Run prediction

Display predicted tumor class

Show input image for reference

📂 9. Repository Structure
📁 Brain-Tumor-MRI-Classification/
│
├── README.md
│
├── src/
│   ├── data_preprocessing.py
│   ├── cnn_model.py
│   ├── transfer_learning.py
│   ├── train.py
│   └── test_and_evaluation.py
│
├── gui/
│   └── app.py
│
├── saved_models/
│   ├── cnn_best.pth
│   ├── vgg16_best.pth
│   └── resnet50_best.pth
│
├── notebooks/
│   └── experiments.ipynb
│
└── requirements.txt

👥 10. Team Members & Responsibilities
Member	Role	Responsibilities
Member 1	Team Leader & System Architect	Project planning, system integration, model testing, evaluation, result analysis
Member 2	Data Engineer	Dataset analysis, preprocessing, data augmentation
Member 3	CNN Model Developer	Custom CNN architecture design and training
Member 4	Transfer Learning Engineer	VGG16 & ResNet50 fine-tuning and optimization
Member 5	GUI Developer	Streamlit GUI development and deployment

All tasks were distributed equally to ensure balanced contribution across the team.

▶️ 11. How to Run the Project
pip install -r requirements.txt
streamlit run gui/app.py

🧰 12. Technologies Used

Python

PyTorch

Streamlit

Scikit-learn

Matplotlib

Seaborn

📄 13. License

This project is developed for educational and academic purposes as part of a Deep Learning course and is intended to demonstrate practical applications of AI in medical imaging.

💬 14. Feedback & Contributions

Suggestions and improvements are welcome to enhance model performance, usability, or documentation quality.
