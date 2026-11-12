# Lungcare-AI-

# 🫁 LungCare AI – Chest X-Ray Classification

### Deep Learning-Based Pneumonia Classification Using CNN and PyTorch

## 📌 Project Overview

LungCare AI is a deep learning-based chest X-ray classification project that uses a custom Convolutional Neural Network (CNN) developed with PyTorch.

The application classifies chest X-ray images into two categories:

- **NORMAL** – The model classifies the X-ray image as Normal.
- **PNEUMONIA** – The model classifies the X-ray image as Pneumonia.

The project uses computer vision techniques for image preprocessing and deep learning for classification. A FastAPI web application allows users to upload chest X-ray images and view the model's prediction.

## 🎯 Problem Statement

Pneumonia is a respiratory infection that causes inflammation in the lungs, particularly in the alveoli (small air sacs).

Chest X-ray images are commonly used by healthcare professionals when evaluating suspected pneumonia.

Manually examining X-ray images can be time-consuming. This project explores how deep learning and computer vision can be used to classify chest X-ray images into NORMAL and PNEUMONIA categories.

The objective is to develop an image classification system that accepts a chest X-ray image and predicts its corresponding category.

## 💡 Proposed Solution

To address this problem, we developed a custom Convolutional Neural Network using PyTorch.

The project is divided into two main stages:

### 1. Model Development and Training

- Load the chest X-ray dataset.
- Select NORMAL and PNEUMONIA images.
- Exclude COVID19 images from the classification task.
- Resize images to 224 × 224 pixels.
- Convert images into RGB format.
- Normalize image pixel values.
- Apply data augmentation to training images.
- Train a custom CNN using PyTorch.
- Evaluate the model using validation and test datasets.
- Save the best trained model as `xray_model.pth`.

### 2. Web Application and Prediction

- Develop a web application using FastAPI.
- Allow users to upload chest X-ray images.
- Validate the uploaded image.
- Preprocess the image before prediction.
- Load the trained CNN model.
- Generate prediction scores using Softmax.
- Display the predicted category and corresponding scores.

## 🏗️ Project Architecture

```text
             MODEL DEVELOPMENT

          Chest X-Ray Dataset
                    |
                    v
             Data Preprocessing
          (Resize, RGB, Normalize)
                    |
                    v
              Data Augmentation
                    |
                    v
              Custom CNN Model
                    |
                    v
               Model Training
              (PyTorch + AdamW)
                    |
                    v
              Model Evaluation
                    |
                    v
              Best Model Saved
               xray_model.pth


               PREDICTION SYSTEM

             User Uploads X-Ray
                    |
                    v
               Web Interface
                (index.html)
                    |
                    v
              FastAPI Backend
                 (app.py)
                    |
                    v
             Image Preprocessing
                    |
                    v
              Trained CNN Model
                    |
                    v
                 Softmax
                    |
                    v
               Classification
                    |
             -----------------
             |               |
             v               v
           NORMAL         PNEUMONIA
```

## 🧠 Model Architecture

The project uses a **Custom Convolutional Neural Network (CNN)** implemented using PyTorch.

The CNN architecture consists of:

- Five Convolutional Layers
- Batch Normalization
- ReLU Activation Function
- Four Max Pooling Operations
- Adaptive Average Pooling
- Fully Connected Layers
- Dropout Regularization

### Model Configuration

| Parameter | Value |
|---|---|
| Model Architecture | Custom CNN |
| Deep Learning Framework | PyTorch |
| Input Image Size | 224 × 224 |
| Input Channels | 3 (RGB) |
| Number of Classes | 2 |
| Optimizer | AdamW |
| Loss Function | Weighted CrossEntropyLoss |
| Batch Size | 16 |
| Learning Rate | 0.001 |
| Maximum Epochs | 15 |
| Regularization | Dropout and Weight Decay |
| Model File | `xray_model.pth` |

## ⚙️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| PyTorch | CNN model development and training |
| Torchvision | Image transformations and preprocessing |
| FastAPI | Backend API development |
| Uvicorn | Running the FastAPI application |
| HTML | Web page structure |
| CSS | Web page styling |
| JavaScript | Frontend interactions |
| Pillow (PIL) | Image loading and processing |
| Git | Version control |
| GitHub | Source code hosting |

## 📂 Project Structure

```text
LungCare-AI/
│
├── app.py
├── index.html
├── train_local.py
├── xray_model.pth
├── requirements.txt
├── .gitignore
│
└── xray/
    └── ml/
        └── model/
            └── arch.py
```

## 📊 Dataset

The dataset contains chest X-ray images belonging to three original categories:

- NORMAL
- PNEUMONIA


### Dataset Organization

```text
Data/
│
├── train/
│   ├── NORMAL/
│   ├── PNEUMONIA/
│
└── test/
    ├── NORMAL/
    ├── PNEUMONIA/
    
```

### Dataset Preprocessing

The following preprocessing operations are applied:

1. Resize images to 224 × 224 pixels.
2. Convert images into RGB format.
3. Convert images into PyTorch tensors.
4. Normalize images using ImageNet mean and standard deviation.
5. Apply random rotation and brightness/contrast augmentation to training images.

## 🔄 Model Training Workflow

The model training process includes:

1. Loading the dataset.
2. Selecting NORMAL and PNEUMONIA classes.
3. Splitting the training data into training and validation sets.
4. Preparing images using preprocessing and augmentation.
5. Passing images through the custom CNN.
6. Calculating classification loss using Weighted CrossEntropyLoss.
7. Updating model weights using the AdamW optimizer.
8. Adjusting the learning rate using ReduceLROnPlateau.
9. Applying early stopping to control overfitting.
10. Saving the best model checkpoint.
11. Evaluating the trained model on the test dataset.

## 🚀 How to Run the Project Locally

### Prerequisites

- Python 3.10.11
- VS Code
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/LungCare-AI.git
```

```bash
cd LungCare-AI
```

Replace `YOUR-USERNAME` with your GitHub username.

### Step 2: Create a Virtual Environment

For Windows PowerShell:

```powershell
python -m venv lungenv
```

Activate the environment:

```powershell
.\lungenv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Check the Trained Model

Make sure `xray_model.pth` is available in the project root directory.

The FastAPI application loads this trained model to perform predictions.

### Step 5: Run the Application

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

### Step 6: Open the Web Application

Open your browser and visit:

```text
http://127.0.0.1:8000
```

### Step 7: Access API Documentation

FastAPI provides interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

### Health Check

```text
http://127.0.0.1:8000/health
```

## 🔍 Prediction Process

1. Open the LungCare AI web application.
2. Upload a chest X-ray image.
3. The application checks the uploaded file.
4. The image is converted to RGB format.
5. The image is resized and normalized.
6. The processed image is passed to the trained CNN.
7. The model generates raw output logits.
8. Softmax converts the logits into normalized class scores.
9. The application displays the predicted class and scores.

### Prediction Output

The application returns:

- Predicted category
- Class index
- NORMAL score
- PNEUMONIA score
- Educational-use disclaimer

## 📈 Model Evaluation

The model is evaluated using a separate test dataset.

The evaluation process includes:

- Accuracy
- Confusion Matrix
- NORMAL Class Recall
- PNEUMONIA Class Recall

Actual final test results can be added here after verifying the training output.

## 🔮 Future Enhancements

- Improve classification performance through further experimentation.
- Evaluate the model using additional independent datasets.
- Add more model evaluation metrics.
- Improve the web interface.
- Explore model explainability techniques.

## 👩‍💻 Author

**Aditi Nikalje**

Computer Science and Engineering

## ⚠️ Disclaimer

LungCare AI is an educational deep learning project. The predictions generated by this application are not medical diagnoses.

The model has not been clinically validated and should not be used as a substitute for professional medical advice, examination, or diagnosis.
