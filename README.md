# 🔤 Arabic Handwritten Character Recognition

This deep learning project classifies Arabic handwritten characters using a Convolutional Neural Network (CNN). It leverages image preprocessing, one-hot encoding, and TensorFlow/Keras to train a high-accuracy image classifier.

## 📁 Project Overview

The notebook covers the following steps:

- Loading and preprocessing grayscale images of Arabic letters
- Resizing and normalizing input images
- Visualizing the dataset
- Building a CNN model using Keras
- Training and evaluating the model
- Displaying a classification report and confusion matrix

## 🧠 Technologies Used

- Python
- OpenCV
- TensorFlow / Keras
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn

## 🗃️ Dataset

The dataset used is the **Final Arabic Alpha Dataset**, containing grayscale images of handwritten Arabic letters categorized by label folders.

- Each image is resized to `128x128`
- Normalized pixel values (0 to 1)
- One-hot encoding applied to labels

## 🏗️ Model Architecture

The CNN model includes:

- Convolutional and pooling layers
- Flattening and dense layers
- Dropout regularization
- Softmax output for multi-class classification

## 📊 Evaluation

The notebook provides:

- Accuracy score
- Confusion matrix
- Classification report (precision, recall, F1-score)

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/arabic-character-cnn.git
   cd arabic-character-cnn
Install dependencies:

pip install numpy pandas opencv-python matplotlib seaborn scikit-learn tensorflow
Place the dataset in the specified path or update train_data_path and test_data_path.

Run the notebook:

jupyter notebook zeina-ahmed-320210137.ipynb
👤 Author
Created by Zeina Ahmed, a machine learning and computer vision enthusiast.
