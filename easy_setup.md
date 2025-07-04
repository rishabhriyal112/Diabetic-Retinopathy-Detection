Here is the updated README file with the correct GitHub repository URL:

# Diabetic Retinopathy Detection 🤖

This project uses a deep learning model to detect diabetic retinopathy from retinal images. 📸

## Table of Contents 📚

* [Introduction](#introduction) 🤔
* [Requirements](#requirements) 📝
* [Installation](#installation) 💻
* [Usage](#usage) 📊
* [Model](#model) 🤖
* [Training](#training) 📚
* [Evaluation](#evaluation) 📊
* [Conclusion](#conclusion) 🎉

## Introduction 🤔

Diabetic retinopathy is a serious complication of diabetes that can lead to blindness. 🤕 Early detection and treatment are crucial to prevent vision loss. This project aims to develop a deep learning model that can detect diabetic retinopathy from retinal images. 📸

## Requirements 📝

* Python 3.8 or later 🐍
* TensorFlow 2.x or later 🤖
* Keras 2.x or later 🤖
* NumPy 1.20 or later 📊
* Pandas 1.3 or later 📊
* Matplotlib 3.4 or later 📊
* Scikit-learn 0.24 or later 📊
* OpenCV 4.5 or later 📸
* Flask 📊
* Streamlit 📊

## Installation 💻

To install the required libraries and set up the environment, follow these steps:

1. Install Miniconda from the official website: <https://docs.conda.io/en/latest/miniconda.html> 📚
2. Create a new environment by running the following command:
```bash
conda create --name diabetic_retinopathy
```
3. Activate the environment by running the following command:
```bash
conda activate diabetic_retinopathy
```
4. Install the required libraries by running the following command:
```bash
conda install -c conda-forge tensorflow numpy pandas matplotlib scikit-learn opencv
```
5. Install Flask and Streamlit by running the following command:
```bash
pip install flask streamlit
```
6. Clone the repository using the following command:
```bash
git clone https://github.com/siddhant-rajhans/Diabetic_Retinopathy_Detection_Using_Deep_Learning.git
```
7. Navigate to the repository directory using the following command:
```bash
cd Diabetic_Retinopathy_Detection_Using_Deep_Learning
```
8. Run the Flask(api) application using the following command:
```bash
python app.py
```
9. Run the Streamlit application using the following command:
```bash
streamlit run streamlit_app.py
```
10. Access the applications by navigating to `http://localhost:5000` and `http://localhost:8501` in your web browser. 📊

## Usage 📊

To use the model, simply run the Flask application and access it through your web browser. 📊

## Model 🤖

The model used in this project is a convolutional neural network (CNN) that takes retinal images as input and outputs a probability distribution over the five classes of diabetic retinopathy. 📸

## Training 📚

The model was trained on a dataset of retinal images with diabetic retinopathy. The training process involved the following steps:

* Data preprocessing: The images were resized to 224x224 pixels and normalized to have zero mean and unit variance. 📊
* Data augmentation: The images were augmented using random rotation, flipping, and cropping. 📸
* Model training: The model was trained using the Adam optimizer and categorical cross-entropy loss function. 📊
