# Diabetic Retinopathy Detection

This project uses a deep learning model to detect diabetic retinopathy from retinal images.

## Table of Contents

* [Introduction](#introduction)
* [Requirements](#requirements)
* [Installation](#installation)
* [Usage](#usage)
* [Model](#model)
* [Training](#training)
* [Evaluation](#evaluation)
* [Conclusion](#conclusion)

## Introduction

Diabetic retinopathy is a serious complication of diabetes that can lead to blindness. Early detection and treatment are crucial to prevent vision loss. This project aims to develop a deep learning model that can detect diabetic retinopathy from retinal images.

## Requirements

* Python 3.8 or later
* TensorFlow 2.x or later
* Keras 2.x or later
* NumPy 1.20 or later
* Pandas 1.3 or later
* Matplotlib 3.4 or later
* Scikit-learn 0.24 or later
* OpenCV 4.5 or later

## Installation

To install the required libraries, follow the instructions in the [Conda Guide](https://github.com/siddhant-rajhans/detect_diabetic_retinopathy/blob/exp/conda_guide.md).

## Usage

To use the model, simply run the following commands:

```
conda install --yes --file requirements.txt
```
```
python app.py
```
```
streamlit run streamlit_app.py
```



This will load the model and start the prediction process.

## Model

The model used in this project is a convolutional neural network (CNN) that takes retinal images as input and outputs a probability distribution over the five classes of diabetic retinopathy.

## Training

The model was trained on a dataset of retinal images with diabetic retinopathy. The training process involved the following steps:

* Data preprocessing: The images were resized to 224x224 pixels and normalized to have zero mean and unit variance.
* Data augmentation: The images were augmented using random rotation, flipping, and cropping.
* Model training: The model was trained using the Adam optimizer and categorical cross-entropy loss function.

## Evaluation

The model was evaluated using the following metrics:

* Accuracy: The proportion of correctly classified images.
* Precision: The proportion of true positives among all positive predictions.
* Recall: The proportion of true positives among all actual positive images.
* F1-score: The harmonic mean of precision and recall.

## Conclusion

This project demonstrates the use of deep learning for diabetic retinopathy detection. The model achieved high accuracy and F1-score on the test dataset, indicating its potential for clinical use.
```
Note that you should replace `https://github.com/username/diabetic-retinopathy-detection/blob/main/conda_guide.md` with the actual link to your Conda guide.
