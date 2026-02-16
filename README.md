# 🩺 Diabetic Retinopathy Severity Classification using EfficientNetV2-S & Ordinal Regression

An end-to-end deep learning system for automated **5-class Diabetic Retinopathy (DR) severity grading** using EfficientNetV2-S with ordinal regression, Grad-CAM interpretability, and a Streamlit-based web application.

---

## 📌 Problem Statement

Diabetic Retinopathy (DR) is a progressive eye disease and one of the leading causes of blindness worldwide. Early and accurate severity grading is crucial for timely treatment.

Traditional diagnosis requires expert ophthalmologists and is time-consuming. This project aims to assist clinicians by automating DR severity classification using deep learning.

---

## 🎯 Classes

The model predicts the following **ordinal severity levels**:

0 – No DR  
1 – Mild DR  
2 – Moderate DR  
3 – Severe DR  
4 – Proliferative DR  

---

## 🧠 Approach

- Transfer Learning using **EfficientNetV2-S**
- Ordinal Regression for ordered multi-class classification
- Five-class DR severity prediction
- PyTorch-based training pipeline
- Evaluation using Accuracy, F1-score, and Confusion Matrix
- Grad-CAM for visual explanation of predictions
- Streamlit web app for inference

Ordinal regression is used instead of standard softmax classification to better model the natural ordering between disease stages.

---

## 🛠 Tech Stack

- Python  
- PyTorch  
- EfficientNetV2-S  
- OpenCV  
- NumPy / Pandas  
- scikit-learn  
- Matplotlib  
- Streamlit  

---

## 📂 Project Structure
