# Diabetic Retinopathy Detection Model

## Project Overview
This project presents a **binary deep learning model** for detecting **Diabetic Retinopathy (DR)** from retinal fundus images. The system classifies images into **DR** or **No_DR** using a convolutional neural network trained on labeled retinal images, with the objective of supporting **early-stage screening** rather than clinical diagnosis.

---

## Model Architecture
- **Model Type:** Custom Convolutional Neural Network (CNN)
- **Input Shape:** (224 × 224 × 3)
- **Output Classes:** 2 (DR, No_DR)
- **Convolutional Blocks:** 5
- **Dense Layers:** 2
- **Total Parameters:** ~17.7 million

---

## Training Configuration
- **Epochs:** 30
- **Batch Size:** 32
- **Optimizer:** Adam
- **Initial Learning Rate:** 0.001
- **Loss Function:** Categorical Cross-Entropy
- **Framework:** TensorFlow / Keras

---

## Dataset Summary
- **Training Samples:** 2,076
- **Validation Samples:** 531
- **Test Samples:** 231
- **Total Samples:** 2,838
- **Classes:**
  - `DR` – Diabetic Retinopathy present
  - `No_DR` – No Diabetic Retinopathy

---

## Final Test Set Performance

| Metric     | Value   |
|------------|---------|
| Accuracy   | **96.10%** |
| Precision  | **96.10%** |
| Recall     | **96.10%** |
| F1-Score   | **96.10%** |
| AUC        | **97.62%** |

### Confusion Matrix Summary
- **True Positives (DR → DR):** 106  
- **False Negatives (DR → No_DR):** 7  
- **False Positives (No_DR → DR):** 2  
- **True Negatives (No_DR → No_DR):** 116  

These results indicate **high sensitivity and low false-positive rates**, which is desirable for medical screening tasks.

---

## Training Behavior
- **Final Training Accuracy:** 93.50%
- **Final Validation Accuracy:** 94.35%
- **Training Loss:** 0.1785
- **Validation Loss:** 0.1493
- **Train–Validation Gap:** 0.85%

No significant overfitting was observed.

---

## Model Usage
- The trained model is saved as **`best_dr_model.h5`**
- Intended for **research and educational purposes only**
- Suitable as a **pre-screening support system**, not a clinical diagnostic tool

---

## Key Notes
- Performance metrics are reported on a **held-out test set**
- Results may vary on external datasets due to imaging and demographic differences
- Further validation is required before real-world deployment

---

## Future Improvements
- External dataset validation
- Cross-dataset generalization testing
- Explainability methods (Grad-CAM)
- Multi-class DR severity grading
