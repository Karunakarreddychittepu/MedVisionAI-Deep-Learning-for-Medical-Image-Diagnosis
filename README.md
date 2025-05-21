# 🧬 MedVisionAI – Deep Learning for Medical Image Diagnosis

This project uses CNNs to analyze medical images and detect diseases like pneumonia and COVID-19. It includes data preprocessing, training models, evaluation metrics, and Grad-CAM visualization for explainability.

---

## 📦 Key Features

- 📁 Handles real-world X-ray/CT/MRI image data
- 🧠 Models: CNN, VGG16, ResNet, EfficientNet
- 📊 Metrics: Accuracy, AUC, Confusion Matrix
- 🔥 Explainability: Grad-CAM visualizations
- 🌐 Optional Web Demo (Streamlit)

---

## 🧠 Model Results

| Model      | Accuracy | AUC  | F1-score |
|------------|----------|------|----------|
| CNN (custom) | 88.2%    | 0.91 | 0.89     |
| VGG16       | 93.1%    | 0.96 | 0.94     |
| ResNet50    | 94.7%    | 0.97 | 0.95     |

---

## 🖼️ Grad-CAM Output

*Visual explanation of model's focus on infected areas*

![Grad-CAM](images/gradcam_sample.png)

---

## 🚀 Run Project

```bash
git clone https://github.com/yourusername/MedVisionAI.git
cd MedVisionAI
pip install -r requirements.txt
python scripts/train_model.py
