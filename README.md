# 🧠 AI vs Real Image Detection

A deep learning project to classify images as **AI-generated** or **real**.

## 🚀 Features
- ResNet50 Transfer Learning
- Grad-CAM Visualization 🔥
- Streamlit Web App
- Confidence Score Display

## 🧠 Model
- Architecture: ResNet50
- Input Size: 224x224
- Binary Classification

## 📊 Results
- Accuracy: ~75–85%
- Balanced dataset
- Confusion matrix evaluation

## 🔥 Grad-CAM
Visualizes where the model is focusing during prediction.

## 🖥️ Run Locally
```bash
pip install -r requirements.txt
python train.py
python evaluate.py
streamlit run app.py
