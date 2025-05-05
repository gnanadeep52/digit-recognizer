# 🖌️ Handwritten Digit Recognizer (Tkinter + TensorFlow)

A simple desktop application that recognizes handwritten digits using a Convolutional Neural Network (CNN) trained on the MNIST dataset. Built with Python, TensorFlow, and Tkinter.

---

## ✨ Features

- Draw a digit using your mouse
- Predict the digit in real-time using a trained model
- See the prediction confidence percentage
- Lightweight and easy to run locally

---

## 🧠 Model Details

- CNN trained on the MNIST dataset (60,000 digits)
- Achieves ~98% accuracy
- Architecture:
  - Conv2D → MaxPooling2D → Flatten → Dense → Output

---

## 🚀 How to Run

Install required libraries with:

```bash
pip install -r requirements.txt

```bash
python train_model.py

```bash
python digit_gui_app.py


