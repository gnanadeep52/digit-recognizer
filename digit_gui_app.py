import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf

# Load trained CNN model
model = tf.keras.models.load_model("digit_model.h5")

# GUI App
class DrawDigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🖌️ Handwritten Digit Recognizer")

        # Canvas to draw
        self.canvas = tk.Canvas(root, width=200, height=200, bg='white')
        self.canvas.pack(pady=10)

        # Buttons
        button_frame = tk.Frame(root)
        button_frame.pack()

        tk.Button(button_frame, text="Predict", command=self.predict_digit, width=10).grid(row=0, column=0, padx=5)
        tk.Button(button_frame, text="Clear", command=self.clear_canvas, width=10).grid(row=0, column=1, padx=5)

        # Label to show result
        self.label = tk.Label(root, text="", font=("Helvetica", 18))
        self.label.pack(pady=10)

        # PIL image for drawing
        self.image = Image.new("L", (200, 200), color=255)
        self.draw = ImageDraw.Draw(self.image)

        # Bind drawing to mouse
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black')
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 200, 200], fill=255)
        self.label.config(text="")

    def predict_digit(self):
        # Resize and invert image
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img)
        img = np.array(img).astype("float32") / 255.0
        img = img.reshape(1, 28, 28, 1)  # CNN expects 4D shape

        # Predict
        prediction = model.predict(img)
        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # Show result
        self.label.config(text=f"Prediction: {digit}  ({confidence:.2f}%)")

# Run app
if __name__ == "__main__":
    root = tk.Tk()
    app = DrawDigitApp(root)
    root.mainloop()
