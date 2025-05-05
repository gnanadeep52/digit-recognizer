# import tensorflow as tf
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten
# from tensorflow.keras.utils import to_categorical
# import matplotlib.pyplot as plt
# import numpy as np

# # Load data
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train / 255.0
# x_test = x_test / 255.0
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

# # Build model
# # model = Sequential([
# #     Flatten(input_shape=(28, 28)),
# #     Dense(128, activation='relu'),
# #     Dense(64, activation='relu'),
# #     Dense(10, activation='softmax')
# # ])
# model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=(28, 28, 1)),
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Train model
# # model.fit(x_train, y_train, epochs=5, batch_size=32)

# # Evaluate
# loss, acc = model.evaluate(x_test, y_test)
# print(f"\nTest Accuracy: {acc:.4f}")

# # Save the model
# model.save("digit_model.h5")
# # model.save("digit_model.keras")
# # Optional: Predict one digit
# index = 0
# prediction = np.argmax(model.predict(x_test)[index])
# actual = np.argmax(y_test[index])
# plt.imshow(x_test[index], cmap='gray')
# plt.title(f"Predicted: {prediction}, Actual: {actual}")
# plt.axis('off')
# plt.show()


import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load and preprocess the MNIST dataset
(x_train, y_train), _ = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# Compile and train the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=3)

# Save the model
model.save("digit_model.h5")
print("✅ Model saved as digit_model.h5")
