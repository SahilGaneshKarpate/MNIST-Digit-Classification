# ==========================================
# Step 1 : Import Required Libraries
# ==========================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# ==========================================
# Step 2 : Load MNIST Dataset
# ==========================================

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Training Images Shape :", X_train.shape)
print("Training Labels Shape :", y_train.shape)

print("Testing Images Shape :", X_test.shape)
print("Testing Labels Shape :", y_test.shape)

# ==========================================
# Step 3 : Understand Dataset Shape
# ==========================================

print("\nDataset Information")
print("Training Images :", X_train.shape)
print("Training Labels :", y_train.shape)
print("Testing Images :", X_test.shape)
print("Testing Labels :", y_test.shape)

# ==========================================
# Step 4 : Visualize Sample Image
# ==========================================

plt.imshow(X_train[0], cmap='gray')
plt.title(f"Label : {y_train[0]}")
plt.axis("off")
plt.show()

# ==========================================
# Step 5 : Normalize the Dataset
# ==========================================

X_train = X_train / 255.0
X_test = X_test / 255.0

print("\nNormalization Completed")

print("Minimum Pixel Value :", X_train.min())
print("Maximum Pixel Value :", X_train.max())

# ==========================================
# Step 6 : Build Neural Network Model
# ==========================================

model = tf.keras.Sequential([

    tf.keras.layers.Flatten(input_shape=(28, 28)),

    tf.keras.layers.Dense(128, activation='relu'),

    tf.keras.layers.Dense(10, activation='softmax')

])

print("\nNeural Network Model Created Successfully")

model.summary()

# ==========================================
# Step 7 : Compile the Model
# ==========================================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Compiled Successfully")

# ==========================================
# Step 8 : Train the Model
# ==========================================

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_test, y_test)
)

print("\nModel Training Completed Successfully")

# ==========================================
# Step 9 : Evaluate the Model
# ==========================================

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("\nModel Evaluation Completed")
print("Test Loss :", test_loss)
print("Test Accuracy :", test_accuracy)
print("Test Accuracy (%):", test_accuracy * 100)

# ==========================================
# Step 10 : Make Predictions
# ==========================================

predictions = model.predict(X_test)

# ==========================================
# Step 11 : Predict a Single Image
# ==========================================

index = 0

predicted_digit = np.argmax(predictions[index])

# ==========================================
# Step 12 : Display Prediction Result
# ==========================================

print("\nActual Digit :", y_test[index])
print("Predicted Digit :", predicted_digit)

plt.imshow(X_test[index], cmap='gray')
plt.title(f"Predicted : {predicted_digit}")
plt.axis("off")
plt.show()

# ==========================================
# Step 13 : Save the Trained Model
# ==========================================

model.save("model.h5")

print("\nModel Saved Successfully as model.h5")