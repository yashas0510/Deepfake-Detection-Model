import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os
import matplotlib.pyplot as plt

# Paths
model_path = "C:/Users/yashas/Downloads/archive/video_classification_model_final.h5"  # Replace with your model's path
train_dir = "C:/Users/yashas/Downloads/archive/dataprep/train"  # Replace with your training data path
val_dir = "C:/Users/yashas/Downloads/archive/dataprep/val"  # Replace with your validation data path
test_dir = "C:/Users/yashas/Downloads/archive/dataprep/test"  # Replace with your test data path
image_size = (224, 224)

# Load Model
model = load_model(model_path)
print("Model loaded successfully!")

# Data Generators with Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode="nearest"
)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=image_size, batch_size=32, class_mode='binary'
)
val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=image_size, batch_size=32, class_mode='binary'
)
test_gen = test_datagen.flow_from_directory(
    test_dir, target_size=image_size, batch_size=32, class_mode='binary', shuffle=False
)

# Fine-Tuning Model
print("\n--- Fine-Tuning Model ---")
base_model = model.layers[0]  # Assuming the first layer is the base model
base_model.trainable = True  # Unfreeze the base model

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_gen, validation_data=val_gen, epochs=5)

# Save Fine-Tuned Model
model.save("fine_tuned_model.h5")
print("Fine-tuned model saved successfully!")

# Plot Training vs Validation Loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Confusion Matrix After Fine-Tuning
print("\n--- Evaluating Fine-Tuned Model ---")
predictions = model.predict(test_gen)
y_pred = (predictions > 0.5).astype(int)
y_true = test_gen.classes

print("Confusion Matrix After Fine-Tuning:\n", confusion_matrix(y_true, y_pred))
print("\nClassification Report After Fine-Tuning:\n", classification_report(y_true, y_pred, target_names=test_gen.class_indices.keys()))

# Threshold Adjustment
def predict_with_threshold(img_path, model, threshold=0.5):
    """
    Predict whether an image is 'Real' or 'Manipulated' using a threshold.
    """
    img = load_img(img_path, target_size=image_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    result = "Manipulated" if prediction > threshold else "Real"
    return result, prediction

# Test with Images
test_images = [
    "C:/Users/yashas/Downloads/archive/frames/real/28__walk_down_hall_angry_frame0027.jpg",  # Replace with a real image path
    "C:/Users/yashas/Downloads/archive/frames/manipulated/28_16__walking_down_street_outside_angry__6DWLCU6T_frame0041.jpg",  # Replace with a fake image path
]

threshold = 0.5  # Adjust based on results
print("\n--- Testing with Images ---")
for img_path in test_images:
    result, confidence = predict_with_threshold(img_path, model, threshold)
    print(f"Image: {img_path}")
    print(f"Prediction: {result} (Confidence: {confidence:.2f})\n")

# Adjust Threshold Dynamically
def optimize_threshold(test_gen, model):
    """
    Dynamically adjust the threshold to maximize classification accuracy on the test set.
    """
    predictions = model.predict(test_gen)
    best_threshold = 0.5
    best_accuracy = 0

    for threshold in np.arange(0.1, 1.0, 0.05):
        y_pred = (predictions > threshold).astype(int)
        accuracy = np.mean(y_pred.flatten() == test_gen.classes)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    print(f"Optimal Threshold: {best_threshold}, Accuracy: {best_accuracy:.2f}")
    return best_threshold

optimal_threshold = optimize_threshold(test_gen, model)
print(f"Use optimal threshold {optimal_threshold:.2f} for improved predictions.")
