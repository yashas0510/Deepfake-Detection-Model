import tensorflow as tf
from keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Paths to datasets
train_dir = "C:/Users/yashas/Downloads/archive/dataprep/train"
val_dir = "C:/Users/yashas/Downloads/archive/dataprep/val"
test_dir = "C:/Users/yashas/Downloads/archive/dataprep/test"

# Image parameters
img_size = (224, 224)
batch_size = 32

# Data Augmentation and Generators
train_datagen = ImageDataGenerator(rescale=1.0/255.0, rotation_range=20, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_gen = train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')
val_gen = val_datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')
test_gen = test_datagen.flow_from_directory(test_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')

# Load Pre-trained Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze pre-trained layers

# Add Custom Layers
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
checkpoint = ModelCheckpoint(
    "video_classification_model_best.keras",  # Change to .keras format
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=3,  # Stop if no improvement for 3 epochs
    verbose=1,
    restore_best_weights=True
)

# Train the Model with Callbacks
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=[checkpoint, early_stopping]
)

# Evaluate the Model
test_loss, test_acc = model.evaluate(test_gen)
print(f"Test Accuracy: {test_acc:.2f}")

# Save the Final Model
model.save("video_classification_model_final.h5")
print("Model saved successfully!")
