import os
import tensorflow

from tensorflow.keras.models import load_model

model_path = "video_classification_model.h5"
if os.path.exists("C:/Users/yashas/Downloads/archive/video_classification_model_final.h5"):
    model = load_model("C:/Users/yashas/Downloads/archive/video_classification_model_final.h5")
    print("Model loaded successfully!")
else:
    print(f"Model file not found: {"C:/Users/yashas/Downloads/archive/video_classification_model_final.h5"}")
