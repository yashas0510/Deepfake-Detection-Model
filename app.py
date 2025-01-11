
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load the model
model = load_model("C:/Users/yashas/Downloads/archive/fine_tuned_model.h5")
model_threshold = 0.5  # Adjust if necessary

# Preprocess image
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Route to serve the frontend
@app.route('/')
def home():
    return render_template('index.html')

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        image = Image.open(file.stream)
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)[0][0]
        if prediction >= 0.5 + (model_threshold / 2):  # Slightly higher threshold for "fake"
            label = "fake , if results do not match the model might be overfit/underfit"
        elif prediction <= 0.5 - (model_threshold / 2):  # Slightly lower threshold for "real"
            label = "real , if results do not match the model might be overfit/underfit"
        else:
            label = "uncertain , if results do not match the model might be overfit/underfit"  # Optional: Introduce an "uncertain" category if close to the threshold
            ##confidence = abs(prediction - 0.5) * 2  # Scale to 0-1
        return jsonify({'label': label })
               
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
