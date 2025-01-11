# Deepfake-Detection-Model AKA Image Classification Web App

## **Project Overview**
This project is a web application that classifies uploaded images as either "real" or "fake" using a fine-tuned deep learning model. It integrates a Flask backend with a dynamic and responsive frontend, providing users with an easy-to-use interface for image classification.

---

## **Features**
1. **Image Upload**: Users can upload images directly from their devices.
2. **Real-Time Predictions**: The model predicts whether the image is real or fake and provides a confidence score.
3. **Seamless Integration**: A Flask backend serves the fine-tuned model and the frontend interface.
4. **Quality UI**: A responsive and user-friendly interface built with HTML, CSS, and JavaScript.
5. **Error Handling**: Handles incorrect or missing file uploads gracefully.

---

## **Project Structure**

```
project/
|
├── app.py                   # Flask backend application
├── fine_tuned_model.h5      # Pre-trained and fine-tuned deep learning model
├── templates/               # Folder containing HTML files
│   └── index.html           # Main HTML file for the frontend
├── static/                  # Folder containing static files (CSS, JS, images)
│   ├── css/
│   │   └── styles.css       # CSS file for styling
│   ├── js/
│   │   └── script.js        # JavaScript file for frontend logic
│   └── images/              # (Optional) Folder for images or assets
└── README.md                # Documentation for the project
```

---

## **Setup and Installation**

### **1. Prerequisites**
Ensure you have the following installed:
- Python 3.8+
- TensorFlow 2.0+
- Flask
- h5py
- Pillow

### **2. Installation Steps**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/image-classification-app.git
   cd image-classification-app
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the `fine_tuned_model.h5` file in the root directory.

4. Run the Flask application:
   ```bash
   python app.py
   ```

5. Open your browser and visit:
   ```
   http://127.0.0.1:5000/
   ```

---

## **How to Use**
1. Open the web app in your browser.
2. Upload an image file using the upload form.
3. Click the **Classify** button.
4. View the prediction and confidence score displayed below the form.

---

## **Model Performance**
The model is fine-tuned for distinguishing between real and fake images. However, it’s important to note:
- **Overfitting or Underfitting**: The model might misclassify images due to overfitting (memorizing training data) or underfitting (not learning sufficient patterns). For optimal performance, retrain the model with a balanced and diverse dataset.
- **Threshold Adjustments**: The default classification threshold is set to 0.5. You can adjust this based on the performance of the model on your dataset.

---

## **Troubleshooting**
1. **"404 Not Found" Error**:
   - Ensure the `templates/` and `static/` folders are correctly structured.
   - Check the file paths in `app.py`.

2. **Model Not Loading**:
   - Verify the path to `fine_tuned_model.h5` in the `load_model` function.

3. **Misclassifications**:
   - Revisit the training process and adjust hyperparameters.
   - Include more diverse data in training to improve generalization.

---

## **Future Enhancements**
- **Model Improvements**:
  - Implement advanced architectures like ResNet or EfficientNet.
  - Use transfer learning for better accuracy.
- **Frontend Enhancements**:
  - Add support for drag-and-drop image uploads.
  - Display a history of recent predictions.
- **Deployment**:
  - Host the app on platforms like AWS, Azure, or Heroku.

---

## **Contributing**
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

---

## **License**
This project is licensed under the MIT License. See the LICENSE file for details.

---

## **Acknowledgments**
- TensorFlow and Keras for the deep learning framework.
- Flask for the lightweight web framework.
- The open-source community for inspiration and support.

---

**Disclaimer**: This project is for educational purposes and demonstrates the integration of machine learning with web applications. Model predictions should not be used for critical decision-making without thorough validation.

