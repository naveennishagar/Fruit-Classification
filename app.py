from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Ensure the uploads folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Fruit names corresponding to class labels
fruit_classes = {
    0: 'Apple',
    1: 'Banana',
    2: 'Grapes',
    3: 'Mango',
    4: 'Strawberry',
    5: 'Other'
}

# Load the model only once when the app starts
model = None
try:
    model = tf.keras.models.load_model(r"C:\Users\Lenovo\Desktop\my_model1.h5")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# Define a route for the main page
@app.route('/')
def home():
    return render_template('fruit.html')  # Your HTML page

# Define a route to handle file uploads
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Save the uploaded file to the uploads folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Load the image for processing
        image = cv2.imread(filepath)
        if image is None:
            os.remove(filepath)  # Clean up
            return jsonify({'error': 'Invalid image format'}), 400

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Preprocess the image to the model's input shape
        image_resized = cv2.resize(image, (224, 224))  # Assuming model input size is 224x224
        image_array = np.expand_dims(image_resized, axis=0)  # Add batch dimension
        image_array = image_array / 255.0  # Normalize the image

        # Make predictions using the loaded model
        if model is not None:
            prediction = model.predict(image_array)
            class_label = int(np.argmax(prediction, axis=1)[0])  # Get predicted class as integer
            confidence = float(np.max(prediction) * 100)  # Get confidence percentage as float

            # If confidence is less than 80%, classify as "Other"
            if confidence < 85:
                predicted_fruit = fruit_classes[5]  # Class 5 = "Other"
            else:
                # Get the predicted class fruit name
                predicted_fruit = fruit_classes.get(class_label, 'Unknown')

            # Delete the uploaded file after processing
            os.remove(filepath)

            return jsonify({
                'prediction': predicted_fruit,
                'confidence': confidence
            })
        else:
            return jsonify({'error': 'Model not loaded'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=5002)  # Change port if needed
