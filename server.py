from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
from keras.models import Sequential
from tensorflow.keras.layers import *
from PIL import Image
import numpy as np
import keras

# Enable CORS
app = Flask(__name__)
CORS(app)

# Load and prepare the model
base_model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False

model = keras.models.Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

# Load model weights
model.load_weights('transferlearnin.weights.h5')  # Ensure this path is correct

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file.content_type.startswith('image/'):
        return jsonify({'error': 'Uploaded file is not an image.'}), 400

    try:
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 150, 150, 3)

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        class_labels = {0: "Cat", 1: "Dog"}
        result = class_labels[predicted_class]

        return jsonify({'prediction': result, 'probabilities': prediction[0].tolist()})
    except Exception as e:
        print(e)  # Log error
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
