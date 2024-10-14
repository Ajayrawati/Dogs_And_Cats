import keras
from keras.models import load_model
from keras.models import Sequential
from tensorflow.keras.layers import *
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import keras

base_model = keras.applications.vgg16.VGG16(weights='imagenet',include_top=False,input_shape=(150,150,3))
base_model.trainable=False

model = keras.models.Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2,activation='softmax'))


def predict_cat_or_dog(model, img_path):
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)

    predicted_class = np.argmax(predictions, axis=1)[0]
    class_labels = {0: "Cat", 1: "Dog"}
    result = class_labels[predicted_class]

    return result, predictions[0]


model.load_weights('transferlearnin.weights.h5')

# Example usage
img_path = 'cat.jpeg'  # Replace with your image path
result, probabilities = predict_cat_or_dog(model, img_path)
print(f'The image is predicted to be: {result}')
print(f'Prediction probabilities: {probabilities}')
