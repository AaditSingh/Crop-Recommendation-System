import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model  # Import load_model from Keras

app = Flask(__name__)

# Load the Keras model
model = load_model('output.h5')

# Define the classes mapping
labels = ['Apple', 'Banana', 'Blackgram', 'Chickpea', 'Coconut', 'Coffee',
       'Cotton', 'Grapes', 'Jute', 'Kidneybeans', 'Lentil', 'Maize',
       'Mango', 'Mothbeans', 'Mungbean', 'Muskmelon', 'Orange', 'Papaya',
       'Pigeonpeas', 'Pomegranate', 'Rice', 'Watermelon']

@app.route('/')
def home():
    return render_template('home.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     int_features = [int(x) for x in request.form.values()]
#     final_features = np.array(int_features).reshape(1, -1)
#     prediction_id = np.argmax(model.predict(final_features), axis=-1)  # Get the predicted class ID
#     prediction_class = classes[prediction_id[0]]  # Map ID to class
#     return render_template('home.html', prediction_text='Predicted Crop: {}'.format(prediction_class))

images = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg', '7.jpg', '8.jpg', '9.jpg', '10.jpg', '11.jpg', '12.jpg', '13.jpg', '14.jpg', '15.jpg','16.jpg','17.jpg','18.jpg','19.jpg','20.jpg','21.jpg','22.jpg']


@app.route('/predict', methods=['POST'])
def predict():
    feature = [int(x) for x in request.form.values()]
    feature = np.array(feature).reshape(1, -1)
    prediction = np.argmax(model.predict(feature))
    predicted_weather = labels[prediction]
    image_url = f"../static/img/{images[prediction]}"
    print(image_url)
    return render_template('home.html',prediction_text='Recommended Crop: {}'.format(predicted_weather), image_url=image_url)

# def predict():
#     feature = [int(x) for x in request.form.values()]
#     feature = np.array(feature).reshape(1, -1)
#     prediction = np.argmax(model.predict(feature))
#     predicted_weather = labels[prediction]
#     image_url = f"../static/img/{images[prediction]}"
#     print(image_url)
#     return render_template('home.html',prediction_text='Predicted Weather: {}'.format(predicted_weather), image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)
