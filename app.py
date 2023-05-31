from flask import Flask, render_template, request
import requests
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)
model = load_model('/Applications/XAMPP/xamppfiles/htdocs/ml_cars/ml/modelcar')  # Ruta a tu modelo entrenado

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        url = request.form['url']
        image = Image.open(requests.get(url, stream=True).raw)
        image = image.resize((224, 224))
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        prediction = model.predict(image)
        # Realiza la lógica necesaria para obtener la predicción de tu modelo

        prediction=procesar_prediccion(prediction) 
    except:
        prediction=""
    return render_template('index.html', prediction=prediction)


def procesar_prediccion(prediction):

    # Obtén el índice del valor más alto

    predicted_class = np.argmax(prediction)

    print(predicted_class)

    if predicted_class == 0:

        return "es un carro"

    if predicted_class == 1:

        return "es un formula 1"

    if predicted_class == 2:

        return "es una moto"
    if predicted_class == 3:

        return "es una troca"
    

if __name__ == '__main__':
    app.run(debug=True, port=5001)
