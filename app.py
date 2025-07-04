from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from io import BytesIO

app = Flask(__name__)

model = load_model('gaussian_filtered_images/model.h5')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_data = request.files['image'].read()
        image = Image.open(BytesIO(image_data))
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        prediction = model.predict(image_array.reshape((1, 224, 224, 3)))
        return jsonify({'prediction': str(np.argmax(prediction))})
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)