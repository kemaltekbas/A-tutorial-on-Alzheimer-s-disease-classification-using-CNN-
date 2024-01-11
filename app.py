from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from tensorflow.keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO

# Flask uygulamanını başlat
app = Flask(__name__)

# Modeli yükleyin
model = load_model('model.keras')

# Sınıf etiketlerini tanımlayın (sırasıyla modelinizin sınıflarına göre)
train_labels = ["MildDemented", "ModerateDemented","NonDemented", "VeryMildDemented"]

# API yolunu tanımlayın
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'base64_image' not in data:
            return jsonify({'error': 'No base64_image key in JSON data'})
        base64_image = data['base64_image']
        
        image_data = base64.b64decode(base64_image)
        image = Image.open(BytesIO(image_data))
        
        desired_size = (208, 176)
        image = image.resize(desired_size)
        image = image.convert('RGB')  # Görüntüyü 3 kanallı bir RGB görüntüsüne dönüştürün
        image_array = img_to_array(image)
        image_array = image_array.reshape((1,) + image_array.shape) / 255.0  # Resmi normalize edin ve boyutu yeniden şekillendirin
        print(image_array.shape)
        
        y_pred = model.predict(image_array)
        class_index = np.argmax(y_pred)
        prediction = train_labels[class_index]
        
        return jsonify({'prediction': prediction})
    
    except Exception as e:
        error_message = str(e)
        return jsonify({'error': error_message})


if __name__ == '__main__':
    app.run(debug=False, port=6000)
