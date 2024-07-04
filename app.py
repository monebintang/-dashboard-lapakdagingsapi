from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('cmknn_model.pkl')

def extract_glcm_features(img):
    graycom = graycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)
    features = {
        'contrast': graycoprops(graycom, 'contrast'),
        'energy': graycoprops(graycom, 'energy'),
        'homogeneity': graycoprops(graycom, 'homogeneity'),
        'correlation': graycoprops(graycom, 'correlation')
    }
    return features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diunggah'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400

    img = Image.open(file.stream)
    img = img.resize((256, 256))
    img_array = np.asarray(img)
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    features = extract_glcm_features(img_gray)
    feature_array = np.array([features[key].flatten() for key in features]).flatten()
    prediction = model.predict([feature_array])

    label_map = {0: 'Segar', 1: 'Tidak Segar', 2: 'Busuk', 3: 'Bukan Gambar Daging Sapi'}
    result = label_map[prediction[0]]

    if result == 'Bukan Gambar Daging Sapi':
        return jsonify({'prediction': 'Gambar yang diunggah tidak dikenali sebagai gambar daging sapi'}), 400

    recommendation = ""
    if result == 'Segar':
        recommendation = "Daging ini layak untuk dikonsumsi."
    elif result == 'Tidak Segar':
        recommendation = "Daging ini masih layak untuk dikonsumsi, namun disarankan untuk memilih daging yang lebih segar dengan warna lebih merah."
    elif result == 'Busuk':
        recommendation = "Daging ini sudah tidak layak untuk dikonsumsi. Silakan memilih daging yang segar untuk kesehatan Anda."

    return jsonify({'prediction': result, 'recommendation': recommendation})

if __name__ == '__main__':
    app.run(debug=True)
