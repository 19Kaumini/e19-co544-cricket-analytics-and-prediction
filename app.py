from MLOps.NRR_prediction import load_data, data_preparation, train_rf, evaluate_model
import pickle
import numpy as np
from flask import Flask, request, jsonify
import os
from flask_cors import CORS, cross_origin
import pandas as pd
from joblib import load
from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
import pandas as pd
from joblib import load

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        model_path = 'models/model.pkl'
        return load(model_path)

    def predict(self, input_data):
        # Assuming input_data is a pandas DataFrame
        prediction = self.model.predict(input_data)
        return prediction

@app.route('/')
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        json_data = request.get_json()
        data = pd.DataFrame(json_data)
        client_app = ClientApp()
        predictions = client_app.predict(data)
        return jsonify(predictions.tolist())
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
