import os
from flask_cors import CORS, cross_origin
import pandas as pd
from joblib import load
from flask import Flask, request, jsonify, render_template

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
        data = request.json

        form_data = {
            'batter': data['batter'],
            'innings': int(data['innings']),
            'wickets_fallen': int(data['wickets_fallen']),
            'bowling_team': data['bowling_team'],
            'batting_team': data['batting_team'],
            'toss_winner': data['toss_winner'],
            'runs_remain': int(data['runs_remain']),
            'first_ball': int(data['first_ball']),
            'current_team_total': int(data['current_team_total']),
            'is_powerplay': int(data['is_powerplay']),
            'Left arm Fast': int(data.get('Left arm Fast', 0)),
            'Left arm Orthodox': int(data.get('Left arm Orthodox', 0)),
            'Left arm Wrist spin': int(data.get('Left arm Wrist spin', 0)),
            'Right arm Fast': int(data.get('Right arm Fast', 0)),
            'Right arm Legbreak': int(data.get('Right arm Legbreak', 0)),
            'Right arm Offbreak': int(data.get('Right arm Offbreak', 0)),
            'venue': data['venue']
        }

        data_df = pd.DataFrame([form_data])

        expected_columns = ['batter', 'innings', 'wickets_fallen', 'bowling_team', 'batting_team',
                            'toss_winner', 'runs_remain', 'first_ball', 'current_team_total',
                            'is_powerplay', 'Left arm Fast', 'Left arm Orthodox', 'Left arm Wrist spin',
                            'Right arm Fast', 'Right arm Legbreak', 'Right arm Offbreak', 'venue']
        data_df = data_df[expected_columns]

        client_app = ClientApp()
        predictions = client_app.predict(data_df)
        return jsonify(predictions.tolist())
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
