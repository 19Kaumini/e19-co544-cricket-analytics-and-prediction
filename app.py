import os
from flask_cors import CORS, cross_origin
import pandas as pd
import numpy as np
from joblib import load
from flask import Flask, request, jsonify, render_template

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

# Load CSV files
venues_df = pd.read_csv("train_venues.csv")
player_vs_bowler_df = pd.read_csv("train_player_vs_bowler.csv")

class ClientApp:
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        model_path = 'models/model.pkl'
        return load(model_path)

    def predict(self, input_data):
        prediction = self.model.predict(input_data)
        return prediction

def derive_features(input_dic):
    bowler_types = ['Left arm Fast', 'Right arm Fast', 'Left arm Orthodox',
                    'Left arm Wrist spin', 'Right arm Legbreak', 'Right arm Offbreak']

    def get_venue_info():
        venue = input_dic['venue']
        if venue in venues_df['venue'].values:
            venue_mean = venues_df.loc[venues_df['venue'] == venue, 'total_mean']
            venue_first_bat_won_ratio = venues_df.loc[venues_df['venue'] == venue, 'first_bat_won_ratio']
            input_dic['venue_mean'] = venue_mean.values[0]
            input_dic['venue_first_bat_won_ratio'] = venue_first_bat_won_ratio.values[0]
        else:
            input_dic['venue_mean'] = venues_df['total_mean'].median()
            input_dic['venue_first_bat_won_ratio'] = venues_df['first_bat_won_ratio'].median()

    def get_runs_remain():
        if input_dic['innings'] == 2:
            return
        if input_dic['innings'] == 1:
            input_dic['runs_remain'] = input_dic['venue_mean'] - input_dic['current_team_total']

    def get_bowler_type_data():
        player_vs_bowler = player_vs_bowler_df[player_vs_bowler_df['batter'] == input_dic['batter']]
        player_vs_bowler.set_index("bowler_type", inplace=True)
        for bowler_type in bowler_types:
            if bowler_type in player_vs_bowler.index:
                input_dic[f'{bowler_type} Expected Runs'] = player_vs_bowler.loc[bowler_type, 'strike_rate'] / 100 * input_dic[bowler_type]
                input_dic[f'{bowler_type} Expected Wickets'] = input_dic[bowler_type] / player_vs_bowler.loc[bowler_type, 'deliveries_per_wicket']
                input_dic[f'{bowler_type} Strike Rate'] = player_vs_bowler.loc[bowler_type, 'strike_rate']
                input_dic[f'{bowler_type} Deliveries Per Wicket'] = player_vs_bowler.loc[bowler_type, 'deliveries_per_wicket']
            else:
                input_dic[f'{bowler_type} Expected Runs'] = 0
                input_dic[f'{bowler_type} Expected Wickets'] = 0
                input_dic[f'{bowler_type} Strike Rate'] = 0
                input_dic[f'{bowler_type} Deliveries Per Wicket'] = 0

    def get_expected():
        bowler_types_expected_wickets_cols = [bowler_type + " Expected Wickets" for bowler_type in bowler_types]
        bowler_types_expected_runs_cols = [bowler_type + " Expected Runs" for bowler_type in bowler_types]
        input_dic['expected_wickets'] = sum([input_dic[key] for key in bowler_types_expected_wickets_cols])
        input_dic['expected_runs'] = sum([input_dic[key] for key in bowler_types_expected_runs_cols])
        if input_dic['expected_wickets'] > 0:
            input_dic['expected_runs'] = input_dic['expected_runs'] / np.sqrt(input_dic['expected_wickets']) 

    get_venue_info()
    get_runs_remain()
    get_bowler_type_data()
    get_expected()

    return input_dic

@app.route('/')
@cross_origin()
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        form_data = {
            'batter': request.form['batter'],
            'innings': int(request.form['innings']),
            'wickets_fallen': int(request.form['wickets_fallen']),
            'bowling_team': request.form['bowling_team'],
            'batting_team': request.form['batting_team'],
            'toss_winner': int(request.form['toss_winner']),
            'runs_remain': float(request.form['runs_remain']),
            'first_ball': int(request.form['first_ball']),
            'current_team_total': int(request.form['current_team_total']),
            'is_powerplay': request.form['is_powerplay'] == 'true',
            'Left arm Fast': float(request.form.get('Left arm Fast', 0)),
            'Left arm Orthodox': float(request.form.get('Left arm Orthodox', 0)),
            'Left arm Wrist spin': float(request.form.get('Left arm Wrist spin', 0)),
            'Right arm Fast': float(request.form.get('Right arm Fast', 0)),
            'Right arm Legbreak': float(request.form.get('Right arm Legbreak', 0)),
            'Right arm Offbreak': float(request.form.get('Right arm Offbreak', 0)),
            'venue': request.form['venue']
        }

        # Derive features
        derived_features = derive_features(form_data)

        # Convert to DataFrame for prediction
        data = pd.DataFrame([derived_features])

        # Keep only the columns that the model expects
        expected_columns = ['batter', 'innings', 'wickets_fallen', 'bowling_team', 'batting_team',
                            'toss_winner', 'runs_remain', 'first_ball', 'current_team_total',
                            'is_powerplay', 'Left arm Fast', 'Left arm Orthodox', 'Left arm Wrist spin',
                            'Right arm Fast', 'Right arm Legbreak', 'Right arm Offbreak', 'venue',
                            'venue_mean', 'venue_first_bat_won_ratio', 'Left arm Fast Expected Runs',
                            'Left arm Fast Expected Wickets', 'Left arm Fast Strike Rate', 
                            'Left arm Fast Deliveries Per Wicket', 'Right arm Fast Expected Runs',
                            'Right arm Fast Expected Wickets', 'Right arm Fast Strike Rate', 
                            'Right arm Fast Deliveries Per Wicket', 'Left arm Orthodox Expected Runs',
                            'Left arm Orthodox Expected Wickets', 'Left arm Orthodox Strike Rate', 
                            'Left arm Orthodox Deliveries Per Wicket', 'Left arm Wrist spin Expected Runs',
                            'Left arm Wrist spin Expected Wickets', 'Left arm Wrist spin Strike Rate',
                            'Left arm Wrist spin Deliveries Per Wicket', 'Right arm Legbreak Expected Runs',
                            'Right arm Legbreak Expected Wickets', 'Right arm Legbreak Strike Rate',
                            'Right arm Legbreak Deliveries Per Wicket', 'Right arm Offbreak Expected Runs',
                            'Right arm Offbreak Expected Wickets', 'Right arm Offbreak Strike Rate',
                            'Right arm Offbreak Deliveries Per Wicket', 'expected_wickets', 'expected_runs']
        data = data[expected_columns]

        client_app = ClientApp()
        predictions = client_app.predict(data)
        return jsonify(predictions.tolist())
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
