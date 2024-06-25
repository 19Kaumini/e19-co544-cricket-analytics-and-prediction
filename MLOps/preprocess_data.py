import pandas as pd

def preprocess_NRR_data():
    df = pd.read_csv("Data/selected_data/processed_data.csv")
    # check winning_team column and if it is "no result" then remove the row
    df = df[df["winning_team"] != "no result"]

    # group data by match_id
    grouped = df.groupby("match_id")

    # get net run rate for each team
    net_run_rate = {}
    for match_id, group in grouped:
        team1 = group[group["innings"] == 1]
        team2 = group[group["innings"] == 2]

        team1_runs = team1["final_team_total"].iloc[0]
        team2_runs = team2["final_team_total"].iloc[0]
        
        team1_overs = team1["over"].iloc[-1] + (team1["delivery"].iloc[-1] / 6)
        team2_overs = team2["over"].iloc[-1] + (team2["delivery"].iloc[-1] / 6)
        

        # check if team1 is all out
        if team1["wickets_fallen"].iloc[-1] == 10:
            team1_overs = 20

        # check if team2 is all out
        if team2["wickets_fallen"].iloc[-1] == 10:
            team2_overs = 20

        team1_net_run_rate = (team1_runs / team1_overs) - (team2_runs / team2_overs)
        team2_net_run_rate = (team2_runs / team2_overs) - (team1_runs / team1_overs)

        # print(f"Match ID: {match_id}")
        # print(f"Team 1: {team1_net_run_rate}")
        # print(f"Team 2: {team2_net_run_rate}")
        # print()


        # introduce net_run_rate column for original dataframe df
        for index, row in group.iterrows():
            if row["innings"] == 1:
                df.at[index, "net_run_rate"] = team1_net_run_rate
            else:
                df.at[index, "net_run_rate"] = team2_net_run_rate

    df.to_csv("Data/selected_data/processed_data_NRR.csv", index=False)

def All_Batters_NRR():
    data = pd.read_csv("Data/selected_data/processed_data_NRR.csv")
    all_batsmen = data['batter'].unique()

    merged_df = pd.DataFrame()

    for player in all_batsmen:
        player_scores = getPlayerScores(player)
        
        merged_df = pd.concat([merged_df, player_scores])
    
    merged_df["strike_rate"] = merged_df["strike_rate"].fillna(0)

    merged_df.drop(columns=["batter_runs","balls_faced","wicket_type","strike_rate","out","last_ball"],inplace=True)
    merged_df.to_csv('Data/selected_data/all_batters_NRR.csv',index=False)




def getPlayerScores(player_name: str, innings: list[int] = [1, 2] ) -> pd.DataFrame:
    # Get the data for BKG Mendis if batter is BKG Mendis or non-striker is BKG Mendis
	player_data = data.loc[
		((data['batter'] == player_name) | (data['non_striker'] == player_name)) & (data['innings'].isin(innings))
	]

	player_data.head()

	# 3 matches missing from the data
	# group data by match_id
	gp = player_data.groupby('match_id')
	cols = ['batting_team', 'batter', 'non_striker', 'batter_runs', 'balls_faced', 'wicket_type', 'won', 'innings', 'over', 'delivery', 'wickets_fallen','bowling_team','venue','net_run_rate']
	player_scores= gp.last().loc[:, cols]

	# get the first ball he faced or at non-striker
	first_ball = gp.first().loc[:, ['over', 'delivery', 'wickets_fallen']]
	first_ball['first_ball'] = (first_ball['over'] * 6 + first_ball['delivery']).astype(int)

	player_scores['first_ball'] = first_ball['first_ball']
	player_scores['wickets_fallen'] = first_ball['wickets_fallen']

	# when BKG Mendis is the non-striker when the last ball was bowled
	# The batter_runs and balls_faced are not his, but the on_strike batter's
	# So, we need to get the last ball he faced
	# he might not even have faced a ball

	# get the last ball he faced

	matches_non_striker = player_scores[player_scores['non_striker'] == player_name].index
	
	# Sometimes the player might not even have faced a single ball
	# Eg: Afghanistan_Sri Lanka_2022-11-01 MD Shanaka not out on the non strikers end

	player_scores.loc[matches_non_striker, ['batter_runs', 'balls_faced']] = [0, 0]
	
	
	# get the last batter == player_name row from gp data
	gp = player_data[(player_data['batter'] == player_name) & (player_data['match_id'].isin(matches_non_striker))].groupby(['match_id'])
	last_batter_scores = gp.last()[['batter_runs', 'balls_faced']]	
	
	# update the rows with non_striker with correct values
	player_scores.update(last_batter_scores)
	
	# adding new features
	# strike rate
	player_scores['strike_rate'] = round(player_scores['batter_runs'] / player_scores['balls_faced'] * 100, 2)
	player_scores['out'] = player_scores['wicket_type'] != '0'
	player_scores['last_ball'] = (player_scores['over'] * 6 + player_scores['delivery']).astype(int)
 
 
	player_scores['batter'] = player_name
	player_scores.drop('non_striker', inplace=True, axis = 1)

	# drop over and delivery
	player_scores.drop(['over', 'delivery'], inplace=True, axis=1)
 
	return player_scores


def preprocess_data():
    # Load the data
    df = pd.read_csv("Data/MLOps_data/merged_data.csv")
    df.drop(columns=['Unnamed: 0',"review","fielder"], inplace=True)

    df["runs_remain"] = df["runs_remain"].fillna(-1)
    df["replacements"] = df["replacements"].fillna("No Replacement")
    df["player_out"] = df["player_out"].fillna("Not Out")
    df["won"] = df["won"].fillna(-1)
    df["bowler_type"] = df["bowler_type"].fillna("Right arm Offbreak")

    # Exporting the processed data
    df.to_csv("Data/MLOps_data/processed_data.csv", index=False)

def check_null_values():
    df = pd.read_csv("Data/MLOps_data/processed_data.csv")
    # if there are any null values in the data exit the program
    if df.isnull().sum().sum() > 0:
        print("There are null values in the data")
        exit(1)
    else:
        print("There are no null values in the data")

# Copy new data to existing data
def copy_data():
    df = pd.read_csv("Data/selected_data/processed_data.csv")
    df_new = pd.read_csv("Data/MLOps_data/processed_data.csv")

    df = pd.concat([df, df_new], axis=0)
    df.to_csv("Data/selected_data/processed_data.csv", index=False)

def main():
    preprocess_data()
    check_null_values()
    copy_data()
    preprocess_NRR_data()
    All_Batters_NRR()

if __name__ == "__main__":
    main()