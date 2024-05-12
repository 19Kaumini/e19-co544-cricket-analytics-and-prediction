import os, json
import pandas as pd
import numpy as np
from datetime import datetime

# global
batter_scores = {}
balls_faced = {}
total_score = 0
wickets_fallen = 0
first_inning_total = 0
current_inning = 0


def convert_over_to_df(over_data):

    over_df = pd.DataFrame(over_data)
    over_df["runs_by_bat"] = over_df["runs"].apply(lambda x: x.get("batter"))
    over_df["extra_runs"] = over_df["runs"].apply(lambda x: x.get("extras"))
    over_df["total_runs_delivery"] = over_df["runs"].apply(lambda x: x.get("total"))
    # Cumulative sum of the total score
    global total_score
    global current_inning
    score = over_df["total_runs_delivery"].cumsum() + total_score
    over_df["current_team_total"] = score
    if current_inning == 1:
        over_df["runs_remain"] = first_inning_total - score
    else:
        over_df["runs_remain"] = np.nan
    total_score = score.iloc[-1]

    over_df["batter_runs"] = [{} for _ in range(len(over_df))]
    over_df["balls_faced"] = [{} for _ in range(len(over_df))]
    over_df["wickets_fallen"] = [wickets_fallen for _ in range(len(over_df))]

    for idx, row in over_df.iterrows():

        batter = row["batter"]
        runs = row["runs_by_bat"]

        if batter in balls_faced:
            balls_faced[batter] += 1
        else:
            balls_faced[batter] = 1
        over_df.at[idx, "balls_faced"] = balls_faced.copy()[batter]

        if batter in batter_scores:
            batter_scores[batter] += runs
        else:
            batter_scores[batter] = runs
        over_df.at[idx, "batter_runs"] = batter_scores.copy()[batter]

    over_df["delivery"] = np.arange(1, len(over_df) + 1)

    if "extras" in over_df.columns:
        over_df["extra_type"] = over_df["extras"].apply(
            lambda x: "".join(list(x.keys())) if type(x) == dict else np.nan
        )

    if "wickets" in over_df.columns:

        w_count_temp = over_df["wickets"].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        global wickets_fallen

        wic = w_count_temp.cumsum() + wickets_fallen
        over_df["wickets_fallen"] = wic
        wickets_fallen = wic.iloc[-1]

        over_df["wicket_type"] = over_df["wickets"].apply(
            lambda x: x[0].get("kind") if type(x) == list else np.nan
        )

        over_df["player_out"] = over_df["wickets"].apply(
            lambda x: x[0].get("player_out") if type(x) == list else np.nan
        )

        def get_fielder_name(x):
            fielder_list = x[0].get("fielders") if type(x) == list else []
            if fielder_list:
                return ";".join(fielder.get("name") for fielder in fielder_list)
            return np.nan

        over_df["fielder"] = over_df["wickets"].apply(get_fielder_name)
        over_df.drop(columns=["wickets"], inplace=True)

    over_df.drop(columns=["runs"], inplace=True)

    return over_df


def complete_team_df(team_overs):
    all_overs = []
    # keep tarck of total runs

    for over_index, over in enumerate(team_overs):
        # over["deliveries"] array of objects(per ball)
        over_df = convert_over_to_df(over["deliveries"])
        over_df["over"] = over_index
        all_overs.append(over_df)
    return pd.concat(all_overs, ignore_index=True)


def json_to_csv(match_file, output_file=False):

    with open(match_file, "r") as f:
        file = json.load(f)

    info = file["info"]

    # decide whether the teams are top 10 or not
    teams = [
        "South Africa",
        "England",
        "Australia",
        "New Zealand",
        "India",
        "Pakistan",
        "Sri Lanka",
        "West Indies",
        "Bangladesh",
        "Zimbabwe",
        "Ireland",
        "Afghanistan",
    ]
    # Filter based on Associate countries
    date = info["dates"][0]

    if info["teams"][0] not in teams or info["teams"][1] not in teams:
        return
    else:
        if info["teams"][0] == "Afghanistan" or info["teams"][1] == "Afghanistan":
            if datetime.strptime(date, "%Y-%m-%d") < datetime(2017, 6, 22):
                return

    # Added toss decision and toss winner
    toss_decision = info["toss"]["decision"]
    toss_win_team = info["toss"]["winner"]
    players = info["registry"]["people"]
    match_id = info["teams"][0] + "_" + info["teams"][1] + "_" + date

    innings = file["innings"]
    length = len(innings)

    if length == 0:
        print("No innings data found")
        return [], info

    all_innings_df = {}
    for idx, inning in enumerate(innings):
        global total_score
        global current_inning
        current_inning = idx
        if idx == 1:
            global first_inning_total
            first_inning_total = total_score

        global batter_scores
        batter_scores = {}

        global balls_faced
        balls_faced = {}

        total_score = 0

        global wickets_fallen
        wickets_fallen = 0

        team = inning["overs"]

        # Call per inning
        df = complete_team_df(team)

        if "extra_type" not in df.columns:
            df["extra_type"] = 0
        df["extra_type"] = df["extra_type"].fillna("-")

        if "wicket_type" not in df.columns:
            df["wicket_type"] = 0

        df["wicket_type"] = df["wicket_type"].fillna(0)
        df["toss_decision"] = toss_decision
        df["toss_winner"] = toss_win_team
        df["innings"] = idx + 1
        df["venue"] = info["venue"]
        df["year"] = info["dates"][0].split("-")[0]
        df["month"] = info["dates"][0].split("-")[1]
        df["match_id"] = match_id

        # add winning team
        if "result" in info["outcome"]:
            df["winning_team"] = info["outcome"]["result"]
        else:
            df["winning_team"] = info["outcome"]["winner"]

        # drop extras column

        # No outcome error handle

        if len(innings) > 1:
            df["batting_team"] = innings[idx]["team"]
            df["bowling_team"] = innings[1 if idx == 0 else 0]["team"]
        else:
            # Handle the case when there is no second inning
            df["batting_team"] = innings[idx]["team"]
            df["bowling_team"] = None

        # outcome
        outcome = info["outcome"]
        if "winner" in outcome:
            df["won"] = outcome["winner"] == df["batting_team"]
        else:
            df["won"] = np.nan

        team_innings = f"{inning['team']}_{idx+1}"

        if "extras" in df.columns:
            df.drop(columns=["extras"], inplace=True)
        df["final_team_total"] = df["current_team_total"].iloc[-1]
        if output_file:

            file_path = f"../Data/selected_data/csv_files/{os.path.splitext(os.path.split(match_file)[-1])[0]}_{team_innings}.csv"
            df.to_csv(file_path)

        all_innings_df[team_innings] = df

    return all_innings_df, players
