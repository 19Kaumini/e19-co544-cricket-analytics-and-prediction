import yaml, os, json
import pandas as pd
import numpy as np


def convert_over_to_df(over_data):
    over_df = pd.DataFrame(over_data)
    over_df["runs_by_bat"] = over_df["runs"].apply(lambda x: x.get("batter"))
    over_df["extra_runs"] = over_df["runs"].apply(lambda x: x.get("extras"))
    over_df["total"] = over_df["runs"].apply(lambda x: x.get("total"))
    over_df["delivery"] = np.arange(1, len(over_df) + 1)

    if "extras" in over_df.columns:
        over_df["extra_type"] = over_df["extras"].apply(
            lambda x: "".join(list(x.keys())) if type(x) == dict else np.nan
        )

    if "wickets" in over_df.columns:
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

    # over_df['extra_type'] = np.nan
    # over_df['wicket_type'] = np.nan
    # over_df['player_out'] = np.nan
    # over_df['fielder'] = np.nan

    over_df.drop(columns=["runs"], inplace=True)
    return over_df


def complete_team_df(team_overs):
    all_overs = []
    for over_index, over in enumerate(team_overs):
        over_df = convert_over_to_df(over["deliveries"])
        over_df["over"] = over_index + 1
        all_overs.append(over_df)
    return pd.concat(all_overs, ignore_index=True)


def json_to_csv(match_file, output_file=False):

    with open(match_file, "r") as f:
        file = json.load(f)

    info = file["info"]
    # Added toss decision and toss winner
    toss_decision = info["toss"]["decision"]
    toss_win_team = info["toss"]["winner"]

    innings = file["innings"]
    length = len(innings)

    if length == 0:
        print("No innings data found")
        return [], info

    all_innings_df = {}
    for idx, inning in enumerate(innings):
        team = inning["overs"]
        # print("team:", team)

        df = complete_team_df(team)
        df["extra_type"] = df["extra_type"].fillna("-")
        df["wicket_type"] = df["wicket_type"].fillna(0)
        df["toss_decision"] = toss_decision
        df["toss_winner"] = toss_win_team
        df["innings"] = idx + 1
        df["venue"] = info["venue"]
        df["date"] = info["dates"][0]
        print("innings", innings)
        print("inning & team", inning)
        df["team_bat"] = innings[idx]["team"]
        df["team_ball"] = innings[1 if idx == 0 else 0]["team"]

        team_innings = f"{inning['team']}_{idx+1}"
        if output_file:

            file_path = f"./csv_files/{os.path.splitext(os.path.split(match_file)[-1])[0]}_{team_innings}.csv"
            df.to_csv(file_path)

        all_innings_df[team_innings] = df

    return all_innings_df, info
