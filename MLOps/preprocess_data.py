import pandas as pd


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
    # copy_data()

if __name__ == "__main__":
    main()