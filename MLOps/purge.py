import os

def delete_downloaded_files(directory):
    print(f"Deleting files in {directory}")
    for file in os.listdir(directory):
        os.remove(os.path.join(directory, file))

# delete csv files in the directories
def delete_csv_files(directories):
    for directory in directories:
        print(f"Deleting csv files in {directory}")
        for file in os.listdir(directory):
            if file.endswith(".csv"):
                os.remove(os.path.join(directory, file))

directory = "Data/MLOps_data/csv_files"
csv_directory = "Data/MLOps_data"

def main():
    delete_downloaded_files(directory)
    delete_csv_files([csv_directory])

if __name__ == "__main__":
    main()