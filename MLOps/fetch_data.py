import requests
import zipfile
import os
import shutil

# URLs and directories
zip_url = "https://cricsheet.org/downloads/t20s_json.zip"
download_directory = "downloads"
data_directory = "Data/t20s_male_json"
temp_directory = "Data/temp"

# Ensure directories exist
os.makedirs(download_directory, exist_ok=True)
os.makedirs(temp_directory, exist_ok=True)
os.makedirs(data_directory, exist_ok=True)

def download_and_unzip(zip_url):
    zip_filename = os.path.join(download_directory, "t20s_json.zip")
    
    # Download the zip file
    response = requests.get(zip_url)
    with open(zip_filename, "wb") as file:
        print(f"Downloading {zip_url} to {zip_filename}")
        file.write(response.content)
    
    # Unzip the file
    with zipfile.ZipFile(zip_filename, "r") as zip_ref:
        zip_ref.extractall(download_directory)
    
    return download_directory

def get_json_file_names(directory):
    return {file for file in os.listdir(directory) if file.endswith(".json")}

def copy_new_json_files(source_dir, dest_dir, existing_files):
    existing_file_numbers = {int(file.split('.')[0]) for file in existing_files}
    max_existing_file_number = max(existing_file_numbers) if existing_file_numbers else -1
    
    for file in os.listdir(source_dir):
        if file.endswith(".json"):
            file_number = int(file.split('.')[0])
            if file_number > max_existing_file_number:
                shutil.copy2(os.path.join(source_dir, file), os.path.join(dest_dir, file))
                print(f"Copied new file: {file}")

# after copying the new files delete all downloaded files in Downloads folder
def delete_downloaded_files(directory):
    print(f"Deleting downloaded files in {directory}")
    for file in os.listdir(directory):
        os.remove(os.path.join(directory, file))

# Copy new JSON files in the temp directory to the data directory
def copy_temp_to_data():
    for file in os.listdir(temp_directory):
        shutil.copy2(os.path.join(temp_directory, file), os.path.join(data_directory, file))
        print(f"Copied new file: {file}")
    print(f"New JSON files have been copied to {data_directory}")

def main():
    download_dir = download_and_unzip(zip_url)
    
    # List of existing JSON files
    existing_files = get_json_file_names(data_directory)
    
    # Copy new JSON files
    copy_new_json_files(download_dir, temp_directory, existing_files)
    
    print(f"New JSON files have been copied to {temp_directory}")

    # Delete downloaded files
    delete_downloaded_files(download_dir)

    # Copy new JSON files to the data directory
    copy_temp_to_data()

if __name__ == "__main__":
    main()
