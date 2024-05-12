import shutil

def zip_directory(directory_path, zip_path):
    # Ensure the directory exists
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory '{directory_path}' does not exist.")

    # Zip the directory
    shutil.make_archive(zip_path, 'zip', directory_path)
