import os

def clear_directory(directory):
    # List all files in the directory
    files = os.listdir(directory)
    
    # Iterate over each file and remove it
    for file in files:
        file_path = os.path.join(directory, file)
        os.remove(file_path)

    print(f"All files in {directory} have been removed.")

# Specify the directory you want to clear
directory_to_clear = "./generated_images"

# Call the function to clear the directory
clear_directory(directory_to_clear)
