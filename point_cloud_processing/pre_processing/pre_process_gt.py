import os
import numpy as np

# Function to process files in a folder
def process_files(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all .npy files in the input folder
    files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]

    # Process each file
    for file in files:
        # Load the array from the .npy file
        file_path = os.path.join(input_folder, file)
        array = np.load(file_path)

        # Reshape the array
        reshaped_array = array.reshape(1, -1)

        # Save the reshaped array to the output folder with the same filename
        output_file_path = os.path.join(output_folder, file)
        np.save(output_file_path, reshaped_array)

        print(f"Processed file: {file}")

# Example usage
dataset_folder = "/home/yi/Desktop/anti_uav/val"
#dataset_folder = "/media/yi/KESU/anti_uav/train"
for seq_folder in os.listdir(dataset_folder):
    seq_folder_path = os.path.join(dataset_folder, seq_folder)
    input_folder = os.path.join(seq_folder_path, "ground_truth")
    output_folder = os.path.join(seq_folder_path, "gt")
    process_files(input_folder, output_folder)
