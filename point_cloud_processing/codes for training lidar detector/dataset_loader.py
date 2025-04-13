import os
import numpy as np
import pandas as pd

def extract_timestamp(file_name):
    # Extract timestamp from file name
    # Assuming file name format: <timestamp>.npy
    return os.path.splitext(file_name)[0]

def read_lidar_files(directory):
    # Initialize an empty dictionary to store point cloud data by timestamp
    point_cloud_data = {}

    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return point_cloud_data

    # Iterate through files in the directory
    files = os.listdir(directory)
    for file_name in sorted(files):
        if file_name.endswith(".npy"):
            # Construct file path
            file_path = os.path.join(directory, file_name)

            # Load .npy file using NumPy
            try:
                data = np.load(file_path)  # Load the data
                # Store point cloud data with timestamp
                point_cloud_data[extract_timestamp(file_name)] = data
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

    return point_cloud_data
    

def update_csv(timestamps, positions, csv_path):
    # Read the existing CSV file into a DataFrame
    df = pd.read_csv(csv_path, delimiter=',')
    seq_set = set()
    # Update the DataFrame with new data
    for ts, pos in zip(timestamps, positions):
        ind_at = df.index[df['Timestamp'] == ts]
        df.at[ind_at, 'Position'] = np.array2string(pos, separator=',')
        seq_set.update(set(df.loc[ind_at, 'Sequence']))
    # Write the updated DataFrame back to the CSV file
    df.to_csv(csv_path, sep=',', index=False)
    print(f"Successfully update {seq_set}")
