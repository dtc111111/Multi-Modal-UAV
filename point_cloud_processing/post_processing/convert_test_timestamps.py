import csv
import os
from datetime import datetime
import numpy as np


# Function to create sequentially named folder
def create_folder(base_dir, sequence):
    folder_path = os.path.join(base_dir, sequence)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

# Path to CSV file
csv_file = 'test_timestamp.csv'  # Change this to your CSV file path

# Directory to save folders
base_dir = '/media/yi/KESU/anti_uav/test'  # Change this to your desired directory

# Create output directory if it doesn't exist
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# Read CSV file and parse timestamps
with open(csv_file, 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip header if exists
    prev_sequence = None
    sequence_timestamps = []
    for row in csv_reader:
        sequence = row[0]  # Assuming sequence is in the first column
        timestamp = row[1]
        if timestamp:
            if sequence != prev_sequence and prev_sequence is not None:
                # Save collected timestamps for previous sequence
                folder_path = create_folder(base_dir, prev_sequence)
                timestamps_array = np.array([float(timestamp) for timestamp in sequence_timestamps])
                np.save(os.path.join(folder_path, "test_time.npy"), timestamps_array)
                # Reset sequence_timestamps for the new sequence
                sequence_timestamps = []
            sequence_timestamps.append(row[1])  # Assuming timestamp is in the second column
            prev_sequence = sequence
    # Save the last sequence's timestamps
    if sequence_timestamps:
        folder_path = create_folder(base_dir, prev_sequence)
        timestamps_array = np.array([float(timestamp) for timestamp in sequence_timestamps])
        np.save(os.path.join(folder_path, "test_time.npy"), timestamps_array)

print("Timestamp files saved successfully as test_time.npy in each sequence folder.")
