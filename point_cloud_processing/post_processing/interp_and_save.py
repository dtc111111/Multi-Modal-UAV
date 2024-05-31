import os
import numpy as np
import pandas as pd
from dataset_loader import extract_timestamp, read_lidar_files

dataset_folder = "/media/yi/KESU/anti_uav/train"
figure_folder = "/media/yi/KESU/anti_uav/figure/png/interpolation"
csv_path = 'test_timestamp.csv'

# Function to update the DataFrame with new data
def update_csv(seq_name, ts, positions, csv_path):
    # Read the existing CSV file into a DataFrame
    df = pd.read_csv(csv_path, delimiter=',')

    # Update the DataFrame with new data
    for ts, pos in zip(timestamps, positions):
        ind_at = df.index[(df['Sequence'] == seq_name) & (df['Timestamp'] == ts)][0]
        df.at[ind_at, 'Position'] = list(pos)

    # Write the updated DataFrame back to the CSV file
    df.to_csv(csv_path, sep=',', index=False)

for seq_folder in os.listdir(dataset_folder):
    print("Processing "+seq_folder)
    trajectory_data = []
    seq_folder_path = os.path.join(dataset_folder, seq_folder)
    lidar_track_directory =  os.path.join(seq_folder_path, "gt_track")
    lidar_track_data = read_lidar_files(lidar_track_directory)
    for timestamp, data in lidar_track_data.items():
        timestamp_float = float(timestamp)
        trajectory_data.append([timestamp_float, *data])
    trajectory_data = np.array(trajectory_data)
    timestamps = trajectory_data[:, 0]  # ROS timestamps (in seconds)
    trajectory_points = trajectory_data[:, 1:]
    test_timestamps = np.arange(timestamps[0]-2, timestamps[-1] + 2, 0.1)  # New timestamps (ROS timestamps)
    #test_timestamps = np.load(os.path.join(seq_folder_path, "test_time.npy"))
    new_trajectory = interpolate_trajectory(timestamps, trajectory_points, test_timestamps)
    fig = plt.figure(figsize = (16,4))
    plt.subplot(1,3,1)
    plt.plot(test_timestamps,new_trajectory[:,0], 'rx', markersize=5)
    plt.plot(timestamps,trajectory_points[:,0], 'bx', markersize=5)
    plt.subplot(1,3,2)
    plt.plot(test_timestamps,new_trajectory[:,1], 'rx', markersize=5)
    plt.plot(timestamps,trajectory_points[:,1], 'bx', markersize=5)
    plt.subplot(1,3,3)
    plt.plot(test_timestamps,new_trajectory[:,2], 'rx', markersize=5)
    plt.plot(timestamps,trajectory_points[:,2], 'bx', markersize=5)
    plt.suptitle(seq_folder)
    plt.savefig(os.path.join(figure_folder, seq_folder+'.png'))
    plt.close()
    update_csv(seq_folder, test_timestamps, new_trajectory, csv_path)
