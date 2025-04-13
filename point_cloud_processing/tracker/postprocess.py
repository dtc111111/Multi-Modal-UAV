import os
import numpy as np
from dataset_loader import read_lidar_files
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d, splev, splrep
import argparse

def interpolate_trajectory(timestamps, points, new_timestamps, interval, interp_method='linear'):
    """Interpolate trajectory using linear interpolation"""
    x_vals = points[:, 0]
    y_vals = points[:, 1]
    z_vals = points[:, 2]

    interp_x = interp1d(timestamps, x_vals, kind=interp_method, fill_value="extrapolate")
    interp_y = interp1d(timestamps, y_vals, kind=interp_method, fill_value="extrapolate")
    interp_z = interp1d(timestamps, z_vals, kind=interp_method, fill_value="extrapolate")

    new_x_vals = interp_x(new_timestamps[interval[0]:interval[1]])
    new_y_vals = interp_y(new_timestamps[interval[0]:interval[1]])
    new_z_vals = interp_z(new_timestamps[interval[0]:interval[1]])

    return np.column_stack((new_x_vals, new_y_vals, new_z_vals))


def interpolate_trajectory_spline(timestamps, points, new_timestamps, interval, s):
    """Interpolate trajectory using spline interpolation"""
    x_vals = points[:, 0]
    y_vals = points[:, 1]
    z_vals = points[:, 2]

    spl_x = splrep(timestamps, x_vals, s=s)
    spl_y = splrep(timestamps, y_vals, s=s)
    spl_z = splrep(timestamps, z_vals, s=s)

    new_x_vals = splev(new_timestamps[interval[0]:interval[1]], spl_x)
    new_y_vals = splev(new_timestamps[interval[0]:interval[1]], spl_y)
    new_z_vals = splev(new_timestamps[interval[0]:interval[1]], spl_z)

    return np.column_stack((new_x_vals, new_y_vals, new_z_vals))


def load_trajectory_data(dataset_folder, seq_folder):
    """Load trajectory data from files"""
    print(f"Processing {seq_folder}")
    seq_folder_path = os.path.join(dataset_folder, seq_folder)
    lidar_track_data = read_lidar_files(seq_folder_path)

    trajectory_data = []
    for timestamp, data in lidar_track_data.items():
        timestamp_float = float(timestamp)
        trajectory_data.append([timestamp_float, *data])

    trajectory_data = np.array(trajectory_data)
    timestamps = trajectory_data[:, 0]
    trajectory_points = trajectory_data[:, 1:]


    start_time = np.min(timestamps)
    end_time = np.max(timestamps)

    test_timestamps = np.arange(start_time, end_time + 0.1, 0.1)

    return timestamps, trajectory_points, test_timestamps


def plot_trajectories(timestamps, trajectory_points, test_timestamps, new_trajectory_spline, interval, seq_folder, output_path):
    """Plot original and interpolated trajectories with velocities"""
    time_diff = test_timestamps[interval[0]:interval[1] - 1] - test_timestamps[interval[0] + 1:interval[1]]
    x_vel = (new_trajectory_spline[0:-1, 0] - new_trajectory_spline[1:, 0]) / time_diff
    y_vel = (new_trajectory_spline[0:-1, 1] - new_trajectory_spline[1:, 1]) / time_diff
    z_vel = (new_trajectory_spline[0:-1, 2] - new_trajectory_spline[1:, 2]) / time_diff

    fig = plt.figure(figsize=(16, 8))
    plt.subplot(2, 3, 1)
    plt.plot(test_timestamps[interval[0]:interval[1]], new_trajectory_spline[:, 0], 'r.', markersize=5)
    plt.plot(timestamps, trajectory_points[:, 0], 'b.', markersize=1)
    plt.title('X Position')

    plt.subplot(2, 3, 2)
    plt.plot(test_timestamps[interval[0]:interval[1]], new_trajectory_spline[:, 1], 'r.', markersize=5)
    plt.plot(timestamps, trajectory_points[:, 1], 'b.', markersize=1)
    plt.title('Y Position')

    plt.subplot(2, 3, 3)
    plt.plot(test_timestamps[interval[0]:interval[1]], new_trajectory_spline[:, 2], 'r.', markersize=5)
    plt.plot(timestamps, trajectory_points[:, 2], 'b.', markersize=1)
    plt.title('Z Position')

    plt.subplot(2, 3, 4)
    plt.plot(test_timestamps[interval[0]:interval[1] - 1], x_vel, 'b.', markersize=2)
    plt.title('X Velocity')

    plt.subplot(2, 3, 5)
    plt.plot(test_timestamps[interval[0]:interval[1] - 1], y_vel, 'b.', markersize=2)
    plt.title('Y Velocity')

    plt.subplot(2, 3, 6)
    plt.plot(test_timestamps[interval[0]:interval[1] - 1], z_vel, 'b.', markersize=2)
    plt.title('Z Velocity')

    plt.suptitle(seq_folder)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()  # Close the figure to free memory


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process and visualize trajectory data with interpolation.")
    parser.add_argument('--dataset_folder', type=str, default=r"C:\Users\Yi\Desktop\result",
                        help="Path to the dataset folder")
    parser.add_argument('--figure_folder', type=str, default=r"C:\Users\Yi\Desktop\figure",
                        help="Path to the output figure folder")
    parser.add_argument('--seq_folder', type=str, default="seq1",
                        help="Sequence folder name to process")
    parser.add_argument('--spline_smoothing', type=float, default=0.5,
                        help="Smoothing factor for spline interpolation")

    # Parse arguments
    args = parser.parse_args()

    # Create figure folder if it doesn't exist
    if not os.path.exists(args.figure_folder):
        os.makedirs(args.figure_folder)
    figure_path = os.path.join(args.figure_folder, args.seq_folder + '.png')

    # Load data
    timestamps, trajectory_points, test_timestamps = load_trajectory_data(args.dataset_folder, args.seq_folder)

    # Set interpolation interval
    interval = [0, len(test_timestamps)]

    # Perform interpolations
    new_trajectory_linear = interpolate_trajectory(timestamps, trajectory_points, test_timestamps, interval)
    new_trajectory_spline = interpolate_trajectory_spline(timestamps, trajectory_points, test_timestamps, interval,
                                                        args.spline_smoothing)

    # Calculate and print mean difference between linear and spline interpolation
    mean_diff = np.mean(np.linalg.norm(new_trajectory_spline - new_trajectory_linear, axis=1))
    #print(f"Mean difference between linear and spline interpolation: {mean_diff}")

    # Plot results
    plot_trajectories(timestamps, trajectory_points, test_timestamps, new_trajectory_spline, interval,
                     args.seq_folder, figure_path)