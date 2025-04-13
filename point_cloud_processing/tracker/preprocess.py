import os
import numpy as np
from sklearn.cluster import DBSCAN
from dataset_loader import read_lidar_files
from extract_feature import extract_feature_set_predict
from lidar_360_detector import MyLSTMClassifier
import argparse
import os

# --- Livox Avia Processing ---
def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

def process_lidar_livox(seq_folder_path, max_pts=100):
    input_folder = os.path.join(seq_folder_path, "livox_avia")
    output_folder = os.path.join(seq_folder_path, "livox_avia_processed")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]
    for file in files:
        file_path = os.path.join(input_folder, file)
        livox_avia_data = np.load(file_path)
        mask = np.any(livox_avia_data != 0, axis=1)
        filtered_data = livox_avia_data[mask]
        if filtered_data.shape[0] > max_pts:
            #print(f"Downsample to max_pts {max_pts}")
            filtered_data = farthest_point_sample(filtered_data, max_pts)
        output_file_path = os.path.join(output_folder, file)
        np.save(output_file_path, filtered_data)

# --- Lidar 360 Processing ---
def process_lidar_360(seq_folder_path, model_path):
    lidar_360_directory = os.path.join(seq_folder_path, "lidar_360")
    lidar_360_data = read_lidar_files(lidar_360_directory)
    output_folder = os.path.join(seq_folder_path, "lidar_360_processed")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    total_frame = len(lidar_360_data)
    accumulated_lidar_360_data = {}
    accumulated_frame_timestamps = []
    accumulated_all_timestamps = {}
    accumulated_data = np.array([])

    for timestamp, data in lidar_360_data.items():
        accumulated_frame_timestamps.append(timestamp)
        mask = np.any(data != 0, axis=1)
        filtered_data = data[mask]
        frame_ind = len(accumulated_frame_timestamps) * np.ones([filtered_data.shape[0], 1])
        data_with_ind = np.concatenate((frame_ind, filtered_data), axis=1)
        if np.size(accumulated_data) == 0:
            accumulated_data = data_with_ind
        else:
            accumulated_data = np.concatenate((accumulated_data, data_with_ind), axis=0)
        if len(accumulated_frame_timestamps) == 20:
            accumulated_lidar_360_data[accumulated_frame_timestamps[-1]] = np.array(accumulated_data)
            accumulated_all_timestamps[accumulated_frame_timestamps[-1]] = accumulated_frame_timestamps
            accumulated_frame_timestamps = []
            accumulated_data = np.array([])

    # Process last frames
    accumulated_frame_timestamps = []
    accumulated_data = np.array([])
    for idx, (timestamp, data) in enumerate(lidar_360_data.items()):
        if idx >= total_frame - 20:
            accumulated_frame_timestamps.append(timestamp)
            mask = np.any(data != 0, axis=1)
            filtered_data = data[mask]
            frame_ind = len(accumulated_frame_timestamps) * np.ones([filtered_data.shape[0], 1])
            data_with_ind = np.concatenate((frame_ind, filtered_data), axis=1)
            if np.size(accumulated_data) == 0:
                accumulated_data = data_with_ind
            else:
                accumulated_data = np.concatenate((accumulated_data, data_with_ind), axis=0)
            if len(accumulated_frame_timestamps) == 20:
                accumulated_lidar_360_data[accumulated_frame_timestamps[-1]] = np.array(accumulated_data)
                accumulated_all_timestamps[accumulated_frame_timestamps[-1]] = accumulated_frame_timestamps

    # Initialize model
    input_size = 9
    hidden_size = 64
    num_layers = 1
    num_classes = 2
    model = MyLSTMClassifier(input_size, hidden_size, num_layers, num_classes)

    for timestamp, data_with_ind in accumulated_lidar_360_data.items():
        frame_time = accumulated_all_timestamps[timestamp]
        point_cloud_data = {key: [] for key in frame_time}
        time_ind = data_with_ind[:, 0]
        data = data_with_ind[:, 1:]
        if np.size(data, 0) != 0:
            try:
                db = DBSCAN(eps=2, min_samples=10).fit(data)
            except AttributeError:
                continue
            labels = db.labels_
            feature_set, cluster_labels = extract_feature_set_predict(data, labels, time_ind)
            predicted = model.detect(feature_set, model_path)
            predicted_labels = cluster_labels[predicted == 1]
            for predicted_label in predicted_labels:
                det_data = data[labels == predicted_label]
                det_time_ind = time_ind[labels == predicted_label].astype(int) - 1
                for ind, i in enumerate(det_time_ind):
                    point_cloud_data[frame_time[i]].append(det_data[ind].tolist())
            for frame_name in frame_time:
                output_file_path = os.path.join(output_folder, frame_name + '.npy')
                saved_pts = point_cloud_data[frame_name]
                np.save(output_file_path, np.array(saved_pts))

# --- Fusion Processing ---
def process_fusion(seq_folder_path):
    lidar_360_directory = os.path.join(seq_folder_path, "lidar_360_processed")
    lidar_360_data = read_lidar_files(lidar_360_directory)
    livox_avia_directory = os.path.join(seq_folder_path, "livox_avia_processed")
    livox_avia_data = read_lidar_files(livox_avia_directory)

    merged_dict = {**livox_avia_data, **lidar_360_data}
    lidar_data = {key: merged_dict[key] for key in sorted(merged_dict, key=lambda x: float(x))}
    output_folder = os.path.join(seq_folder_path, "lidar_fusion")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    accumulated_timestamps = []
    accumulated_data = np.array([])

    for timestamp, data in lidar_data.items():
        if data.size != 0:
            accumulated_timestamps.append(timestamp)
            mask = np.any(data != 0, axis=1)
            filtered_data = data[mask]
            frame_ind = len(accumulated_timestamps) * np.ones([filtered_data.shape[0], 1])
            data_with_ind = np.concatenate((frame_ind, filtered_data), axis=1)
            if np.size(accumulated_data) == 0:
                accumulated_data = data_with_ind
            else:
                accumulated_data = np.concatenate((accumulated_data, data_with_ind), axis=0)

    accumulated_data = np.array(accumulated_data)
    #print(accumulated_data.shape)

    if accumulated_data.shape[0] < 5e4:
        time_ind = accumulated_data[:, 0]
        data = accumulated_data[:, 1:]
        if np.size(data, 0) != 0:
            db = DBSCAN(eps=1, min_samples=10).fit(data)
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)
            #print("Estimated number of clusters: %d" % n_clusters_)
            #print("Estimated number of noise points: %d" % n_noise_)

        detections = accumulated_data[labels != -1]
        point_cloud_data = {key: [] for key in accumulated_timestamps}
        det_data = detections[:, 1:]
        det_time_ind = detections[:, 0].astype(int) - 1
        for ind, i in enumerate(det_time_ind):
            point_cloud_data[accumulated_timestamps[i]].append(det_data[ind].tolist())

        for frame_name in accumulated_timestamps:
            output_file_path = os.path.join(output_folder, frame_name + '.npy')
            saved_pts = point_cloud_data[frame_name]
            np.save(output_file_path, np.array(saved_pts))
    else:
        print("Cannot process " + os.path.basename(seq_folder_path))



if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process LiDAR data and perform fusion for multiple sequences.")
    parser.add_argument('--dataset_folder', type=str, default=r"C:\Users\Yi\Desktop\val",
                        help="Path to the dataset folder containing sequence subfolders")
    parser.add_argument('--model_path', type=str, default=None,
                        help="Path to the LSTM model file (default: lstm_model.pth in script directory)")

    # Parse arguments
    args = parser.parse_args()

    # Set dataset folder
    dataset_folder = args.dataset_folder

    # Set model path (default to lstm_model.pth in script directory if not provided)
    if args.model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "lstm_model.pth")
    else:
        model_path = args.model_path

    # Ensure dataset folder exists
    if not os.path.exists(dataset_folder):
        print(f"Error: Dataset folder '{dataset_folder}' does not exist.")
        exit(1)

    # Ensure model path exists if provided or default is used
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' does not exist.")
        exit(1)

    # Process each sequence folder
    for seq_folder in os.listdir(dataset_folder):
        print(f"Processing {seq_folder}")
        seq_folder_path = os.path.join(dataset_folder, seq_folder)

        # Check if seq_folder_path is a directory
        if not os.path.isdir(seq_folder_path):
            print(f"Skipping '{seq_folder}' (not a directory)")
            continue

        # Process Livox Avia
        process_lidar_livox(seq_folder_path)

        # Process Lidar 360
        process_lidar_360(seq_folder_path, model_path)

        # Process Fusion
        process_fusion(seq_folder_path)