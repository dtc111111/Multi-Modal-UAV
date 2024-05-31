import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from extract_feature import extract_feature_set_predict
from dataset_loader import extract_timestamp, read_lidar_files
from sklearn.cluster import DBSCAN
from lidar_360_detector import MyLSTMClassifier


#dataset_folder = "/media/yi/Backup Plus/BaiduNetdiskDownload/test/test"
dataset_folder = "/media/yi/KESU/anti_uav/val"


# Initialize model
input_size = 9  # Number of features
hidden_size = 64
num_layers = 1
num_classes = 2  # Number of unique classes
model = MyLSTMClassifier(input_size, hidden_size, num_layers, num_classes)
model_path = 'lstm_model.pth'


for seq_folder in os.listdir(dataset_folder):
    print("Processing "+seq_folder)
    seq_folder_path = os.path.join(dataset_folder, seq_folder)
    lidar_360_directory = os.path.join(seq_folder_path, "lidar_360")
    lidar_360_data = read_lidar_files(lidar_360_directory)


    output_folder =  os.path.join(seq_folder_path, "lidar_360_processed")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    total_frame = len(lidar_360_data)
    accumulated_lidar_360_data = {}
    accumulated_frame_timestamps = []
    accumulated_all_timestamps = {}
    accumulated_data = np.array([])

    for timestamp, data in lidar_360_data.items():
        #accumulated_all_timestamps.append(timestamp)
        accumulated_frame_timestamps.append(timestamp)
        mask = np.any(data != 0, axis = 1)
        filtered_data = data[mask]
        frame_ind = len(accumulated_frame_timestamps)*np.ones([filtered_data.shape[0],1])
        data_with_ind = np.concatenate((frame_ind, filtered_data), axis = 1)
        if np.size(accumulated_data) ==0:
            accumulated_data = data_with_ind
        else:
            accumulated_data = np.concatenate((accumulated_data, data_with_ind), axis=0)
        if len(accumulated_frame_timestamps) == 20:
            accumulated_lidar_360_data[accumulated_frame_timestamps[-1]] = np.array(accumulated_data)
            accumulated_all_timestamps[accumulated_frame_timestamps[-1]] = accumulated_frame_timestamps
            accumulated_frame_timestamps = []
            accumulated_data = np.array([])

# accumulated_all_timestamps = accumulated_all_timestamps[:-len(accumulated_frame_timestamps)]
#print(len(accumulated_all_timestamps))
# process last frames
    accumulated_frame_timestamps = []
    accumulated_data = np.array([])
    for idx, (timestamp, data) in enumerate(lidar_360_data.items()):
        if idx >= total_frame-20:
        #accumulated_all_timestamps.append(timestamp)
            accumulated_frame_timestamps.append(timestamp)
            mask = np.any(data != 0, axis = 1)
            filtered_data = data[mask]
            frame_ind = len(accumulated_frame_timestamps)*np.ones([filtered_data.shape[0],1])
            data_with_ind = np.concatenate((frame_ind, filtered_data), axis = 1)
            if np.size(accumulated_data) ==0:
                accumulated_data = data_with_ind
            else:
                accumulated_data = np.concatenate((accumulated_data, data_with_ind), axis=0)        
            if len(accumulated_frame_timestamps) == 20:
                accumulated_lidar_360_data[accumulated_frame_timestamps[-1]] = np.array(accumulated_data)
                accumulated_all_timestamps[accumulated_frame_timestamps[-1]] = accumulated_frame_timestamps


    for timestamp, data_with_ind in accumulated_lidar_360_data.items():
        frame_time = accumulated_all_timestamps[timestamp]
        point_cloud_data = {key:[] for key in frame_time}
        time_ind = data_with_ind[:,0]
        data = data_with_ind[:,1:]
        if np.size(data,0) != 0: 
            try:
                db = DBSCAN(eps=2, min_samples=10).fit(data)
           
            except AttributeError:
                pass
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)
            feature_set, cluster_labels = extract_feature_set_predict(data, labels, time_ind)
            predicted = model.detect(feature_set, model_path)
            predicted_labels = cluster_labels[predicted==1]
            for predicted_label in predicted_labels:
                det_data =  data[labels == predicted_label]
                det_time_ind = time_ind[labels == predicted_label].astype(int)-1
                #time_name = [frame_time[i] for i in det_time_ind]
                for ind, i in enumerate(det_time_ind):
                    point_cloud_data[frame_time[i]].append(det_data[ind].tolist())
                
            for frame_name in frame_time:
                output_file_path = os.path.join(output_folder, frame_name+'.npy')
                saved_pts = point_cloud_data[frame_name]
                np.save(output_file_path, np.array(saved_pts))
