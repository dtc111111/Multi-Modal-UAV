import os
import numpy as np
from dataset_loader import extract_timestamp, read_lidar_files
from extract_feature import find_gt, extract_feature_set
from sklearn.cluster import DBSCAN

dataset_folder = "/media/yi/KESU/anti_uav/val"
result_folder = "/media/yi/KESU/anti_uav/result"
save_directory =  os.path.join(result_folder, "lidar_360_feature_set")
if not os.path.exists(save_directory):
    os.makedirs(save_directory)


feature_set_all = []
label_set_all = []

for seq_folder in os.listdir(dataset_folder):
    print("Processing "+seq_folder)
    seq_folder_path = os.path.join(dataset_folder, seq_folder)

    # initial state
    gt_directory = os.path.join(seq_folder_path, "gt")
    gt_data = read_lidar_files(gt_directory)

    lidar_360_directory = os.path.join(seq_folder_path, "lidar_360")
    lidar_360_data = read_lidar_files(lidar_360_directory)

    accumulated_lidar_360_data = {}
    accumulated_timestamps = []
    accumulated_data = np.array([])
    for timestamp, data in lidar_360_data.items():
        accumulated_timestamps.append(timestamp)
        mask = np.any(data != 0, axis = 1)
        filtered_data = data[mask]
        frame_ind = len(accumulated_timestamps)*np.ones([filtered_data.shape[0],1])
        data_with_ind = np.concatenate((frame_ind, filtered_data), axis = 1)
        if np.size(accumulated_data) ==0:
            accumulated_data = data_with_ind
        else:
            accumulated_data = np.concatenate((accumulated_data, data_with_ind), axis=0)
        if len(accumulated_timestamps) == 20:
            accumulated_lidar_360_data[accumulated_timestamps[0]] = np.array(accumulated_data)
            accumulated_timestamps = []
            accumulated_data = np.array([])

    for timestamp, data_with_ind in accumulated_lidar_360_data.items():
        time_ind = data_with_ind[:,0]
        data = data_with_ind[:,1:]
        if np.size(data,0) != 0: 
            try:
                db = DBSCAN(eps=1, min_samples=10).fit(data)
           
            except AttributeError:
                pass
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)
            #print("Estimated number of clusters: %d" % n_clusters_)
            #print("Estimated number of noise points: %d" % n_noise_)
            gt = find_gt(gt_data, timestamp)
            feature_set, label_set = extract_feature_set(data, labels, time_ind, gt)
            gt_cluster = np.median(feature_set[label_set == 1,:,:3],axis=1)
            if np.size(gt_cluster) !=0:
                print(f"cluster is {gt_cluster}\n")
                print(f"ground truth is {gt}\n")
            feature_set_all.append(feature_set)
            label_set_all.append(label_set)

feature_set_all = np.concatenate(feature_set_all, axis = 0)
label_set_all = np.concatenate(label_set_all, axis = 0)

np.save(os.path.join(save_directory, 'feature_val.npy'), feature_set_all)
np.save(os.path.join(save_directory, 'label_val.npy'), label_set_all)
