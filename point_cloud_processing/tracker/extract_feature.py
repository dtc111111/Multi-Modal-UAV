import numpy as np

def find_gt(gt_data, timestamp_lidar):
    time_diff_min = np.inf
    for timestamp, data in gt_data.items():
        time_diff = np.abs(float(timestamp_lidar) - float(timestamp))
        if time_diff < time_diff_min:
            gt_data = data
            time_diff_min = time_diff
    return gt_data

def extract_feature_set(data, labels, time_ind, gt):
    unique_labels = set(labels)
    unique_time_ind = set(time_ind)
    feature_set_list = []
    label_cluster_list = []
    for k in unique_labels:
        if k == -1:
            # Black used for noise.
            continue
        feature_set = np.array([])
        class_member_mask = labels == k
        for time in unique_time_ind:
            time_mask = time_ind == time
            masked_data = data[class_member_mask & time_mask]
            if np.size(masked_data) != 0:
                xyz_mean = np.mean(masked_data, axis=0)
                xyz_std = np.std(masked_data, axis=0)
                xyz_range = np.max(masked_data, axis=0) - np.min(masked_data, axis=0)
            else:
                xyz_mean = np.zeros(3)
                xyz_std = np.zeros(3)
                xyz_range = np.zeros(3)
            feature = np.concatenate((xyz_mean, xyz_std, xyz_range), axis = 0).reshape(1,-1)
            if np.size(feature_set) == 0:
                feature_set = feature
            else:
                feature_set = np.vstack([feature_set, feature])
        feature_set_list.append(feature_set)
        
    for k in unique_labels:
        if k == -1:
            continue
        label_cluster = 0
        class_member_mask = labels == k
        for time in unique_time_ind:
            time_mask = time_ind == time
            masked_data = data[class_member_mask & time_mask]
            if np.size(masked_data) != 0:
                cluster_mean = np.mean(masked_data, axis=0)
                dist_temp = np.linalg.norm(cluster_mean - gt)
                if dist_temp < 1:
                    label_cluster = 1
                
        label_cluster_list.append(label_cluster)

    
    feature_set_all = np.stack(feature_set_list, axis = 0)
    label_set_all = np.array(label_cluster_list)
    return feature_set_all, label_set_all



def extract_feature_set_predict(data, labels, time_ind):
    unique_labels = set(labels)
    unique_time_ind = set(time_ind)
    feature_set_list = []
    cluster_label_list = []
    for k in unique_labels:
        if k == -1:
            # Black used for noise.
            continue
        feature_set = np.array([])
        class_member_mask = labels == k
        for time in unique_time_ind:
            time_mask = time_ind == time
            masked_data = data[class_member_mask & time_mask]
            if np.size(masked_data) != 0:
                xyz_mean = np.mean(masked_data, axis=0)
                xyz_std = np.std(masked_data, axis=0)
                xyz_range = np.max(masked_data, axis=0) - np.min(masked_data, axis=0)
            else:
                xyz_mean = np.zeros(3)
                xyz_std = np.zeros(3)
                xyz_range = np.zeros(3)
            feature = np.concatenate((xyz_mean, xyz_std, xyz_range), axis = 0).reshape(1,-1)
            if np.size(feature_set) == 0:
                feature_set = feature
            else:
                feature_set = np.vstack([feature_set, feature])
        feature_set_list.append(feature_set)
        cluster_label_list.append(k)

    feature_set_all = np.stack(feature_set_list, axis = 0)
    cluster_label_set_all = np.array(cluster_label_list).reshape(-1,1)
    return feature_set_all, cluster_label_set_all
