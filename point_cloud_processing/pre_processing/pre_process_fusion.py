import os
import numpy as np
from matplotlib import pyplot as plt
from dataset_loader import extract_timestamp, read_lidar_files
from sklearn.cluster import DBSCAN


dataset_folder = "/media/yi/Backup Plus/BaiduNetdiskDownload/test/test"
figure_folder = "/media/yi/Backup Plus/BaiduNetdiskDownload/test/figure/fusion_plot"
if not os.path.exists(figure_folder):
    os.makedirs(figure_folder)

for seq_folder in os.listdir(dataset_folder):
    print("Processing "+seq_folder)
    seq_folder_path = os.path.join(dataset_folder, seq_folder)

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
    #accumulated_all_timestamps.append(timestamp)
        if data.size != 0:
            accumulated_timestamps.append(timestamp)
            mask = np.any(data != 0, axis = 1)
            filtered_data = data[mask]
            frame_ind = len(accumulated_timestamps)*np.ones([filtered_data.shape[0],1])
            data_with_ind = np.concatenate((frame_ind, filtered_data), axis = 1)
            if np.size(accumulated_data) ==0:
                accumulated_data = data_with_ind
            else:
                accumulated_data = np.concatenate((accumulated_data, data_with_ind), axis=0)
    
    accumulated_data = np.array(accumulated_data)
    print(accumulated_data.shape)

    if accumulated_data.shape[0] < 5e4:

        time_ind = accumulated_data[:,0]
        data = accumulated_data[:,1:]
        if np.size(data,0) != 0: 
            db = DBSCAN(eps=1, min_samples=10).fit(data) 
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)
            print("Estimated number of clusters: %d" % n_clusters_)
            print("Estimated number of noise points: %d" % n_noise_)
    
        detections = accumulated_data[labels != -1]

        point_cloud_data = {key:[] for key in accumulated_timestamps}
        det_data =  detections[:,1:]
        det_time_ind = detections[:,0].astype(int)-1
        for ind, i in enumerate(det_time_ind):
            point_cloud_data[accumulated_timestamps[i]].append(det_data[ind].tolist())
                
        for frame_name in accumulated_timestamps:
            output_file_path = os.path.join(output_folder, frame_name+'.npy')
            saved_pts = point_cloud_data[frame_name]
            np.save(output_file_path, np.array(saved_pts))

        fig = plt.figure()
        ax = fig.add_subplot(121, projection = '3d')
        ax.scatter(accumulated_data[:,1], accumulated_data[:,2], accumulated_data[:,3], s=1, c = 'b')
        ax = fig.add_subplot(122, projection = '3d')
        ax.scatter(detections[:,1], detections[:,2], detections[:,3], s=1, c = 'r')
        plt.suptitle(seq_folder)
        plt.savefig(os.path.join(figure_folder, seq_folder+'.png'))
        plt.close
    
    else:
        print("Cannot process "+seq_folder)
