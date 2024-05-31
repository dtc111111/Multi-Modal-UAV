import os
import numpy as np

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

  # Function to process files in a folder
def data_filter(input_folder, output_folder):
    max_pts = 100
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all .npy files in the input folder
    files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]

    # Process each file
    for file in files:
        # Load the array from the .npy file
        file_path = os.path.join(input_folder, file)
        livox_avia_data = np.load(file_path)
        mask = np.any(livox_avia_data != 0, axis = 1)
        filtered_data = livox_avia_data[mask]
        if (filtered_data.shape[0] > max_pts):
            print(f"Downsample to max_pts {max_pts}")
            filtered_data = farthest_point_sample(filtered_data, max_pts)
        # Save the reshaped array to the output folder with the same filename
        output_file_path = os.path.join(output_folder, file)
        np.save(output_file_path, filtered_data)

        #print(f"Processed file: {file}")

# Example usage
dataset_folder = "/media/yi/KESU/anti_uav/val"
#dataset_folder = "/media/yi/Backup Plus/BaiduNetdiskDownload/test/test"
for seq_folder in os.listdir(dataset_folder):
    print("Processing "+seq_folder)
    seq_folder_path = os.path.join(dataset_folder, seq_folder)
    input_folder = os.path.join(seq_folder_path, "livox_avia")
    output_folder = os.path.join(seq_folder_path, "livox_avia_processed")
    data_filter(input_folder, output_folder)
