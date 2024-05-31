import os
import numpy as np
from datetime import datetime, timedelta
from stonesoup.models.transition.linear import ConstantVelocity, CombinedLinearGaussianTransitionModel
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater

from stonesoup.types.state import GaussianState, State
from stonesoup.types.detection import Detection
from stonesoup.types.update import GaussianStateUpdate
from stonesoup.types.track import Track
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.initiator.simple import SimpleMeasurementInitiator
from stonesoup.measures import Euclidean
from stonesoup.dataassociator.neighbour import NearestNeighbour
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.tracker.simple import SingleTargetTracker
from stonesoup.deleter.error import CovarianceBasedDeleter

from stonesoup.initiator.simple import MultiMeasurementInitiator



from matplotlib import pyplot as plt

from sklearn.cluster import DBSCAN

from dataset_loader import extract_timestamp, read_lidar_files

dataset_folder = "/media/yi/KESU/anti_uav/reorg_data"
figure_folder = "/media/yi/KESU/anti_uav/figure/png/tracking_plot_with_gt"


def point_cloud_detector(filtered_data):
    db = DBSCAN(eps=1, min_samples=1).fit(filtered_data)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    unique_labels = set(labels)

    cluster_centers = []
    for k in unique_labels:
        if k == -1:
            continue
        class_mask = (labels == k)
        cluster_center = np.mean(filtered_data[class_mask], axis=0)
        cluster_centers.append(cluster_center)
    cluster_centers = np.array(cluster_centers)
    return cluster_centers

for seq_folder in os.listdir(dataset_folder):
    print("Processing "+seq_folder)

    seq_folder_path = os.path.join(dataset_folder, seq_folder)

    # initial state
    gt_directory = os.path.join(seq_folder_path, "gt")
    gt_data = read_lidar_files(gt_directory)

    lidar_directory = os.path.join(seq_folder_path, "lidar")
    lidar_data = read_lidar_files(lidar_directory)

    for timestamp, data in lidar_data.items():
        if data.size != 0:
            start_time = timestamp
            timestamp_float = float(timestamp)
            seconds = int(timestamp_float)
            microseconds = int((timestamp_float - seconds)*1e6)
            start_datatime_object = datetime.fromtimestamp(seconds) + timedelta(microseconds = microseconds)
            break
        
    prior = GaussianState([[data[0][0]], [0.001], [data[0][1]], [0.001], [data[0][2]], [0.001]], 
                      np.diag([0.01, 0.1, 0.01, 0.1, 0.01, 0.1]), 
                      timestamp=start_datatime_object)
    
    meas_model = LinearGaussian(
        ndim_state=6,  # Number of state dimensions (position and velocity in 2D)
        mapping=(0, 2, 4),  # Mapping measurement vector index to state index
        noise_covar=np.array([[0.001, 0, 0],  # Covariance matrix for Gaussian PDF
                          [0, 0.001, 0],
                          [0, 0, 0.001]]))
    
    all_measurements = []
    for timestamp, data in lidar_data.items():
        measurement_set = set()
        filtered_data = data
        if filtered_data.size != 0:
            cluster_data = point_cloud_detector(filtered_data)
            #if cluster_data.shape[0] > 1:
            timestamp_float = float(timestamp)
            seconds = int(timestamp_float)
            microseconds = int((timestamp_float - seconds)*1e6)
            datetime_object = datetime.fromtimestamp(seconds) + timedelta(microseconds = microseconds)
            for detection in cluster_data:
                measurement_set.add(Detection(detection.transpose(),
                                  timestamp=datetime_object,
                                  measurement_model=meas_model))
            all_measurements.append(measurement_set)

    transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(0.15),
     ConstantVelocity(0.15),
     ConstantVelocity(0.15)])

    predictor = ExtendedKalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model=meas_model)


    deleter = CovarianceBasedDeleter(covar_trace_thresh=30)

    meas = Euclidean()
    hypothesiser = DistanceHypothesiser(predictor, updater, meas, missed_distance=3)
    data_associator = NearestNeighbour(hypothesiser)

    #initiator = SimpleMeasurementInitiator(prior, meas_model)
    initiator = MultiMeasurementInitiator(prior_state= prior,
        measurement_model=meas_model, deleter=deleter, data_associator=data_associator, 
                                      updater=updater,min_points=1)

    tracks = set()
    for i, measurements in enumerate(all_measurements):
    # Calculate all hypothesis pairs and associate the elements in the best subset to the tracks.
        for det in measurements:
            timestamp = det.timestamp
            break
        hypotheses = data_associator.associate(tracks,
                                           measurements,
                                           timestamp)
        associated_measurements = set()
        for track in tracks:
            hypothesis = hypotheses[track]
            if hypothesis.measurement:
                post = updater.update(hypothesis)
                track.append(post)
                associated_measurements.add(hypothesis.measurement)
            else:  # When data associator says no detections are good enough, we'll keep the prediction
                track.append(hypothesis.prediction)

        # Carry out deletion and initiation
        tracks -= deleter.delete_tracks(tracks)
        tracks |= initiator.initiate(measurements - associated_measurements, timestamp)
        #print(f"the number of detected points is {len(measurements)}, and there are {len(tracks)} tracks at frame {i}")

    accumulated_gt = np.array([])

    for timestamp, data in gt_data.items():
        if np.size(accumulated_gt) ==0:
            accumulated_gt = data
        else:
            accumulated_gt = np.concatenate((accumulated_gt, data), axis=0)
    
    accumulated_gt = np.array(accumulated_gt)

    gt_x_list = []
    gt_y_list = []
    gt_z_list = []

    for t, measurement in enumerate(all_measurements):
    #print(measurement.state_vector)
        for m in measurement:
            x = m.state_vector[0]
            y = m.state_vector[1]
            z = m.state_vector[2]
            gt_x_list.append([t, x])
            gt_y_list.append([t, y])
            gt_z_list.append([t, z])

    time = [data[0] for data in gt_x_list]
    gt_x = [data[1] for data in gt_x_list]
    gt_y = [data[1] for data in gt_y_list]
    gt_z = [data[1] for data in gt_z_list]

    fig = plt.figure(figsize = (16,4))
    plt.subplot(1,3,1)
    plt.plot(time,gt_x, 'rx', markersize=5)
    plt.subplot(1,3,2)
    plt.plot(time,gt_y, 'rx', markersize=5)
    plt.subplot(1,3,3)
    plt.plot(time,gt_z, 'rx', markersize=5)


    for track in tracks:
        x_list = []
        y_list = []
        z_list = []
        for t in track:
            x = t.state_vector[0]
            y = t.state_vector[2]
            z = t.state_vector[4]
            x_list.append(x)
            y_list.append(y)
            z_list.append(z)
        plt.subplot(1,3,1)
        plt.plot(range(len(track)),x_list,'.')
        plt.subplot(1,3,2)
        plt.plot(range(len(track)),y_list,'.')
        plt.subplot(1,3,3)
        plt.plot(range(len(track)),z_list,'.')

    plt.suptitle(seq_folder)
    plt.savefig(os.path.join(figure_folder, seq_folder+'.png'))
    plt.close
