import os
import argparse
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

import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
from typing import Dict, List, Tuple

# Assuming these are in your project
from extract_feature import extract_feature_set_predict
from dataset_loader import read_lidar_files
from lidar_360_detector import MyLSTMClassifier

import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from dataset_loader import extract_timestamp, read_lidar_files


def point_cloud_detector(filtered_data, eps=1, min_samples=1):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(filtered_data)
    labels = db.labels_
    unique_labels = set(labels)
    cluster_centers = []

    for k in unique_labels:
        if k == -1:
            continue
        class_mask = (labels == k)
        cluster_center = np.mean(filtered_data[class_mask], axis=0)
        cluster_centers.append(cluster_center)

    return np.array(cluster_centers)


def process_sequence(seq_folder_path,
                     result_folder_path,
                     eps, min_samples,
                     noise_covar,
                     missed_distance,
                     covar_trace_thresh,
                     min_points):
    print(f"Processing {os.path.basename(seq_folder_path)}")


    lidar_directory = os.path.join(seq_folder_path, "lidar_fusion")
    lidar_data = read_lidar_files(lidar_directory)

    # Find start time
    for timestamp, data in lidar_data.items():
        if data.size != 0:
            timestamp_float = float(timestamp)
            seconds = int(timestamp_float)
            microseconds = int((timestamp_float - seconds) * 1e6)
            start_datetime = datetime.fromtimestamp(seconds) + timedelta(microseconds=microseconds)
            break

    # Initialize prior state
    if lidar_data:
        initial_data = next(iter(lidar_data.values()))
        if initial_data.size > 0:
            prior = GaussianState(
                [[initial_data[0][0]], [0.001], [initial_data[0][1]], [0.001], [initial_data[0][2]], [0.001]],
                np.diag([0.01, 0.1, 0.01, 0.1, 0.01, 0.1]),
                timestamp=start_datetime
            )
        else:
            print(f"No initial data for {seq_folder_path}, skipping")
            return
    else:
        print(f"No LiDAR data for {seq_folder_path}, skipping")
        return

    # Measurement model
    meas_model = LinearGaussian(
        ndim_state=6,
        mapping=(0, 2, 4),
        noise_covar=np.array([
            [noise_covar, 0, 0],
            [0, noise_covar, 0],
            [0, 0, noise_covar]
        ])
    )

    # Create measurements
    all_measurements = []
    for timestamp, data in lidar_data.items():
        measurement_set = set()
        if data.size != 0:
            cluster_data = point_cloud_detector(data, eps=eps, min_samples=min_samples)
            timestamp_float = float(timestamp)
            seconds = int(timestamp_float)
            microseconds = int((timestamp_float - seconds) * 1e6)
            datetime_object = datetime.fromtimestamp(seconds) + timedelta(microseconds=microseconds)

            for detection in cluster_data:
                measurement_set.add(Detection(
                    detection.transpose(),
                    timestamp=datetime_object,
                    measurement_model=meas_model
                ))
            all_measurements.append(measurement_set)

    # Transition model
    transition_model = CombinedLinearGaussianTransitionModel([
        ConstantVelocity(0.15),
        ConstantVelocity(0.15),
        ConstantVelocity(0.15)
    ])

    # Predictor and updater
    predictor = ExtendedKalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model=meas_model)

    # Tracker components
    deleter = CovarianceBasedDeleter(covar_trace_thresh=covar_trace_thresh)
    meas = Euclidean()
    hypothesiser = DistanceHypothesiser(predictor, updater, meas, missed_distance=missed_distance)
    data_associator = NearestNeighbour(hypothesiser)

    initiator = MultiMeasurementInitiator(
        prior_state=prior,
        measurement_model=meas_model,
        deleter=deleter,
        data_associator=data_associator,
        updater=updater,
        min_points=min_points
    )

    # Tracking
    tracks = set()
    for i, measurements in enumerate(all_measurements):
        if not measurements:
            continue

        timestamp = next(iter(measurements)).timestamp
        hypotheses = data_associator.associate(tracks, measurements, timestamp)
        associated_measurements = set()

        for track in tracks.copy():
            hypothesis = hypotheses[track]
            if hypothesis.measurement:
                post = updater.update(hypothesis)
                track.append(post)
                associated_measurements.add(hypothesis.measurement)
            else:
                track.append(hypothesis.prediction)


        # Deletion and initiation
        tracks -= deleter.delete_tracks(tracks)
        tracks |= initiator.initiate(measurements - associated_measurements, timestamp)

        #print(f"the number of detected points is {len(measurements)}, and there are {len(tracks)} tracks at frame {i}")
    #for i, track in enumerate(tracks):
    #    print("saved track:", i, "with lenght ", len(track))
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
    for track in tracks:
        prediction_list = []
        #print(len(track))
        for t in track:
            prediction = {
                    'timestamp': t.timestamp.timestamp(),
                    'state_vector': np.array(t.state_vector).tolist(),
                    'covar': np.array(t.covar).tolist()}
            prediction_list.append(prediction)
            xyz = np.array([t.state_vector[0], t.state_vector[2], t.state_vector[4]])
            np.save(os.path.join(result_folder_path, str(t.timestamp.timestamp()) + '.npy'), xyz)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LiDAR Tracking Processor')

    # Required arguments
    parser.add_argument('--dataset_folder', type=str, required=True,
                        help='Path to dataset folder')
    parser.add_argument('--result_folder', type=str, required=True,
                        help='Path to result folder')
    parser.add_argument('--sequence', type=str, default=None,
                        help='Specific sequence to process (optional)')

    # Tunable parameters
    parser.add_argument('--eps', type=float, default=1.0,
                        help='DBSCAN epsilon parameter (default: 1.0)')
    parser.add_argument('--min_samples', type=int, default=1,
                        help='DBSCAN minimum samples parameter (default: 1)')
    parser.add_argument('--noise_covar', type=float, default=0.001,
                        help='Measurement noise covariance (default: 0.001)')
    parser.add_argument('--missed_distance', type=float, default=3.0,
                        help='Missed distance threshold (default: 3.0)')
    parser.add_argument('--covar_trace_thresh', type=float, default=30.0,
                        help='Covariance trace threshold for deletion (default: 30.0)')
    parser.add_argument('--min_points', type=int, default=1,
                        help='Minimum points for track initiation (default: 1)')

    args = parser.parse_args()


    if args.sequence:
        seq_folder_path = os.path.join(args.dataset_folder, args.sequence)
        seq_result_path = os.path.join(args.result_folder, args.sequence)
        if os.path.exists(seq_folder_path):
            process_sequence(
                seq_folder_path,
                seq_result_path,
                args.eps,
                args.min_samples,
                args.noise_covar,
                args.missed_distance,
                args.covar_trace_thresh,
                args.min_points
            )
        else:
            print(f"Sequence folder {seq_folder_path} does not exist")
    else:
        for seq_folder in os.listdir(args.dataset_folder):
            seq_folder_path = os.path.join(args.dataset_folder, seq_folder)
            seq_result_path = os.path.join(args.result_folder_path, seq_folder)
            if os.path.isdir(seq_folder_path):
                process_sequence(
                    seq_folder_path,
                    seq_result_path,
                    args.eps,
                    args.min_samples,
                    args.noise_covar,
                    args.missed_distance,
                    args.covar_trace_thresh,
                    args.min_points
                )