import os
import numpy as np
import json
from dataset_loader import extract_timestamp, read_lidar_files
from datetime import datetime, timedelta
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.models.transition.linear import ConstantVelocity, CombinedLinearGaussianTransitionModel
from stonesoup.models.measurement.linear import LinearGaussian
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



dataset_folder = "/media/yi/KESU/anti_uav/train"
result_folder = "/media/yi/KESU/anti_uav/result/train"

for seq_folder in os.listdir(dataset_folder):
    print("Processing "+seq_folder)
    seq_folder_path = os.path.join(dataset_folder, seq_folder)
    result_folder_path = os.path.join(result_folder, seq_folder)

    # initial state
    gt_directory = os.path.join(seq_folder_path, "gt")
    gt_data = read_lidar_files(gt_directory)

    save_directory =  os.path.join(seq_folder_path, "gt_track")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    for timestamp, data in gt_data.items():
        start_time = timestamp
        timestamp_float = float(timestamp)
        seconds = int(timestamp_float)
        microseconds = int((timestamp_float - seconds)*1e6)
        start_datatime_object = datetime.fromtimestamp(seconds) + timedelta(microseconds = microseconds)
        #print(f"start time is {start_datatime_object}")
        break

    prior = GaussianState([[data[0][0]], [0.01], [data[0][1]], [0.01], [data[0][2]], [0.01]], 
                          np.diag([0.01, 0.1, 0.01, 0.1, 0.01, 0.1]), 
                          timestamp=start_datatime_object)

    meas_model = LinearGaussian(
        ndim_state=6,  # Number of state dimensions (position and velocity in 2D)
        mapping=(0, 2, 4),  # Mapping measurement vector index to state index
        noise_covar=np.array([[0.0001, 0, 0],  # Covariance matrix for Gaussian PDF
                              [0, 0.0001, 0],
                              [0, 0, 0.0001]])
        )

    all_measurements = []
    for timestamp, data in gt_data.items():
        measurement_set = set()
        mask = np.any(data != 0, axis = 1)
        filtered_data = data[mask]
        timestamp_float = float(timestamp)
        seconds = int(timestamp_float)
        microseconds = int((timestamp_float - seconds)*1e6)
        datetime_object = datetime.fromtimestamp(seconds) + timedelta(microseconds = microseconds)
        for detection in filtered_data:
            measurement_set.add(Detection(detection.transpose(),
                                      timestamp=datetime_object,
                                      measurement_model=meas_model))
        all_measurements.append(measurement_set)

    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(0.15),
         ConstantVelocity(0.15),
         ConstantVelocity(0.5)])

    predictor = ExtendedKalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(measurement_model=meas_model)

    initiator = SimpleMeasurementInitiator(prior, meas_model)

    class MyDeleter:
        def delete_tracks(self, tracks):
            return set()

    deleter = MyDeleter()
    meas = Euclidean()
    hypothesiser = DistanceHypothesiser(predictor, updater, meas)
    data_associator = NearestNeighbour(hypothesiser)

    tracks = set()

    for measurements in all_measurements:
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
        tracks |= initiator.initiate(measurements - associated_measurements,
                                 timestamp)
        
    # track_fusion
    print(len(tracks))
    for track in tracks:
        prediction_list = []
        for t in track:
            prediction = {
            'timestamp': t.timestamp.timestamp(),
            'state_vector': np.array(t.state_vector).tolist(),
            'covar': np.array(t.covar).tolist()}
            prediction_list.append(prediction)
            xyz = np.array([t.state_vector[0], t.state_vector[2], t.state_vector[4]])
            np.save(os.path.join(save_directory, str(t.timestamp.timestamp())+'.npy'), xyz)
        with open(os.path.join(result_folder_path+'state.json'),'w') as file:
            json.dump(prediction_list, file)
        
