
<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center">PLGSLAM: Progressive Neural Scene Represenation with Local to Global Bundle Adjustment</h1>
  <h3 align="center">CVPR 2024 </h3>
  <h3 align="center"><a href="https://openaccess.thecvf.com/content/CVPR2024/papers/Deng_PLGSLAM_Progressive_Neural_Scene_Represenation_with_Local_to_Global_Bundle_CVPR_2024_paper.pdf">Paper</a> </h3>
  <div align="center"></div>
</p>

<p align="center">
  <a href="">
    <img src="./framework.png" alt="Logo" width="100%">
  </a>
</p>

<br>
<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#demo">Online Demo</a>
    </li>
    <li>
      <a href="#usage">Usage</a>
    </li>
    <li>
      <a href="#downloads">Downloads</a>
    </li>
    <li>
      <a href="#benchmarking">Benchmarking</a>
    </li>
    <li>
      <a href="#acknowledgement">Acknowledgement</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
  </ol>
</details>
## Installation

##### (Recommended Environment)
Our method has been tested on python 3.10, CUDA>=11.6. The simplest way to install all dependences is to use [anaconda](https://www.anaconda.com/) and [pip](https://pypi.org/project/pip/) in the following steps: 

```bash
conda create -n MMUAV python=3.10
conda activate MMUAV
conda install -c "nvidia/label/cuda-11.6.0" cuda-toolkit
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r requirements.txt
```

<!-- Alternatively, we also provide a conda environment.yml file :
```bash
conda env create -f environment.yml
conda activate MMUAV
``` -->

## Drone Detection and Classification

### Preprocess the  dataset by Yolov9

Install Yolov9 Dependency
```bash
> pip install yolov9/requirements.txt
```
And download pretrained weight of Yolov9 under `./yolov9`.
Then Prepare the dataset utilizing the zero-shot UAV detection by Yolov9
```bash
> python3 yolov9/detect.py --source $SOURCE --img 640 --device 0 --weights './yolov9-e.pt' --name $OUTPUT --num_kf 5 --kf_int 50 --save-crop
```
Replace `$SOURCE` and `$OUTPUT` to the source and desired output directories of the target trajectory. `num_kf` is the maximum number of keyframe selected from the trajectory. `kf_int` is the interval threshold between each keyframe to prevent selecting similar keyframes. `save-crop` will save the cropped image for detected drones, necessary for the training of efficientnet.



# LiDAR Point Cloud Processing Pipeline

This section introduces Python scripts for processing LiDAR point cloud data, tracking objects, and visualizing trajectories. The pipeline consists of three main stages: preprocessing, fusion tracking, and postprocessing. Each stage is executed via a separate script with configurable command-line arguments.

## Prerequisites

- **Python 3.x**: Ensure Python is installed on your system.
- **Required Libraries**: Install the necessary dependencies by running:
  ```bash
  pip install numpy sklearn torch matplotlib scipy stonesoup
  ```
- **Custom Modules**: The scripts rely on additional modules (`dataset_loader.py`, `extract_feature.py`, `lidar_360_detector.py`). Ensure these are available in your project directory.
- **Input Data**: LiDAR data should be organized in `.npy` files within the specified dataset folder.

## Directory Structure

- **Input Data**: Place your LiDAR data in a folder (e.g., `C:\Users\Yi\Desktop\val`) with subfolders for sequences (e.g., `seq1`).
- **Output Folders**: Results and figures will be saved to specified directories (e.g., `C:\Users\Yi\Desktop\result`, `C:\Users\Yi\Desktop\figure`).

## Pipeline Stages

### 1. Preprocessing (`preprocess.py`)
This script processes raw LiDAR data from Livox Avia and 360-degree LiDAR sensors, performs clustering, and fuses the results.

#### Command
```bash
python .\point_cloud_processing\tracker\preprocess.py --dataset_folder "C:\Users\Yi\Desktop\val"
```

#### Description
- **Purpose**: Reads LiDAR data, downsamples it (if necessary), applies clustering (DBSCAN), and uses an LSTM model for detection. The processed data is saved in subfolders (`livox_avia_processed`, `lidar_360_processed`, `lidar_fusion`).
- **Arguments**:
  - `--dataset_folder`: Path to the folder containing sequence subfolders (default: `C:\Users\Yi\Desktop\val`).
  - `--model_path`: (Optional) Path to the LSTM model file. Defaults to `lstm_model.pth` in the script directory if not specified.
- **Output**: Processed `.npy` files in the sequence subfolders.

#### Notes
- Ensure the dataset folder contains subfolders with `livox_avia` and `lidar_360` directories.
- The script skips non-directory entries and creates output folders as needed.

---

### 2. Fusion Tracking (`fusion_tracking.py`)
This script performs object tracking on the fused LiDAR data using a Kalman filter-based tracker from the Stone Soup library.

#### Command
```bash
python .\point_cloud_processing\tracker\fusion_tracking.py --dataset_folder "C:\Users\Yi\Desktop\val" --result_folder "C:\Users\Yi\Desktop\result" --sequence "seq1"
```

#### Description
- **Purpose**: Clusters the fused point cloud data (using DBSCAN), tracks objects over time with an Extended Kalman Filter, and saves the tracked states.
- **Arguments**:
  - `--dataset_folder`: Path to the folder containing the preprocessed data (e.g., `C:\Users\Yi\Desktop\val`).
  - `--result_folder`: Path to save the tracking results (e.g., `C:\Users\Yi\Desktop\result`).
  - `--sequence`: Specific sequence to process (e.g., `seq1`). If omitted, all sequences in the dataset folder are processed.
  - `--eps`: DBSCAN epsilon parameter (default: `1.0`).
  - `--min_samples`: DBSCAN minimum samples (default: `1`).
  - `--noise_covar`: Measurement noise covariance (default: `0.001`).
  - `--missed_distance`: Distance threshold for missed detections (default: `3.0`).
  - `--covar_trace_thresh`: Covariance threshold for track deletion (default: `30.0`).
  - `--min_points`: Minimum points for track initiation (default: `1`).
- **Output**: `.npy` files containing tracked object positions in the result folder (e.g., `C:\Users\Yi\Desktop\result\seq1`).

#### Notes
- The script processes the `lidar_fusion` subfolder from the preprocessing step.
- Adjust tunable parameters (e.g., `--eps`, `--missed_distance`) based on your data characteristics.

---

### 3. Postprocessing (`postprocess.py`)
This script interpolates tracked trajectories and generates visualizations of positions and velocities.

#### Command
```bash
python .\point_cloud_processing\tracker\postprocess.py --dataset_folder "C:\Users\Yi\Desktop\result" --figure_folder "C:\Users\Yi\Desktop\figure" --seq_folder "seq1" --spline_smoothing 0.7
```

#### Description
- **Purpose**: Loads tracked data, interpolates trajectories using spline interpolation, and plots the results (positions and velocities) in a figure.
- **Arguments**:
  - `--dataset_folder`: Path to the folder containing tracking results (e.g., `C:\Users\Yi\Desktop\result`).
  - `--figure_folder`: Path to save the output figures (e.g., `C:\Users\Yi\Desktop\figure`).
  - `--seq_folder`: Sequence folder name to process (e.g., `seq1`).
  - `--spline_smoothing`: Smoothing factor for spline interpolation (default: `0.5`, example: `0.7`).
- **Output**: A `.png` file (e.g., `seq1.png`) in the figure folder, showing X/Y/Z positions and velocities.

#### Notes
- The script uses the tracking results from the previous step.
- Adjust `--spline_smoothing` to control the smoothness of the interpolated trajectory.

---

## Full Workflow Example
To process a single sequence (`seq1`) from start to finish:

1. **Preprocess**:
   ```bash
   python .\point_cloud_processing\tracker\preprocess.py --dataset_folder "C:\Users\Yi\Desktop\val"
   ```
2. **Track**:
   ```bash
   python .\point_cloud_processing\tracker\fusion_tracking.py --dataset_folder "C:\Users\Yi\Desktop\val" --result_folder "C:\Users\Yi\Desktop\result" --sequence "seq1"
   ```
3. **Postprocess**:
   ```bash
   python .\point_cloud_processing\tracker\postprocess.py --dataset_folder "C:\Users\Yi\Desktop\result" --figure_folder "C:\Users\Yi\Desktop\figure" --seq_folder "seq1" --spline_smoothing 0.7
   ```

