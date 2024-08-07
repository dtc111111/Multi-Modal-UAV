
<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center">Multi-Modal UAV Detection, Classification, and Tracking Algorithm</h1>
  <h3 align="center">CVPR 2024 UG2 Challenge Champion</h3>
  <h3 align="center"><a href="https://arxiv.org/pdf/2405.16464">Paper</a> </h3>
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
> pip install yolov9/requirements.txt

And download pretrained weight of Yolov9 under `./yolov9`.
Then Prepare the dataset utilizing the zero-shot UAV detection by Yolov9
> python3 yolov9/detect.py --source $SOURCE --img 640 --device 0 --weights './yolov9-e.pt' --name $OUTPUT --num_kf 5 --kf_int 50 --save-crop

Replace `$SOURCE` and `$OUTPUT` to the source and desired output directories of the target trajectory. `num_kf` is the maximum number of keyframe selected from the trajectory. `kf_int` is the interval threshold between each keyframe to prevent selecting similar keyframes. `save-crop` will save the cropped image for detected drones, necessary for the training of efficientnet.

