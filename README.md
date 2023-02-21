# Restoration of Analog Videos Using Swin-UNet

## Table of Contents


* [About the Project](#about-the-project)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
 * [Training](#training)
 * [Test](#test) 
* [Authors](#authors)
* [Citation](#citation)

## About The Project
![restoration example](readme.png)


This is the **official repository** of "[**Restoration of Analog Videos Using Swin-UNet**](https://dl.acm.org/doi/10.1145/3503161.3547730)" **[Demo ACM MM 2022]**.

In this work, we present an approach to restore analog videos of historical archives. These videos often contain severe visual degradation due to the deterioration of their tape supports that require costly and slow manual interventions to recover the original content. The proposed method uses a multi-frame approach and is able to deal also with severe tape mistracking, which results in completely scrambled frames. Tests on real-world videos from a major historical video archive show the effectiveness of our approach.


## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

We strongly recommend the use of the [**Anaconda**](https://www.anaconda.com/) package manager in order to avoid dependency/reproducibility problems.
A conda installation guide for Linux systems can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

### Installation
 
1. Clone the repo
```sh
git clone https://github.com/miccunifi/analog-video-restoration.git
```
2. Install Python dependencies
```sh
conda create -n analog_video_restoration -y python=3.9
conda activate analog_video_restoration
pip install -r requirements.txt
```

## Usage

### Training

### Test

1. If needed, download the pretrained model from [Google Drive](https://drive.google.com/drive/folders/1omIk6qHKqbvO7T09Ixiez7zq08S7OaxE?usp=share_link) and copy it inside the folder ```pretrained_models/video_swin_unet```

2. Extract the frames of the video in jpg images and save them in a folder
```
mkdir <folder-name>
ffmpeg -i <video-file-name> -qscale:v 2 <folder-name>/%00d.jpg
```

3. Run inference on the folder with
```
python src/real_world_test.py --experiment-name video_swin_unet --data-base-path <path-to-folder> --results-path results --patch-size 512 --fps 60
```

## Authors
* [**Lorenzo Agnolucci**](https://scholar.google.com/citations?user=hsCt4ZAAAAAJ&hl=en)
* [**Leonardo Galteri**](https://scholar.google.com/citations?user=_n2R2bUAAAAJ&hl=en)
* [**Marco Bertini**](https://scholar.google.it/citations?user=SBm9ZpYAAAAJ&hl=en)
* [**Alberto Del Bimbo**](https://scholar.google.it/citations?user=bf2ZrFcAAAAJ&hl=en)

## Citation
If you find this work useful for your research, please consider citing:
<pre>
@inproceedings{10.1145/3503161.3547730,
  author = {Agnolucci, Lorenzo and Galteri, Leonardo and Bertini, Marco and Del Bimbo, Alberto},
  title = {Restoration of Analog Videos Using Swin-UNet},
  year = {2022},
  isbn = {9781450392037},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3503161.3547730},
  doi = {10.1145/3503161.3547730},
  abstract = {In this paper we present a system to restore analog videos of historical archives. These videos often contain severe visual degradation due to the deterioration of their tape supports that require costly and slow manual interventions to recover the original content. The proposed system uses a multi-frame approach and is able to deal also with severe tape mistracking, which results in completely scrambled frames. Tests on real-world videos from a major historical video archive show the effectiveness of our demo system.},
  booktitle = {Proceedings of the 30th ACM International Conference on Multimedia},
  pages = {6985–6987},
  numpages = {3},
  keywords = {old videos restoration, analog videos, unet, swin transformer},
  location = {Lisboa, Portugal},
  series = {MM '22}
}
</pre>
