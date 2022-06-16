Created by [Martin Hahner](https://sites.google.com/view/martinhahner/home) at the [Computer Vision Lab](https://vision.ee.ethz.ch/) of [ETH Zurich](https://ethz.ch/).

[![Support Ukraine](https://img.shields.io/badge/Support-Ukraine-FFD500?style=flat&labelColor=005BBB)](https://opensource.fb.com/support-ukraine)  [![arXiv](https://img.shields.io/badge/arXiv-2203.15118-00ff00.svg)](https://arxiv.org/abs/2203.15118) ![visitors](https://visitor-badge.laobi.icu/badge?page_id=SysCV.LiDAR_snow_sim)

[![PapersWithCode](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lidar-snowfall-simulation-for-robust-3d/3d-object-detection-on-clear-weather)](https://paperswithcode.com/sota/3d-object-detection-on-clear-weather?p=lidar-snowfall-simulation-for-robust-3d) <br>
[![PapersWithCode](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lidar-snowfall-simulation-for-robust-3d/3d-object-detection-on-light-snowfall)](https://paperswithcode.com/sota/3d-object-detection-on-light-snowfall?p=lidar-snowfall-simulation-for-robust-3d) <br>
[![PapersWithCode](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/lidar-snowfall-simulation-for-robust-3d/3d-object-detection-on-stf-heavy-snowfall)](https://paperswithcode.com/sota/3d-object-detection-on-stf-heavy-snowfall?p=lidar-snowfall-simulation-for-robust-3d)

# 🌨 LiDAR Snowfall Simulation <br> for Robust 3D Object Detection

*by [Martin Hahner](https://www.trace.ethz.ch/team/members/martin.html), [Christos Sakaridis](https://www.trace.ethz.ch/team/members/christos.html), [Mario Bijelic](http://mariobijelic.de), [Felix Heide](https://www.cs.princeton.edu/~fheide/), [Fisher Yu](https://www.trace.ethz.ch/team/members/fisher.html),  [Dengxin Dai](https://www.trace.ethz.ch/team/members/dengxin.html), and [Luc van Gool](https://www.trace.ethz.ch/team/members/luc.html)* <br>

📣 Oral at [CVPR 2022](https://cvpr2022.thecvf.com/). <br>
Please visit our [paper website](https://trace.ethz.ch/lidar_snow_sim) for more details.

<img src="teaser.gif" width="850">

## Overview

    .
    ├── calib                     # contains the LiDAR sensor calibration file used in STF
    │   └── ...
    ├── lib                       # contains external libraries as submodules
    │   └── ...
    ├── splits                    # contains the splits we used for our experiments
    │   └── ...
    ├── tools                     # contains our snowfall and wet ground simulation code
    │   ├── snowfall
    │   │   ├── geometry.py
    │   │   ├── precompute.py
    │   │   ├── sampling.py
    │   │   └── simulation.py
    │   └── wet_ground
    │       ├── augmentation.py
    │       ├── phy_equations.py
    │       ├── planes.py
    │       └── utils.py
    ├── .gitignore
    ├── .gitmodules
    ├── LICENSE
    ├── pointcloud_viewer.py      # to visualize LiDAR point clouds and apply various augmentations
    ├── README.md
    └── teaser.gif

**Datasets supported by [pointcloud_viewer.py](pointcloud_viewer.py):**
- [H3D](https://usa.honda-ri.com/H3D)
- [A2D2](https://www.a2d2.audi/a2d2/en.html)
- [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
- [LyftL5](https://self-driving.lyft.com/level5/prediction/)
- [Pandaset](https://pandaset.org/)
- [nuScenes](https://www.nuscenes.org/nuscenes)
- [Argoverse](https://www.argoverse.org/data.html#tracking-link)
- [ApolloScape](http://apolloscape.auto/tracking.html)
- **[SeeingThroughFog](https://www.cs.princeton.edu/~fheide/AdverseWeatherFusion/)** &nbsp;:arrow_left: works best
- [WaymoOpenDataset](https://waymo.com/open/) (via [waymo_kitti_converter](https://github.com/caizhongang/waymo_kitti_converter))

**Note: <br> The snowfall and wet ground simulation is only tested on the [SeeingThroughFog](https://www.cs.princeton.edu/~fheide/AdverseWeatherFusion/) (STF) dataset.** <br>
To support other datasets as well, code changes are required.

## License

This software is made available for non-commercial use under a Creative Commons [License](LICENSE).<br>
A summary of the license can be found [here](https://creativecommons.org/licenses/by-nc/4.0/).


## Citation(s)

If you find this work useful, please consider citing our paper.
```bibtex
@inproceedings{HahnerCVPR22,
  author = {Hahner, Martin and Sakaridis, Christos and Bijelic, Mario and Heide, Felix and Yu, Fisher and Dai, Dengxin and Van Gool, Luc},
  title = {{LiDAR Snowfall Simulation for Robust 3D Object Detection}},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2022},
}
```
You may also want to check out our earlier work <br>
[*Fog Simulation on Real LiDAR Point Clouds for 3D Object Detection in Adverse Weather*](https://github.com/MartinHahner/LiDAR_fog_sim).

```bibtex
@inproceedings{HahnerICCV21,
  author = {Hahner, Martin and Sakaridis, Christos and Dai, Dengxin and Van Gool, Luc},
  title = {{Fog Simulation on Real LiDAR Point Clouds for 3D Object Detection in Adverse Weather}},
  booktitle = {IEEE International Conference on Computer Vision (ICCV)},
  year = {2021},
}
```

## Getting Started

### Setup

1) Install [anaconda](https://docs.anaconda.com/anaconda/install/).

2) Execute the following commands.
```bash
# Create a new conda environment.
conda create --name snowy_lidar python=3.9 -y

# Activate the newly created conda environment.
conda activate snowy_lidar

# Install dependencies.
conda install matplotlib pandas plyfile pyaml pyopengl pyqt pyqtgraph scipy scikit-learn tqdm -c conda-forge -y
pip install PyMieScatt pyquaternion

# Clone this repository (including submodules!).
git clone git@github.com:SysCV/LiDAR_snow_sim.git --recursive
cd LiDAR_snow_sim
```

3) If you want to use our precomputed snowflake patterns, you can download them (2.3GB) as mentioned below.
```bash
wget https://www.trace.ethz.ch/publications/2022/lidar_snow_simulation/snowflakes.zip
unzip snowflakes.zip
rm snowflakes.zip
```

4) If you want to use [DROR](https://ieeexplore.ieee.org/document/8575761) as well, <br>
you need to install [PCL](https://pointclouds.org/) or download the point indices (215MB) as mentioned below.
```bash
wget https://www.trace.ethz.ch/publications/2022/lidar_snow_simulation/DROR.zip
unzip DROR.zip
rm DROR.zip
```

5) Enjoy [pointcloud_viewer.py](pointcloud_viewer.py).
```bash
python pointcloud_viewer.py
```

6) If you also want to run inference on the [STF dataset](https://www.cs.princeton.edu/~fheide/AdverseWeatherFusion/), a couple of extra steps are required. <br>
Note: For unknown reasons, this can roughly slow down the augmentation(s) by a factor of two.
```bash
# Download our checkpoints (265MB)
wget https://www.trace.ethz.ch/publications/2022/lidar_snow_simulation/experiments.zip
unzip experiments.zip
rm experiments.zip

# Install PyTorch.
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c conda-forge -c pytorch -y

# Install spconv
pip install spconv-cu113

# build pcdet
cd lib/OpenPCDet
python setup.py develop
cd ../..
```


### Disclaimer

The code has been successfully tested on
- Ubuntu 18.04.6 LTS + CUDA 11.3 + conda 4.13.0
- Debian GNU/Linux 10 (buster) + conda 4.13.0
- MacOS Big Sur 11.6.6 + conda 4.13.0


## Contributions
Please feel free to suggest improvements to this repository.<br>
We are always open to merge useful pull request.

## Acknowledgments

**This work is supported by [Toyota](https://www.toyota-europe.com/) via the [TRACE](https://www.trace.ethz.ch/) project.**

The work also received funding by the [AI-SEE](https://www.ai-see.eu/) project with national funding from
- the [Austrian Research Promotion Agency (FFG)](https://www.ffg.at/),
- [Business Finland](https://www.businessfinland.fi/),
- [Federal Ministry of Education and Research (BMBF)](https://www.bmbf.de/bmbf/en/home/home_node.html) and
- [National Research Council of Canada Industrial Research Assistance Program (NRC-IRAP)](https://nrc.canada.ca/en/support-technology-innovation).

We also thank the [Federal Ministry for Economic Affairs and Energy](https://www.bmwi.de/Navigation/EN/Home/home.html) for support within <br>
[*VVM-Verification and Validation Methods for Automated Vehicles Level 4 and 5*](https://www.vvm-projekt.de/en/project), a [PEGASUS](https://pegasus-family.de/) family project.

[Felix Heide](https://www.cs.princeton.edu/~fheide/) was supported by an [NSF CAREER Award](https://www.cs.princeton.edu/news/national-science-foundation-awarded-professor-felix-heide-nsf-career-award) (2047359), <br>
a [Sony Young Faculty Award](https://www.sony.com/en/SonyInfo/research-award-program/), and a [Project X Innovation Award](https://aspire-report.princeton.edu/engineering/project-x-fund).

We thank Emmanouil Sakaridis for verifying our derivation of occlusion angles in our snowfall simulation.

[<img src="https://user-images.githubusercontent.com/14181188/160494058-9a965ac4-3ae3-4633-9d3c-25ef8462286f.png" height="40">](https://ethz.ch) &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; [<img src="https://user-images.githubusercontent.com/14181188/160494439-cca6665b-0732-4dda-90d9-1d3c77e7f6f8.png" height="40">](https://www.princeton.edu)  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; [<img src="https://user-images.githubusercontent.com/14181188/160494968-189c96cc-0a34-4e56-96c7-3a33ea439919.png" height="40">](https://www.mpi-inf.mpg.de)  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; [<img src="https://user-images.githubusercontent.com/14181188/160495259-f60ee657-3d04-40a8-abad-d8a9c42dd8fc.png" height="40">](https://www.kuleuven.be)
