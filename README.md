[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2212.03010)
[![GitHub Stars](https://img.shields.io/github/stars/Nightmare-n/GD-MAE?style=social)](https://github.com/Nightmare-n/GD-MAE)
![visitors](https://visitor-badge.glitch.me/badge?page_id=Nightmare-n/GD-MAE)

# GD-MAE: Generative Decoder for MAE Pre-training on LiDAR Point Clouds (CVPR 2023)

## NEWS
[2023-03-31] Code is released.

[2023-02-28] GD-MAE is accepted at CVPR 2023.

[2022-12-14] The result of [GD-MAE](https://waymo.com/open/challenges/entry/?challenge=DETECTION_3D&challengeId=DETECTION_3D&emailId=50be7d97-96bd&timestamp=1671074591082186) on the Waymo Leaderboard is reported.

## Installation
We test this project on NVIDIA A100 GPUs and Ubuntu 18.04.
```
conda create -n gd-mae python=3.7
conda activate gd-mae
conda install -y pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -y pytorch3d -c pytorch3d
pip install numpy==1.19.5 protobuf==3.19.4 scikit-image==0.19.2 waymo-open-dataset-tf-2-2-0 nuscenes-devkit==1.0.5 spconv-cu111 numba scipy pyyaml easydict fire tqdm shapely matplotlib opencv-python addict pyquaternion awscli open3d pandas future pybind11 tensorboardX tensorboard Cython
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu111.html
git clone https://github.com/Nightmare-n/GD-MAE
cd GD-MAE && python setup.py develop --user
```

## Data Preparation

Please follow the [instruction](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md) of OpenPCDet to prepare the dataset. For the Waymo dataset, we use the [evaluation toolkits](https://drive.google.com/drive/folders/1aa1kI9hhzBoZkIBcr8RBO3Zhg_RkOAag?usp=sharing) to evaluate detection results.
```
data
│── waymo
│   │── ImageSets/
│   │── raw_data
│   │   │── segment-xxxxxxxx.tfrecord
│   │   │── ...
│   │── waymo_processed_data
│   │   │── segment-xxxxxxxx/
│   │   │── ...
│   │── waymo_processed_data_gt_database_train_sampled_1/
│   │── waymo_processed_data_waymo_dbinfos_train_sampled_1.pkl
│   │── waymo_processed_data_infos_test.pkl
│   │── waymo_processed_data_infos_train.pkl
│   │── waymo_processed_data_infos_val.pkl
│   │── compute_detection_metrics_main
│   │── gt.bin
│── kitti
│   │── ImageSets/
│   │── training
│   │   │── label_2/
│   │   │── velodyne/
│   │   │── ...
│   │── testing
│   │   │── label_2/
│   │   │── velodyne/
│   │   │── ...
│   │── gt_database/
│   │── kitti_dbinfos_train.pkl
│   │── kitti_infos_test.pkl
│   │── kitti_infos_train.pkl
│   │── kitti_infos_val.pkl
│   │── kitti_infos_trainval.pkl
│── once
│   │── ImageSets/
│   │── data
│   │   │── 000000/
│   │   │── ...
│   │── gt_database/
│   │── once_dbinfos_train.pkl
│   │── once_infos_raw_large.pkl
│   │── once_infos_raw_medium.pkl
│   │── once_infos_raw_small.pkl
│   │── once_infos_train.pkl
│   │── once_infos_val.pkl
│── kitti-360
│   │── data_3d_raw
│   │   │── xxxxxxxx_sync/
│   │   │── ...
│── ckpts
│   │── graph_rcnn_po.pth
│   │── ...
```

## Training & Testing
```
# mae pretrain & finetune
bash scripts/dist_ssl_train.sh

# one-stage model & two-stage model (separately)
bash scripts/dist_ts_train.sh

# one-stage model | two-stage model (end-to-end)
bash scripts/dist_train.sh

# test
bash scripts/dist_test.sh
```

## Results

### Waymo
|                                             | Vec_L1 | Vec_L2 | Ped_L1 | Ped_L2 | Cyc_L1 | Cyc_L2 | Model |
|---------------------------------------------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|[Graph RCNN (w/o PointNet)](tools/cfgs/waymo_models/graph_rcnn_ce.yaml)| 80.57/80.09|72.30/71.85|82.86/77.28|75.02/69.70|77.16/76.01|74.38/73.28| [log](https://drive.google.com/file/d/1paPQ_c5ayGrbrVxGD9YtgBAy7TCMp2Gm/view?usp=sharing) |
|[GD-MAE_0.2 (20% labeled data)](tools/cfgs/waymo_models/gd_mae.yaml)| 76.24/75.74|67.67/67.22|80.50/72.29|73.17/65.50|72.61/71.40|69.86/68.69| [log](https://drive.google.com/file/d/1TGoxSAJi6o6seA6XxASAajZrJ5jjqOJR/view?usp=sharing) |
|[GD-MAE_iou (iou head)](tools/cfgs/waymo_models/gd_mae_iou.yaml)| 79.40/78.94|70.91/70.49|82.20/75.85|74.82/68.79|75.75/74.77|72.98/72.03| [log](https://drive.google.com/file/d/1-6tfzhdDIpv5UaOQrdE4LotA_1AE7ZAF/view?usp=sharing) |
|[GD-MAE_ts (two-stage)](tools/cfgs/waymo_models/gd_mae_ts.yaml)| 80.21/79.77|72.37/71.96|83.10/76.72|75.54/69.43|77.19/76.16|74.41/73.40| [log](https://drive.google.com/file/d/1fOFQGmJcJK3qep44D1qca9Jk95mgfAdg/view?usp=sharing) |

### KITTI
|                                             | Easy | Moderate | Hard | Model |
|---------------------------------------------|:-------:|:-------:|:-------:|:-------:|
|[Graph-Vo](tools/cfgs/kitti_models/graph_rcnn_vo.yaml)| 93.29 | 86.08 | 83.15 | [ckpt](https://drive.google.com/file/d/1DQtzf14LzYVGPJUkiolI2qd4mVvfsKgs/view?usp=sharing) |
|[Graph-VoI](tools/cfgs/kitti_models/graph_rcnn_voi.yaml)| 95.80 | 86.72 | 83.93 | [ckpt](https://drive.google.com/file/d/1RLVdzcAhbHrH7H3aBEYRQQS0fLh-K2d5/view?usp=sharing) |
|[Graph-Po](tools/cfgs/kitti_models/graph_rcnn_po.yaml)| 93.44 | 86.54 | 83.90 | [ckpt](https://drive.google.com/file/d/12mNhuNB-X2GQDxL-sDnqnRBl1FD_H-l7/view?usp=sharing) |

|                                             | Car | Pedestrian | Cyclist | Model |
|---------------------------------------------|:-------:|:-------:|:-------:|:-------:|
|[GD-MAE](tools/cfgs/kitti_models/gd_mae.yaml)| 82.01 | 48.40 | 67.16 | [pretrain](https://drive.google.com/file/d/1dlS-x4qgWP1erL5khOeUlGJqQ2HYHasl/view?usp=sharing)/[ckpt](https://drive.google.com/file/d/10m8kUUybkjMLnJK5O31-ZRxHuPBYYiLh/view?usp=sharing) |

### ONCE
|                                             | Vehicle | Pedestrian | Cyclist | Model |
|---------------------------------------------|:-------:|:-------:|:-------:|:-------:|
|[CenterPoint-Pillar](tools/cfgs/once_models/centerpoint_pillar.yaml)| 74.10 | 40.94 | 62.17 | [ckpt](https://drive.google.com/file/d/12D24zjXvWOAC38EQJSoRWpZ0_AuTHLyi/view?usp=sharing) |
|[GD-MAE](tools/cfgs/once_models/gd_mae.yaml)| 76.79 | 48.84 | 69.14 | [pretrain](https://drive.google.com/file/d/1Qdhu4pOPCt288Opry-B84O-uAEh-G5sr/view?usp=sharing)/[ckpt](https://drive.google.com/file/d/1CALOwvXcxQEit2-EAE7j1h-ucdpvKvRF/view?usp=sharing) |

## Citation 
If you find this project useful in your research, please consider citing:
```
@inproceedings{yang2023gdmae,
    author = {Honghui Yang and Tong He and Jiaheng Liu and Hua Chen and Boxi Wu and Binbin Lin and Xiaofei He and Wanli Ouyang},
    title = {GD-MAE: Generative Decoder for MAE Pre-training on LiDAR Point Clouds},
    booktitle = {CVPR},
    year = {2023},
}
```
```
@inproceedings{yang2022graphrcnn,
    author = {Honghui Yang and Zili Liu and Xiaopei Wu and Wenxiao Wang and Wei Qian and Xiaofei He and Deng Cai},
    title = {Graph R-CNN: Towards Accurate 3D Object Detection with Semantic-Decorated Local Graph},
    booktitle = {ECCV},
    year = {2022},
}
```

## Acknowledgement
This project is mainly based on the following codebases. Thanks for their great works!

* [GraphRCNN](https://github.com/Nightmare-n/GraphRCNN)
* [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
* [CenterPoint](https://github.com/tianweiy/CenterPoint)
* [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
* [SST](https://github.com/tusen-ai/SST)
* [ONCE_Benchmark](https://github.com/PointsCoder/Once_Benchmark)
* [SASA](https://github.com/blakechen97/SASA)
