# *Location and Path Planning of Dorsal Acupoints for a Robotic Massage System*

> **Brief:** This repository contains code for acupoint localization and path planning for a robotic massage system.
![pic](pic/pipeline.svg)
---

## ✨ **Features**
- An adaptive brightness mapping data augmentation strategy for the training phase
![pic](pic/RBM.svg)

- A state-space network equipped with a flexible multiplicative gating mechanism
![pic](pic/LiteViM.svg)

- A structural alignment loss to constrain the relative geometric relationships among keypoints
![pic](pic/SAL.svg)

- Quality-Aware Adaptive Local Reconstruction for ACO-Based Acupoint Path Planning


---

## 📁 Project Structure
```text
.
├─ dataset/                 # Dataset
├─ pic/                      
├─ examples/                 
├─ tests/
├─ path/                    # Core code for path  planning
├─ ultralytics/             # Core code for acupoints detection
└─ README.md
```

## 🚀 **Quick Start**
### (1) Clone 
```bash
git clone https://github.com/huansu/Location-and-Path-Planning-of-Dorsal-Acupoints-for-a-Robotic-Massage-System.git

cd Location-and-Path-Planning-of-Dorsal-Acupoints-for-a-Robotic-Massage-System
```
### (2) Prepare Data
Put your dataset under dataset/. Example structure:
```text
dataset/
├─ images/
│  ├─ train/
│  └─ val/
└─ labels/
   ├─ train/
   └─ val/
```

### (3) Train or Inference

This project follows the Ultralytics YOLO pose pipeline (task=`pose`).\
(We sincerely thank the Ultralytics team and contributors for their outstanding open-source work and well-maintained ecosystem.) \
Please prepare the dataset YAML in Ultralytics pose format before training.


## 📊 **Results**

### (1) Ablation Experiments 【Visual Algorithm】
**Table 1. Results of the ablation experiments.**

| Methods  | +RBM | +LiteViM | +SAL | mAP<sub>0.5</sub><sup>OKS</sup> (%) | mAP<sub>0.5:0.95</sub><sup>OKS</sup> (%) |
|---------|:----:|:--------:|:----:|------------------------------------:|-----------------------------------------:|
| Baseline | × | × | × | 93.1 | 60.1 |
|          | √ | × | × | 93.7 | 61.4 |
|          | × | √ | × | 95.2 | 62.0 |
|          | × | × | √ | 94.1 | 62.2 |
|          | × | √ | √ | 95.1 | 60.9 |
|          | √ | × | √ | 94.7 | 63.3 |
|          | √ | √ | × | **95.5** | 62.6 |
|          | √ | √ | √ | 95.2 | **63.5** |


### (2) Comparisons with State-of-the-Art Methods【Visual Algorithm】
**Table 2. Results of multi-model comparison experiments.**

| Models         | mAP<sub>0.5</sub><sup>OKS</sup> (%) | mAP<sub>0.5:0.95</sub><sup>OKS</sup> (%) | Params | FLOPs   |
|----------------|------------------------------------:|-----------------------------------------:|-------:|--------:|
| YOLOv8L-pose   | 91.1 | 58.7 | 45.3M | 172.2G |
| YOLOv11L-pose  | 92.6 | 59.6 | 26.9M | 94.0G  |
| YOLOX-Pose     | 82.1 | 44.2 | 10.8M | 2.208G |
| HRNet          | 90.4 | 50.7 | 29.6M | 8.55G  |
| SCNet          | 87.5 | 44.6 | 34.1M | 5.517G |
| RTMPose        | 88.3 | 45.1 | 6.0M  | 0.72G  |
| **Ours**       | **95.2** | **63.5** | **21.6M** | **81.4G** |


### (3) Comparisons with State-of-the-Art Methods【Path Planning Algorithm】
**Table 3. Lengths of paths planned by different algorithms.**

| Map size      | Greedy | 2-opt  | 3-opt  | ACO    | Tabu search | **Ours** |
|--------------|-------:|------:|------:|------:|-----------:|--------:|
| 591×821      | 1456.70 | 1041.67 | 990.32 | 1104.02 | 1004.74 | **982.27** |
| 1279×1706    | 4379.79 | 2641.62 | 2547.02 | 2903.04 | 2789.27 | **2536.67** |
| 3072×4096    | 10912.57 | 9348.37 | 8413.03 | 8457.25 | 9397.98 | **8096.96** |
| 4000×6000    | 11647.12 | 9216.58 | 8309.28 | 8930.49 | 9159.41 | **7294.31** |


![pic](pic/PathPlanning.svg)

## 🙏 **Acknowledgements**

This work builds upon and is inspired by several open-source projects that provide essential foundations for pose estimation, detection frameworks, and efficient model design. The following repositories are acknowledged in no particular order.

- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)**
- **[MMPose](https://github.com/open-mmlab/mmpose)**
- **[VSSD](https://github.com/YuHengsss/VSSD)**
- **[DMDBAK](https://github.com/Ye-ChunZhe/DMDBAK)**

