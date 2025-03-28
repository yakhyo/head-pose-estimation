# Head Pose Estimation: Lightweight Head Pose Estimation using MobileNet and ResNets

![Downloads](https://img.shields.io/github/downloads/yakhyo/head-pose-estimation/total)
[![GitHub Repo stars](https://img.shields.io/github/stars/yakhyo/head-pose-estimation)](https://github.com/yakhyo/head-pose-estimation/stargazers)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/yakhyo/head-pose-estimation)

<!--
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest updates.</h5>
-->

<video controls autoplay loop src="https://github.com/user-attachments/assets/307262d3-8fa0-4084-be6c-29ee1a3903ef" muted="false" width="100%"></video>
<video controls autoplay loop src="https://github.com/user-attachments/assets/50f010cf-6fcf-46b0-87cc-53065cba3fe7" muted="false" width="100%"></video>
Video by Yan Krukau: https://www.pexels.com/video/male-teacher-with-his-students-8617126/

This project focuses on head pose estimation using various deep learning models, including ResNet (18, 34, 50) and MobileNet v2. It builds upon [6DRepNet](https://github.com/thohemp/6DRepNet) by incorporating additional pre-trained models and refined code to enhance performance and flexibility.

## ✨ Features

| Date       | Feature Description                                                                                                                                                                   |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2024-12-16 | 🔄 **Updated Pre-trained Models**: New weights for existing **ResNet (18, 34, 50)** and **MobileNet (v2, v3)** backbones, offering improved accuracy and efficiency.                  |
| 2024-12-16 | 🚀 **Multi-GPU Training Support**: Enabled distributed training across multiple GPUs for faster model training and improved scalability.                                              |
| 2024-08-31 | 🧠 **Face Detection Integration**: [Sample and Computation Redistribution](https://arxiv.org/abs/2105.04714) employed for efficient inference and processing in face detection tasks. |
| 2024-08-31 | 🔄 **Pre-trained Models**: Support for **ResNet (18, 34, 50)** and **MobileNet (v2, v3)** backbones for feature extraction and optimized performance across diverse applications.     |
| 2024-08-31 | 🎯 **Head Pose Estimation**: Enhanced model architecture for precise head pose estimation, utilizing pre-trained backbones for robust and efficient performance.                      |

## Evaluation results on AFLW2000

| Model              | Size    | Yaw        | Pitch      | Roll       | MAE        |
| ------------------ | ------- | ---------- | ---------- | ---------- | ---------- |
| ResNet-18          | 43 MB   | 4.5027     | 5.8261     | 4.2188     | 4.8492     |
| ResNet-34          | 81.6 MB | 4.4538     | 5.2690     | 3.8855     | 4.5361     |
| ResNet-50          | 91.3 MB | 3.5529     | 4.9962     | 3.4986     | 4.0159     |
| MobileNet V2       | 9.59 MB | 5.6880     | 6.0391     | 4.4433     | 5.3901     |
| MobileNet V3 small | 6 MB    | 8.6926     | 7.7089     | 6.0035     | 7.4683     |
| MobileNet V3 large | 17 MB   | 5.6068     | 6.6022     | 4.9959     | 5.7350     |

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yakyo/head-pose-estimation.git
cd head-pose-estimation
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download weight files:

   a) Download weights from the following links (Trained on 300W-LP and evaluated on AFLW200 dataset):

| Model              | Weights                                                                                                              | Size    |
| ------------------ | -------------------------------------------------------------------------------------------------------------------- | ------- |
| ResNet-18          | [resnet18.pt](https://github.com/yakhyo/head-pose-estimation/releases/download/v0.0.1/resnet18.pt)                   | 42.7 MB |
| ResNet-34          | [resnet34.pt](https://github.com/yakhyo/head-pose-estimation/releases/download/v0.0.1/resnet34.pt)                   | 81.3 MB |
| ResNet-50          | [resnet50.pt](https://github.com/yakhyo/head-pose-estimation/releases/download/v0.0.1/resnet50.pt)                   | 90 MB   |
| MobileNet V2       | [mobilenetv2.pt](https://github.com/yakhyo/head-pose-estimation/releases/download/v0.0.1/mobilenetv2.pt)             | 8.74 MB |
| MobileNet V3 small | [mobilenetv3_small.pt](https://github.com/yakhyo/head-pose-estimation/releases/download/v0.0.1/mobilenetv3_small.pt) | 5.93 MB |
| MobileNet V3 large | [mobilenetv3_large.pt](https://github.com/yakhyo/head-pose-estimation/releases/download/v0.0.1/mobilenetv3_large.pt) | 16.2 MB |

b) Run the command below to download weights to the `weights` directory (Linux):

```bash
sh download.sh [model_name]
            resnet18
            resnet34
            resnet50
            mobilenetv2
            mobilenetv3_small
            mobilenetv3_large
```

## Usage

### Datasets

Dataset folder structure:

```

data/
├── 300W_LP/
└── AFLW2000/

```

**300W_LP**

- Link to download dataset: [google drive link](https://drive.google.com/file/d/0B7OEHD3T4eCkVGs0TkhUWFN6N1k/view?usp=sharing&resourcekey=0-WT5tO4TOCbNZY6r6z6WmOA)
- Homepage: http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm

**AFLW200**

- Link to download dataset: http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip
- Homepage: http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm

### Training
**Note**: DDP training is also supported. To use, add `torchrun --nproc_per_node=num_gpus`

```bash
python main.py --data data/300W_LP --network resnet18
                                             resnet34
                                             resnet50
                                             mobilenetv2
                                             mobilenetv3_small
                                             mobilenetv3_large
```

`main.py` arguments:

```
usage: main.py [-h] [--data DATA] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--network NETWORK] [--lr LR] [--num-workers NUM_WORKERS] [--checkpoint CHECKPOINT] [--lr-scheduler {StepLR,MultiStepLR}] [--step-size STEP_SIZE] [--gamma GAMMA] [--milestones MILESTONES [MILESTONES ...]] [--print-freq PRINT_FREQ] [--world-size WORLD_SIZE]
               [--local-rank LOCAL_RANK] [--save-path SAVE_PATH]

Head pose estimation training.

options:
  -h, --help            show this help message and exit
  --data DATA           Directory path for data.
  --epochs EPOCHS       Maximum number of training epochs.
  --batch-size BATCH_SIZE
                        Batch size.
  --network NETWORK     Network architecture, currently available: resnet18/34/50, mobilenetv2
  --lr LR               Base learning rate.
  --num-workers NUM_WORKERS
                        Number of workers for data loading.
  --checkpoint CHECKPOINT
                        Path to checkpoint to continue training.
  --lr-scheduler {StepLR,MultiStepLR}
                        Learning rate scheduler type.
  --step-size STEP_SIZE
                        Period of learning rate decay for StepLR.
  --gamma GAMMA         Multiplicative factor of learning rate decay for StepLR and ExponentialLR.
  --milestones MILESTONES [MILESTONES ...]
                        List of epoch indices to reduce learning rate for MultiStepLR (ignored if StepLR is used).
  --print-freq PRINT_FREQ
                        Frequency (in batches) for printing training progress. Default: 100.
  --world-size WORLD_SIZE
                        Number of distributed processes
  --local-rank LOCAL_RANK
                        Local rank for distributed training
  --save-path SAVE_PATH
                        Path to save model checkpoints. Default: `weights`.
```

### Evaluation

```bash
python evaluate.py --data data/AFLW200 --weights weights/resnet18.pt --network resnet18
                                                 resnet34.pt                   resnet34
                                                 resnet50.pt                   resnet50
                                                 mobilenetv2.pt                mobilenetv2
                                                 mobilenetv3_small.pt          mobilenetv3_small
                                                 mobilenetv3_large.pt          mobilenetv3_large
```

`evaluate.py` arguments:

```
usage: evaluate.py [-h] [--data DATA] [--network NETWORK] [--num-workers NUM_WORKERS] [--batch-size BATCH_SIZE] [--weights WEIGHTS]

Head pose estimation evaluation.

options:
  -h, --help            show this help message and exit
  --data DATA           Directory path for data.
  --network NETWORK     Network architecture, currently available: resnet18/34/50, mobilenetv2
  --num-workers NUM_WORKERS
                        Number of workers for data loading.
  --batch-size BATCH_SIZE
                        Batch size.
  --weights WEIGHTS     Path to model weight for evaluation.
```

### Inference

```bash
detect.py --input assets/in_video.mp4 --weights weights/resnet18.pt --network resnet18 --output output.mp4
```

`detect.py` arguments:

```
usage: detect.py [-h] [--network NETWORK] [--input INPUT] [--view] [--draw-type {cube,axis}] --weights WEIGHTS [--output OUTPUT]

Head pose estimation inference.

options:
  -h, --help            show this help message and exit
  --network NETWORK     Model name, default `resnet18`
  --input INPUT         Path to input video file or camera id
  --view                Display the inference results
  --draw-type {cube,axis}
                        Draw cube or axis for head pose
  --weights WEIGHTS     Path to head pose estimation model weights
  --output OUTPUT       Path to save output file
```

## Reference

1. https://github.com/thohemp/6DRepNet
2. https://github.com/yakhyo/face-reidentification (used for inference, modified from [insightface](https://github.com/deepinsight/insightface))
