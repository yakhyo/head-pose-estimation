# Head Pose Estimation: Lightweight Head Pose Estimation using MobileNet and ResNets

![Downloads](https://img.shields.io/github/downloads/yakhyo/head-pose-estimation/total) 
[![GitHub Repo stars](https://img.shields.io/github/stars/yakhyo/head-pose-estimation)](https://github.com/yakhyo/head-pose-estimation/stargazers)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/yakhyo/head-pose-estimation)

<video controls autoplay loop src="https://github.com/user-attachments/assets/307262d3-8fa0-4084-be6c-29ee1a3903ef" muted="false" width="100%"></video>
<video controls autoplay loop src="https://github.com/user-attachments/assets/50f010cf-6fcf-46b0-87cc-53065cba3fe7" muted="false" width="100%"></video>
Video by Yan Krukau: https://www.pexels.com/video/male-teacher-with-his-students-8617126/

This project focuses on head pose estimation using various deep learning models, including ResNet (18, 34, 50) and MobileNet v2. It builds upon [6DRepNet](https://github.com/thohemp/6DRepNet) by incorporating additional pre-trained models and refined code to enhance performance and flexibility.

## Features

- [x] **Pre-trained Backbones**:
  - [x] **ResNet**: Implemented using [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) for robust feature extraction and improved accuracy.
  - [x] **MobileNet v2**: Leveraging [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) for efficient and lightweight model performance.
  - [x] **MobileNet v3**: Integration of [MobileNetV3: An Efficient Network Architecture for Mobile Vision](https://arxiv.org/abs/1905.02244) for optimized performance in resource-constrained environments.
- [x] **Head Pose Estimation**: Enhanced model architecture for precise head pose estimation using the aforementioned backbones.
- [x] **Face Detection Integration**: Utilizes [Sample and Computation Redistribution for Efficient Face Detection](https://arxiv.org/abs/2105.04714) to enable efficient inference and processing.

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

   | Model              | Weights                                                                                                              | Size    | Epochs | Yaw  | Pitch | Roll | MAE  |
   | ------------------ | -------------------------------------------------------------------------------------------------------------------- | ------- | ------ | ---- | ----- | ---- | ---- |
   | ResNet-18          | [resnet18.pt](https://github.com/yakhyo/head-pose-estimation/releases/download/v0.0.1/resnet18.pt)                   | 43 MB   | 100    | 4.48 | 5.75  | 4.06 | 4.76 |
   | ResNet-34          | [resnet34.pt](https://github.com/yakhyo/head-pose-estimation/releases/download/v0.0.1/resnet34.pt)                   | 81.6 MB | 100    | 4.61 | 5.46  | 3.89 | 4.65 |
   | ResNet-50          | [resnet50.pt](https://github.com/yakhyo/head-pose-estimation/releases/download/v0.0.1/resnet50.pt)                   | 91.3 MB | 100    | 3.62 | 5.31  | 3.85 | 4.26 |
   | MobileNet V2       | [mobilenetv2.pt](https://github.com/yakhyo/head-pose-estimation/releases/download/v0.0.1/mobilenetv2.pt)             | 9.59 MB | 100    | 5.46 | 6.05  | 4.43 | 5.31 |
   | MobileNet V3 small | [mobilenetv3_small.pt](https://github.com/yakhyo/head-pose-estimation/releases/download/v0.0.1/mobilenetv3_small.pt) | 6 MB    | 50     | 8.65 | 7.66  | 6.03 | 7.45 |
   | MobileNet V3 large | [mobilenetv3_large.pt](https://github.com/yakhyo/head-pose-estimation/releases/download/v0.0.1/mobilenetv3_large.pt) | 17 MB   | 100    | 5.85 | 6.83  | 5.08 | 5.92 |

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

```bash
python main.py --data data/300W_LP --dataset 300W --arch resnet18
                                                         resnet34
                                                         resnet50
                                                         mobilenetv2
                                                         mobilenetv3_small
                                                         mobilenetv3_large
```

`main.py` arguments:

```
usage: main.py [-h] [--data DATA] [--dataset DATASET] [--num-epochs NUM_EPOCHS] [--batch-size BATCH_SIZE] [--arch ARCH] [--lr LR] [--num-workers NUM_WORKERS] [--checkpoint CHECKPOINT]
               [--scheduler {StepLR,MultiStepLR}] [--step-size STEP_SIZE] [--gamma GAMMA] [--milestones MILESTONES [MILESTONES ...]] [--output OUTPUT]

Head pose estimation training.

options:
  -h, --help            show this help message and exit
  --data DATA           Directory path for data.
  --dataset DATASET     Dataset name.
  --num-epochs NUM_EPOCHS
                        Maximum number of training epochs.
  --batch-size BATCH_SIZE
                        Batch size.
  --arch ARCH           Network architecture, currently available: resnet18/34/50, mobilenetv2
  --lr LR               Base learning rate.
  --num-workers NUM_WORKERS
                        Number of workers for data loading.
  --checkpoint CHECKPOINT
                        Path to checkpoint to continue training.
  --scheduler {StepLR,MultiStepLR}
                        Learning rate scheduler type.
  --step-size STEP_SIZE
                        Period of learning rate decay for StepLR.
  --gamma GAMMA         Multiplicative factor of learning rate decay for StepLR and ExponentialLR.
  --milestones MILESTONES [MILESTONES ...]
                        List of epoch indices to reduce learning rate for MultiStepLR (ignored if StepLR is used).
  --output OUTPUT       Path of model output.
```

### Evaluation

```bash
python evaluate.py --data data/AFLW200 --dataset AFLW200 --weights weights/resnet18.pt     --arch resnet18
                                                                           resnet34.pt            resnet34
                                                                           resnet50.pt            resnet50
                                                                           mobilenetv2.pt         mobilenetv2
                                                                           mobilenetv3_small.pt   mobilenetv3_small
                                                                           mobilenetv3_large.pt   mobilenetv3_large
```

`evaluate.py` arguments:

```
usage: evaluate.py [-h] [--data DATA] [--dataset DATASET] [--arch ARCH] [--num-workers NUM_WORKERS] [--batch-size BATCH_SIZE] [--weights WEIGHTS]

Head pose estimation evaluation.

options:
  -h, --help            show this help message and exit
  --data DATA           Directory path for data.
  --dataset DATASET     Dataset type.
  --arch ARCH           Network architecture, currently available: resnet18/34/50, mobilenetv2
  --num-workers NUM_WORKERS
                        Number of workers for data loading.
  --batch-size BATCH_SIZE
                        Batch size.
  --weights WEIGHTS     Path to model weight for evaluation.
```

### Inference

```bash
detect.py --input assets/in_video.mp4 --weights weights/resnet18.pt --arch resnet18 --output output.mp4
```

`detect.py` arguments:

```
usage: detect.py [-h] [--arch ARCH] [--cam CAM] [--view] [--draw-type {cube,axis}] --weights WEIGHTS [--output OUTPUT]

Head pose estimation inference.

options:
  -h, --help            show this help message and exit
  --arch ARCH           Model name, default `resnet18`
  --cam CAM             Camera device id to use [0]
  --view                Display the inference results
  --draw-type {cube,axis}
                        Draw cube or axis for head pose
  --weights WEIGHTS     Path to head pose estimation model weights
  --output OUTPUT       Path to save output file
```

## Reference

1. https://github.com/thohemp/6DRepNet
2. https://github.com/yakhyo/face-reidentification (used for inference, modified from [insightface](https://github.com/deepinsight/insightface))
