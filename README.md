AR-PPLiteSeg: Attention-Refined PP-LiteSeg for UAV Forest Segmentation

This project is an improvement based on the PP-LiteSeg network, aiming to enhance the cross-regional generalization capability of UAV-based forest segmentation. By introducing an attention-refined module, it enhances the model's adaptability to different geographic environments, seasonal conditions, and image quality.

🛠️ Environment Dependencies
Base Environment

Python >= 3.8

PaddlePaddle >= 2.4.0

Installation

# Install PaddlePaddle (GPU version)
python -m pip install paddlepaddle-gpu==2.4.0.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# Install project dependencies
pip install -r requirements.txt


📦 Usage Instructions

Prepare the Data
Organize the data as follows:

data/
├── train/
│   ├── images/  # Training images
│   └── labels/  # Training labels
├── val/         # Validation set
└── test/        # Test set


Model Training

# Single-GPU training
python train.py --config configs/yourdata.yaml

# Multi-GPU training
python -m paddle.distributed.launch train.py \
    --config configs/ar_ppliteseg.yaml \
    --num_workers 4


Model Evaluation

python eval.py \
    --config configs/yourdata.yaml \
    --model_path outputs/best_model.pdparams


Model Inference

python predict.py \
    --config configs/yourdata.yaml \
    --model_path outputs/best_model.pdparams \
    --image_path your_image \
    --save_path result


Model Export

python export_model.py \
    --config configs/yourdata.yaml \
    --model_path outputs/best_model.pdparams \
    --save_dir inference_model/


⚙️ Configuration File
The main configuration file is located in configs/ar_ppliteseg.yaml and can be modified for:

Data path

Training hyperparameters

Model parameters

Loss function settings

📁 Project Structure

.
├── configs/           # Configuration files
├── models/            # Model definitions
├── utils/             # Utility functions
├── train.py           # Training script
├── eval.py            # Evaluation script
├── predict.py         # Inference script
└── requirements.txt   # Dependency list

## <img src="./docs/images/anli.png" width="20"/> Special Features
  * [Interactive Segmentation](./EISeg)
  * [Image Matting](./Matting)
  * [PP-HumanSeg](./contrib/PP-HumanSeg)
  * [3D Medical Segmentation](./contrib/MedicalSeg)
  * [Cityscapes SOTA](./contrib/CityscapesSOTA)
  * [Panoptic Segmentation](./contrib/PanopticDeepLab)
  * [CVPR Champion Solution](./contrib/AutoNUE)
  * [Domain Adaptation](./contrib/DomainAdaptation)

## <img src="https://user-images.githubusercontent.com/48054808/157801371-9a9a8c65-1690-4123-985a-e0559a7f9494.png" width="20"/> Industrial Tutorial Examples

* [Using PP-HumanSegV2 for Human Segmentation](https://aistudio.baidu.com/aistudio/projectdetail/4504982?contributionType=1)
* [Using PP-HumanSegV1 for Human Segmentation](https://aistudio.baidu.com/aistudio/projectdetail/2189481?channelType=0&channel=0)
* [Using PP-LiteSeg for Road Segmentation](https://aistudio.baidu.com/aistudio/projectdetail/3873145?contributionType=1)
* [Using PaddleSeg for Face Parsing and Makeup](https://aistudio.baidu.com/aistudio/projectdetail/5326422)
* [Using PaddleSeg for Mini-dataset Spine Segmentation](https://aistudio.baidu.com/aistudio/projectdetail/3878920)
* [Using PaddleSeg for Lane Segmentation](https://aistudio.baidu.com/aistudio/projectdetail/1752986?channelType=0&channel=0)
* [PaddleSeg in APIs](https://aistudio.baidu.com/aistudio/projectdetail/1339458?channelType=0&channel=0)
* [Learn Paddleseg in 10 Mins](https://aistudio.baidu.com/aistudio/projectdetail/1672610?channelType=0&channel=0)
* [Application of Interactive Segmentation Technology in Smart Mapping](https://aistudio.baidu.com/aistudio/projectdetail/5089472)
* [Nail art preview machine based on PaddleSeg](https://aistudio.baidu.com/aistudio/projectdetail/5156312)
* [Overrun monitoring of steel bar length based on PaddleSeg](https://aistudio.baidu.com/aistudio/projectdetail/5633532)

For more examples, see [here](https://aistudio.baidu.com/aistudio/projectdetail/5436669).

📄 License
This project is licensed under the Apache 2.0 License.
