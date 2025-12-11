# Enhancing Cross-Regional Generalization in UAV Forest Segmentation with Attention-Refined PP-LiteSeg Networks

English | [简体中文](README_CN.md)

<div align="center">

<p align="center">
  <img src="./docs/images/paddleseg_logo.png" align="middle" width = "500" />
</p>

**An Enhanced PP-LiteSeg Implementation for UAV Forest Segmentation with Cross-Regional Generalization**

[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
![python version](https://img.shields.io/badge/python-3.8+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)

</div>

## Introduction

This repository extends the **PP-LiteSeg** architecture with attention refinement mechanisms for robust **UAV-based forest segmentation** across diverse geographical regions. Our work focuses on improving cross-regional generalization capabilities to handle varying forest structures, lighting conditions, and environmental factors.

### Key Features
- **Two Enhanced Variants**: PP-LiteSeg-V01 (Multi-Branch Attention Fusion) and PP-LiteSeg-V02 (Residual-Enhanced Attention)
- **Cross-Regional Adaptation**: Optimized for plantation forests (regular structure) and natural forests (high heterogeneity)
- **Lightweight Design**: Maintains PP-LiteSeg's efficiency while enhancing feature representation
- **Comprehensive Evaluation**: Systematic testing on multiple forest types in Yunnan Province, China

### Research Contributions
1. **Novel Attention Mechanisms**: Multi-Branch Attention Fusion Module (MAFM) and its residual-enhanced variant
2. **Cross-Regional Dataset**: Constructed UAV forest datasets from structurally different regions
3. **Scene-Specific Optimization**: V01 excels in plantation forests; V02 performs better in natural forests

## Model Variants

### 1. PP-LiteSeg-V01
- **Architecture**: Standard PP-LiteSeg with Multi-Branch Attention Fusion Module
- **Strengths**: Superior boundary precision in regular plantation forests
- **Application**: Forest farms with uniform structure and clear boundaries

### 2. PP-LiteSeg-V02  
- **Architecture**: PP-LiteSeg with Residual-Enhanced Attention Fusion
- **Strengths**: Better robustness in heterogeneous natural forests
- **Application**: Natural forests with complex textures and illumination variations

## Quick Start

### Installation
```bash
# Install PaddlePaddle
pip install paddlepaddle-gpu==2.5.0

# Install dependencies
pip install -r requirements.txt

## <img src="https://user-images.githubusercontent.com/48054808/157801371-9a9a8c65-1690-4123-985a-e0559a7f9494.png" width="20"/> Industrial Segmentation Model

</details>


<details>
<summary><b>Super Lightweight Semantic Segmentation Models</b></summary>

#### These super lightweight semantic segmentation models are designed for X86 CPU and ARM CPU.

| Model | Backbone | ADE20K mIoU(%) | Snapdragon 855 Inference latency(ms) | params(M) | Links |
|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
|TopFormer-Base|TopTransformer-Base|38.28|480.6|5.13|[config](./configs/topformer/)
|PP-MobileSeg-Base :star2: |StrideFormer-Base|41.57|265.5|5.62|[config](./configs/pp_mobileseg/)|
|TopFormer-Tiny|TopTransformer-Tiny|32.46|490.3|1.41|[config](./configs/topformer/)
|PP-MobileSeg-Tiny :star2: |StrideFormer-Tiny|36.39|215.3|1.61|[config](./configs/pp_mobileseg/)|

Note that:
* We test the inference speed on Snapdragon 855. We use [PaddleLite](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/README_en.md) CPP API with 1 thread, and the shape of input tensor is 1x3x512x512. We test the latency with the final argmax operator on.


| Model    | Backbone | Cityscapes mIoU(%)    |  V100 TRT Inference Speed(FPS)  | Snapdragon 855 Inference Speed(FPS) | Config File |
|:-------- |:--------:|:---------------------:|:-------------------------------:|:-----------------------------------:|:-----------:|
| MobileSeg      | MobileNetV2              | 73.94                 | 67.57          | 27.01   | [yml](./configs/mobileseg/)  |
| MobileSeg :star2:  | MobileNetV3              | 73.47                 | 67.39          | 32.90   | [yml](./configs/mobileseg/)  |
| MobileSeg      | Lite_HRNet_18            | 70.75                 | *10.5*         | 13.05   | [yml](./configs/mobileseg/)  |
| MobileSeg      | ShuffleNetV2_x1_0        | 69.46                 | *37.09*        | 39.61  | [yml](./configs/mobileseg/)  |
| MobileSeg      | GhostNet_x1_0            | 71.88                 | *35.58*        | 38.74  | [yml](./configs/mobileseg/)  |

Note that:
* We test the inference speed on Nvidia GPU V100. We use PaddleInference Python API with TensorRT enabled. The data type is FP32, and the shape of input tensor is 1x3x1024x2048.
* We test the inference speed on Snapdragon 855. We use PaddleLite CPP API with 1 thread, and the shape of input tensor is 1x3x256x256.

</details>


## <img src="./docs/images/teach.png" width="20"/> Tutorials

**Introductory Tutorials**

* [Installation](./docs/install.md)
* [Quick Start](./docs/quick_start.md)
* [A 20 minutes Blitz to Learn PaddleSeg](./docs/whole_process.md)
* [Model Zoo](./docs/model_zoo_overview.md)

**Basic Tutorials**

* Data Preparation
    * [Prepare Public Dataset](./docs/data/pre_data.md)
    * [Prepare Customized Dataset](./docs/data/marker/marker.md)
    * [Label Data with EISeg](./EISeg)
* [Config Preparation](./docs/config/pre_config.md)
* [Model Training](/docs/train/train.md)
* [Model Evaluation](./docs/evaluation/evaluate.md)
* [Model Prediction](./docs/predict/predict.md)

* Model Export
    * [Export Inference Model](./docs/model_export.md)
    * [Export ONNX Model](./docs/model_export_onnx.md)

* Model Deployment
    * [FastDeploy](./deploy/fastdeploy)
    * [Paddle Inference (Python)](./docs/deployment/inference/python_inference.md)
    * [Paddle Inference (C++)](./docs/deployment/inference/cpp_inference.md)
    * [Paddle Lite](./docs/deployment/lite/lite.md)
    * [Paddle Serving](./docs/deployment/serving/serving.md)
    * [Paddle JS](./docs/deployment/web/web.md)
    * [Benchmark](./docs/deployment/inference/infer_benchmark.md)

**Advanced Tutorials**

* [Training Tricks](./docs/train/train_tricks.md)

*  Model Compression
    * [Quantization](./docs/deployment/slim/quant/quant.md)
    * [Distillation](./docs/deployment/slim/distill/distill.md)
    * [Pruning](./docs/deployment/slim/prune/prune.md)
    * [Auto Compression](./docs/deployment/slim/act/readme.md)

* [FAQ](./docs/faq/faq/faq.md)

**Welcome to Contribute**

* [API Documention](./docs/apis)

*  Advanced Development
    * [Detailed Configuration File](./docs/design/use/use.md)
    * [Create Your Own Model](./docs/design/create/add_new_model.md)
*  Pull Request
    * [PR Tutorial](./docs/pr/pr/pr.md)
    * [PR Style](./docs/pr/pr/style.md)

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

## License

PaddleSeg is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement
* Thanks [jm12138](https://github.com/jm12138) for contributing U<sup>2</sup>-Net.
* Thanks [zjhellofss](https://github.com/zjhellofss) (Fu Shenshen) for contributing Attention U-Net, and Dice Loss.
* Thanks [liuguoyu666](https://github.com/liguoyu666), [geoyee](https://github.com/geoyee) for contributing U-Net++ and U-Net3+.
* Thanks [yazheng0307](https://github.com/yazheng0307) (LIU Zheng) for contributing quick-start document.
* Thanks [CuberrChen](https://github.com/CuberrChen) for contributing STDC(rethink BiSeNet), PointRend and DetailAggregateLoss.
* Thanks [stuartchen1949](https://github.com/stuartchen1949) for contributing SegNet.
* Thanks [justld](https://github.com/justld) (Lang Du) for contributing UPerNet, DDRNet, CCNet, ESPNetV2, DMNet, ENCNet, HRNet_W48_Contrast, FastFCN, BiSeNetV1, SECrossEntropyLoss and PixelContrastCrossEntropyLoss.
* Thanks [Herman-Hu-saber](https://github.com/Herman-Hu-saber) (Hu Huiming) for contributing ESPNetV2.
* Thanks [zhangjin12138](https://github.com/zhangjin12138) for contributing RandomCenterCrop.
* Thanks [simuler](https://github.com/simuler) for contributing ESPNetV1.
* Thanks [ETTR123](https://github.com/ETTR123)(Zhang Kai) for contributing ENet, PFPNNet.


