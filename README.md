English | [简体中文](README_CN.md)

<div align="center">


**Enhancing Cross-Regional Generalization in UAV Forest Segmentation with Attention-Refined PP-LiteSeg Networks **



[PaddlePaddle](https://github.com/paddlepaddle/paddle).**

[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
[![Version](https://img.shields.io/github/release/PaddlePaddle/PaddleSeg.svg)](https://github.com/PaddlePaddle/PaddleSeg/releases)
![python version](https://img.shields.io/badge/python-3.6+-orange.svg)
![support os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-yellow.svg)
![stars](https://img.shields.io/github/stars/PaddlePaddle/PaddleSeg?color=ccf)

</div>


<div align="center">
<img src="https://github.com/shiyutang/files/blob/9590ea6bfc36139982ce75b00d3b9f26713934dd/teasor.gif"  width = "800" />  
</div>

## <img src="./docs/images/seg_news_icon.png" width="20"/> News
<ul class="nobull">
  <li>[2023-10-29] :fire: PaddleSeg v2.9 is released! Check more details in <a href="https://github.com/PaddlePaddle/PaddleSeg/releases">Release Notes</a>.</li>
    <ul>
        <li>Support <a href="./configs/multilabelseg">Multi-label segmentation</a>, it procides multi-label segmentation support on a serie of semantic segmetation models.</li>
        <li>Release <a href="./configs/SegmentAnything">Mobile SAM</a>, faster version of Segment Anything Model. </li>
        <li>Support <a href="./deploy/slim/act">Quant Aware Distillation Training Compression</a>  for PP-LiteSeg, PP-MobileSeg, OCRNet, and SegFormer-B0 to improve model inference speed. </li>
    </ul>
  <li>[2022-04-11] PaddleSeg v2.8 released <a href="./contrib/SegmentAnything">Segment Anything Model</a>, an original light-weight semantic segmentation model on mobile devices <a href="./configs/pp_mobileseg">PP-MobileSeg</a>,  <a href="./contrib/QualityInspector">QualityInspector v0.5</a>, a full-process solution for industrial quality inspection, and <a href="./contrib/PanopticSeg">PanopticSeg v0.5</a>, a universal panoptic segmentation solution.
  <li>[2022-11-30] PaddleSeg v2.7 released a real-time human matting model <a href="./Matting/">PP-MattingV2</a>, a 3D medical image segmentation solution <a href="./contrib/MedicalSeg/">MedicalSegV2</a>, and a real-time semantic segmentation model <a href="./configs/rtformer/">RTFormer</a>.
  <li>[2022-07-20] PaddleSeg v2.6 released a real-time human segmentation SOTA solution <a href="./contrib/PP-HumanSeg">PP-HumanSegV2</a>, a stable-version semi-automatic segmentation annotation tool <a href="./EISeg">EISeg v1.0</a>, a pseudo label pre-training method PSSL, and the source code of PP-MattingV1. </li>
  <li>[2022-04-20] PaddleSeg v2.5 released a real-time semantic segmentation model <a href="./configs/pp_liteseg">PP-LiteSeg</a>, a trimap-free image matting model PP-MattingV1, and an easy-to-use solution for 3D medical image segmentation MedicalSegV1.</li>
  <li>[2022-01-20] We release PaddleSeg v2.4 with EISeg v0.4, and PP-HumanSegV1 including an open-sourced dataset <a href="./contrib/PP-HumanSeg/paper.md#pp-humanseg14k-a-large-scale-teleconferencing-video-dataset">PP-HumanSeg14K</a>. </li>

</ul>


## <img src="https://user-images.githubusercontent.com/48054808/157795569-9fc77c85-732f-4870-9be0-99a7fe2cff27.png" width="20"/> Introduction

PaddleSeg is an end-to-end high-efficent development toolkit for image segmentation based on PaddlePaddle, which helps both developers and researchers in the whole process of designing segmentation models, training models, optimizing performance and inference speed, and deploying models. A lot of well-trained models and various real-world applications in both industry and academia help users conveniently build hands-on experiences in image segmentation.

<div align="center">
<img src="https://github.com/shiyutang/files/raw/main/teasor_new.gif"  width = "800" />  
</div>


## <img src="./docs/images/feature.png" width="20"/> Features

* **High-Performance Model**: Following the state of the art segmentation methods and using high-performance backbone networks, we provide 45+ models and 150+ high-quality pre-training models, which are better than other open-source implementations.

* **High Efficiency**: PaddleSeg provides multi-process asynchronous I/O, multi-card parallel training, evaluation, and other acceleration strategies, combined with the memory optimization function of the PaddlePaddle, which can greatly reduce the training overhead of the segmentation model, all these allowing developers to train image segmentation models more efficiently and at a lower cost.

* **Modular Design**: We build PaddleSeg with the modular design philosophy. Therefore, based on actual application scenarios, developers can assemble diversified training configurations with *data augmentation strategies*, *segmentation models*, *backbone networks*, *loss functions*, and other different components to meet different performance and accuracy requirements.

* **Complete Flow**: PaddleSeg supports image labeling, model designing, model training, model compression, and model deployment. With the help of PaddleSeg, developers can easily finish all tasks in the entire workflow.

<div align="center">
<img src="https://user-images.githubusercontent.com/14087480/176402154-390e5815-1a87-41be-9374-9139c632eb66.png" width = "800" />  
</div>

## <img src="./docs/images/chat.png" width="20"/> Community

* If you have any questions, suggestions or feature requests, please do not hesitate to create an issue in [GitHub Issues](https://github.com/PaddlePaddle/PaddleSeg/issues).
* Please scan the following QR code to join PaddleSeg WeChat group to communicate with us:
<div align="center">
<img src="https://paddleseg.bj.bcebos.com/images/seg_qr_code.png"  width = "200" />  
</div>


## <img src="./docs/images/model.png" width="20"/> Overview

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>Models</b>
      </td>
      <td colspan="2">
        <b>Components</b>
      </td>
      <td>
        <b>Special Cases</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
        <details><summary><b>Semantic Segmentation</b></summary>
          <ul>
            <li><a href="./configs/pp_liteseg">PP-LiteSeg</a> </li>
            <li><a href="./configs/pp_mobileseg">PP-MobileSeg</a> </li>
            <li><a href="./configs/deeplabv3p">DeepLabV3P</a> </li>
            <li><a href="./configs/ocrnet">OCRNet</a> </li>
            <li><a href="./configs/mobileseg">MobileSeg</a> </li>
            <li><a href="./configs/ann">ANN</a></li>
            <li><a href="./configs/attention_unet">Att U-Net</a></li>
            <li><a href="./configs/bisenetv1">BiSeNetV1</a></li>
            <li><a href="./configs/bisenet">BiSeNetV2</a></li>
            <li><a href="./configs/ccnet">CCNet</a></li>
            <li><a href="./configs/danet">DANet</a></li>
            <li><a href="./configs/ddrnet">DDRNet</a></li>
            <li><a href="./configs/decoupled_segnet">DecoupledSeg</a></li>
            <li><a href="./configs/deeplabv3">DeepLabV3</a></li>
            <li><a href="./configs/dmnet">DMNet</a></li>
            <li><a href="./configs/dnlnet">DNLNet</a></li>
            <li><a href="./configs/emanet">EMANet</a></li>
            <li><a href="./configs/encnet">ENCNet</a></li>
            <li><a href="./configs/enet">ENet</a></li>
            <li><a href="./configs/espnetv1">ESPNetV1</a></li>
            <li><a href="./configs/espnet">ESPNetV2</a></li>
            <li><a href="./configs/fastfcn">FastFCN</a></li>
            <li><a href="./configs/fastscnn">Fast-SCNN</a></li>
            <li><a href="./configs/gcnet">GCNet</a></li>
            <li><a href="./configs/ginet">GINet</a></li>
            <li><a href="./configs/glore">GloRe</a></li>
            <li><a href="./configs/gscnn">GSCNN</a></li>
            <li><a href="./configs/hardnet">HarDNet</a></li>
            <li><a href="./configs/fcn">HRNet-FCN</a></li>
            <li><a href="./configs/hrnet_w48_contrast">HRNet-Contrast</a></li>
            <li><a href="./configs/isanet">ISANet</a></li>
            <li><a href="./configs/pfpn">PFPNNet</a></li>
            <li><a href="./configs/pointrend">PointRend</a></li>
            <li><a href="./configs/portraitnet">PotraitNet</a></li>
            <li><a href="./configs/pp_humanseg_lite">PP-HumanSeg-Lite</a></li>
            <li><a href="./configs/pspnet">PSPNet</a></li>
            <li><a href="./configs/pssl">PSSL</a></li>
            <li><a href="./configs/segformer">SegFormer</a></li>
            <li><a href="./configs/segmenter">SegMenter</a></li>
            <li><a href="./configs/segmne">SegNet</a></li>
            <li><a href="./configs/setr">SETR</a></li>
            <li><a href="./configs/sfnet">SFNet</a></li>
            <li><a href="./configs/stdcseg">STDCSeg</a></li>
            <li><a href="./configs/u2net">U<sup>2</sup>Net</a></li>
            <li><a href="./configs/unet">UNet</a></li>
            <li><a href="./configs/unet_plusplus">UNet++</a></li>
            <li><a href="./configs/unet_3plus">UNet3+</a></li>
            <li><a href="./configs/upernet">UperNet</a></li>
            <li><a href="./configs/rtformer">RTFormer</a></li>
            <li><a href="./configs/uhrnet">UHRNet</a></li>
            <li><a href="./configs/topformer">TopFormer</a></li>
            <li><a href="./configs/mscale_ocrnet">MscaleOCRNet-PSA</a></li>
            <li><a href="./configs/cae">CAE</a></li>
            <li><a href="./configs/maskformer">MaskFormer</a></li>
            <li><a href="./configs/vit_adapter">ViT-Adapter</a></li>
            <li><a href="./configs/hrformer">HRFormer</a></li>
            <li><a href="./configs/lpsnet">LPSNet</a></li>
            <li><a href="./configs/segnext">SegNeXt</a></li>
            <li><a href="./configs/knet">K-Net</a></li>
          </ul>
        </details>
        <details><summary><b>Interactive Segmentation</b></summary>
          <ul>
            <li><a href="./EISeg">EISeg</a></li>
            <li>RITM</li>
            <li>EdgeFlow</li>
          </ul>
        </details>
        <details><summary><b>Image Matting</b></summary>
          <ul>
              <li><a href="./Matting/configs/ppmattingv2">PP-MattingV2</a></li>
              <li><a href="./Matting/configs/ppmatting">PP-MattingV1</a></li>
              <li><a href="./Matting/configs/dim/dim-vgg16.yml">DIM</a></li>
              <li><a href="./Matting/configs/modnet/modnet-hrnet_w18.yml">MODNet</a></li>
              <li><a href="./Matting/configs/human_matting/human_matting-resnet34_vd.yml">PP-HumanMatting</a></li>
              <li><a href="./Matting/configs/rvm">RVM</a></li>
          </ul>
        </details>
        <details><summary><b>Panoptic Segmentation</b></summary>
          <ul>
            <li><a href="./contrib/PanopticSeg/configs/mask2former">Mask2Former</a></li>
            <li><a href="./contrib/PanopticSeg/configs/panoptic_deeplab">Panoptic-DeepLab</a></li>
          </ul>
        </details>
      </td>
      <td>
        <details><summary><b>Backbones</b></summary>
          <ul>
            <li><a href="./paddleseg/models/backbones/hrnet.py">HRNet</a></li>
            <li><a href="./paddleseg/models/backbones/resnet_cd.py">ResNet</a></li>
            <li><a href="./paddleseg/models/backbones/stdcnet.py">STDCNet</a></li>
            <li><a href="./paddleseg/models/backbones/mobilenetv2.py">MobileNetV2</a></li>
            <li><a href="./paddleseg/models/backbones/mobilenetv3.py">MobileNetV3</a></li>
            <li><a href="./paddleseg/models/backbones/shufflenetv2.py">ShuffleNetV2</a></li>
            <li><a href="./paddleseg/models/backbones/ghostnet.py">GhostNet</a></li>
            <li><a href="./paddleseg/models/backbones/lite_hrnet.py">LiteHRNet</a></li>
            <li><a href="./paddleseg/models/backbones/xception_deeplab.py">XCeption</a></li>
            <li><a href="./paddleseg/models/backbones/vision_transformer.py">VIT</a></li>
            <li><a href="./paddleseg/models/backbones/mix_transformer.py">MixVIT</a></li>
            <li><a href="./paddleseg/models/backbones/swin_transformer.py">Swin Transformer</a></li>
            <li><a href="./paddleseg/models/backbones/top_transformer.py">TopTransformer</a></li>
            <li><a href="./paddleseg/models/backbones/hrformer.py">HRTransformer</a></li>
            <li><a href="./paddleseg/models/backbones/mscan.py">MSCAN</a></li>
          </ul>
        </details>
        <details><summary><b>Losses</b></summary>
          <ul>
            <li><a href="./paddleseg/models/losses/binary_cross_entropy_loss.py">Binary CE Loss</a></li>
            <li><a href="./paddleseg/models/losses/bootstrapped_cross_entropy_loss.py">Bootstrapped CE Loss</a></li>
            <li><a href="./paddleseg/models/losses/cross_entropy_loss.py">Cross Entropy Loss</a></li>
            <li><a href="./paddleseg/models/losses/decoupledsegnet_relax_boundary_loss.py">Relax Boundary Loss</a></li>
            <li><a href="./paddleseg/models/losses/detail_aggregate_loss.py">Detail Aggregate Loss</a></li>
            <li><a href="./paddleseg/models/losses/dice_loss.py">Dice Loss</a></li>
            <li><a href="./paddleseg/models/losses/edge_attention_loss.py">Edge Attention Loss</a></li>
            <li><a href="./paddleseg/models/losses/focal_loss.py">Focal Loss</a></li>
            <li><a href="./paddleseg/models/losses/binary_cross_entropy_loss.py">MultiClassFocal Loss</a></li>
            <li><a href="./paddleseg/models/losses/gscnn_dual_task_loss.py">GSCNN Dual Task Loss</a></li>
            <li><a href="./paddleseg/models/losses/kl_loss.py">KL Loss</a></li>
            <li><a href="./paddleseg/models/losses/l1_loss.py">L1 Loss</a></li>
            <li><a href="./paddleseg/models/losses/lovasz_loss.py">Lovasz Loss</a></li>
            <li><a href="./paddleseg/models/losses/mean_square_error_loss.py">MSE Loss</a></li>
            <li><a href="./paddleseg/models/losses/ohem_cross_entropy_loss.py">OHEM CE Loss</a></li>
            <li><a href="./paddleseg/models/losses/pixel_contrast_cross_entropy_loss.py">Pixel Contrast CE Loss</a></li>
            <li><a href="./paddleseg/models/losses/point_cross_entropy_loss.py">Point CE Loss</a></li>
            <li><a href="./paddleseg/models/losses/rmi_loss.py">RMI Loss</a></li>
            <li><a href="./paddleseg/models/losses/semantic_connectivity_loss.py">Connectivity Loss</a></li>
          </ul>
        </details>
        <details><summary><b>Metrics</b></summary>
          <ul>
            <li>mIoU</li>
            <li>Accuracy</li>
            <li>Kappa</li>
            <li>Dice</li>
            <li>AUC_ROC</li>
          </ul>  
        </details>
      </td>
      <td>
        <details><summary><b>Datasets</b></summary>
          <ul>
            <li><a href="./paddleseg/datasets/ade.py">ADE20K</a></li>  
            <li><a href="./paddleseg/datasets/cityscapes.py">Cityscapes</a></li>
            <li><a href="./paddleseg/datasets/cocostuff.py">COCO Stuff</a></li>
            <li><a href="./paddleseg/datasets/voc.py">Pascal VOC</a></li>
            <li><a href="./paddleseg/datasets/eg1800.py">EG1800</a></li>
            <li><a href="./paddleseg/datasets/pascal_context.py">Pascal Context</a></li>  
            <li><a href="./paddleseg/datasets/supervisely.py">SUPERVISELY</a></li>
            <li><a href="./paddleseg/datasets/optic_disc_seg.py">OPTIC DISC SEG</a></li>
            <li><a href="./paddleseg/datasets/chase_db1.py">CHASE_DB1</a></li>
            <li><a href="./paddleseg/datasets/hrf.py">HRF</a></li>
            <li><a href="./paddleseg/datasets/drive.py">DRIVE</a></li>
            <li><a href="./paddleseg/datasets/stare.py">STARE</a></li>
            <li><a href="./paddleseg/datasets/pp_humanseg14k.py">PP-HumanSeg14K</a></li>
            <li><a href="./paddleseg/datasets/pssl.py">PSSL</a></li>
          </ul>
        </details>
        <details><summary><b>Data Augmentation</b></summary>
          <ul>
            <li>Flipping</li>  
            <li>Resize</li>  
            <li>ResizeByLong</li>
            <li>ResizeByShort</li>
            <li>LimitLong</li>  
            <li>ResizeRangeScaling</li>  
            <li>ResizeStepScaling</li>
            <li>Normalize</li>
            <li>Padding</li>
            <li>PaddingByAspectRatio</li>
            <li>RandomPaddingCrop</li>  
            <li>RandomCenterCrop</li>
            <li>ScalePadding</li>
            <li>RandomNoise</li>  
            <li>RandomBlur</li>  
            <li>RandomRotation</li>  
            <li>RandomScaleAspect</li>  
            <li>RandomDistort</li>  
            <li>RandomAffine</li>  
          </ul>
        </details>
      </td>
      <td>
        <details><summary><b>Segment Anything</b></summary>
          <ul>
              <li><a href="./contrib/SegmentAnything">SegmentAnything</a></li>
          </ul>
        </details>
        <details><summary><b>Model Selection Tool</b></summary>
          <ul>
              <li><a href="./configs/smrt">PaddleSMRT</a></li>
          </ul>
        </details>
        <details><summary><b>Human Segmentation</b></summary>
          <ul>
              <li><a href="./contrib/PP-HumanSeg/README_cn.md">PP-HumanSegV1</a></li>
              <li><a href="./contrib/PP-HumanSeg/README_cn.md">PP-HumanSegV2</a></li>
          </ul>
        </details>
        <details><summary><b>MedicalSeg</b></summary>
          <ul>
            <li><a href="./contrib/MedicalSeg/configs/lung_coronavirus">VNet</a></li>
            <li><a href="./contrib/MedicalSeg/configs/msd_brain_seg">UNETR</a></li>
            <li><a href="./contrib/MedicalSeg/configs/acdc">nnFormer</a></li>
            <li><a href="./contrib/MedicalSeg/configs/nnunet/msd_lung">nnUNet-D</a></li>
            <li><a href="./contrib/MedicalSeg/configs/synapse">TransUNet</a></li>
            <li><a href="./contrib/MedicalSeg/configs/synapse">SwinUNet</a></li>
          </ul>
        </details>
        <details><summary><b>Cityscapes SOTA Model</b></summary>
          <ul>
              <li><a href="./contrib/CityscapesSOTA">HMSA</a></li>
          </ul>
        </details>
        <details><summary><b>CVPR Champion Model</b></summary>
          <ul>
              <li><a href="./contrib/AutoNUE">MLA Transformer</a></li>
          </ul>
        </details>
        <details><summary><b>Domain Adaptation</b></summary>
          <ul>
              <li><a href="./contrib/DomainAdaptation">PixMatch</a></li>
          </ul>
        </details>
      </td>  
    </tr>
</td>
    </tr>
  </tbody>
</table>


## <img src="https://user-images.githubusercontent.com/48054808/157801371-9a9a8c65-1690-4123-985a-e0559a7f9494.png" width="20"/> Industrial Segmentation Models

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

```
