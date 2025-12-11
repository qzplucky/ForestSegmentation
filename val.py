# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import numpy as np
from tqdm import tqdm

import paddle

from paddleseg.cvlibs import manager, Config, SegBuilder
from paddleseg.core import evaluate
from paddleseg.utils import get_sys_env, logger, utils


def parse_args():
    parser = argparse.ArgumentParser(description='Model evaluation')

    # Common params
    parser.add_argument("--config", help="The path of config file.", type=str)
    parser.add_argument(
        '--model_path',
        help='The path of trained weights to be loaded for evaluation.',
        type=str)
    parser.add_argument(
        '--num_workers',
        help='Number of workers for data loader. Bigger num_workers can speed up data processing.',
        type=int,
        default=0)
    parser.add_argument(
        '--device',
        help='Set the device place for evaluating model.',
        default='gpu',
        choices=['cpu', 'gpu', 'xpu', 'npu', 'mlu'],
        type=str)

    # Data augment params
    parser.add_argument(
        '--aug_eval',
        help='Whether to use mulit-scales and flip augment for evaluation.',
        action='store_true')
    parser.add_argument(
        '--scales',
        nargs='+',
        help='Scales for data augment.',
        type=float,
        default=1.0)
    parser.add_argument(
        '--flip_horizontal',
        help='Whether to use flip horizontally augment.',
        action='store_true')
    parser.add_argument(
        '--flip_vertical',
        help='Whether to use flip vertically augment.',
        action='store_true')

    # Sliding window evaluation params
    parser.add_argument(
        '--is_slide',
        help='Whether to evaluate images in sliding window method.',
        action='store_true')
    parser.add_argument(
        '--crop_size',
        nargs=2,
        help='The crop size of sliding window, the first is width and the second is height.'
        'For example, `--crop_size 512 512`',
        type=int)
    parser.add_argument(
        '--stride',
        nargs=2,
        help='The stride of sliding window, the first is width and the second is height.'
        'For example, `--stride 512 512`',
        type=int)

    # Other params
    parser.add_argument(
        '--data_format',
        help='Data format that specifies the layout of input. It can be "NCHW" or "NHWC". Default: "NCHW".',
        type=str,
        default='NCHW')
    parser.add_argument(
        '--auc_roc',
        help='Whether to use auc_roc metric.',
        type=bool,
        default=False)
    parser.add_argument(
        '--opts',
        help='Update the key-value pairs of all options.',
        default=None,
        nargs='+')
    # Set multi-label mode
    parser.add_argument(
        '--use_multilabel',
        action='store_true',
        default=False,
        help='Whether to enable multilabel mode. Default: False.')

    return parser.parse_args()


def merge_test_config(cfg, args):
    test_config = cfg.test_config
    if args.aug_eval:
        test_config['aug_eval'] = args.aug_eval
        test_config['scales'] = args.scales
        test_config['flip_horizontal'] = args.flip_horizontal
        test_config['flip_vertical'] = args.flip_vertical
    if args.is_slide:
        test_config['is_slide'] = args.is_slide
        test_config['crop_size'] = args.crop_size
        test_config['stride'] = args.stride
    if args.use_multilabel:
        test_config['use_multilabel'] = args.use_multilabel
    return test_config


def compute_iou_per_class(pred, label, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        label_inds = (label == cls)
        intersection = np.logical_and(pred_inds, label_inds).sum()
        union = np.logical_or(pred_inds, label_inds).sum()
        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union
        ious.append(iou)
    return ious


# ---------- helpers: sample extraction & shape unify ----------
def _extract_img_label(sample):
    """
    支持 dataset[idx] 返回 dict (常见: {'img':..., 'label':...})
    或 (img, label) / [img, label]
    """
    if isinstance(sample, dict):
        img = sample.get('img', sample.get('image', sample.get('img_data')))
        lab = sample.get('label', sample.get('mask', sample.get('gt')))
        if img is None or lab is None:
            raise KeyError(f"Sample keys not found. Available keys: {list(sample.keys())}")
        return img, lab
    elif isinstance(sample, (list, tuple)):
        if len(sample) < 2:
            raise ValueError(f"Sample length {len(sample)} < 2")
        return sample[0], sample[1]
    else:
        raise TypeError(f"Unsupported sample type: {type(sample)}")


def _ensure_hw(arr):
    """
    把标签/预测统一为 (H, W)，去掉多余的 1 维；若仍是 3D，则尽量 squeeze。
    """
    a = np.asarray(arr)
    if a.ndim == 3:
        # 支持 (1,H,W) 或 (H,W,1)
        if a.shape[0] == 1:
            a = a[0]
        elif a.shape[-1] == 1:
            a = a[..., 0]
        else:
            a = np.squeeze(a)
    return a


# -------- 修复后的逐图 IoU（维度对齐 & 最近邻对齐） --------
def evaluate_single_sample_iou(model, dataset, num_classes=2, save_path=None):
    model.eval()
    results = []

    f = None
    if save_path:
        f = open(save_path, 'w', encoding='utf-8')

    for idx in tqdm(range(len(dataset)), desc="Evaluating per-image IoU"):
        sample = dataset[idx]
        image, label = _extract_img_label(sample)

        # to tensor & add batch dim
        if isinstance(image, paddle.Tensor):
            image_t = image.unsqueeze(0)
        else:
            image_t = paddle.to_tensor(np.asarray(image)[np.newaxis, ...], dtype='float32')

        with paddle.no_grad():
            pred = model(image_t)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            pred = paddle.argmax(pred, axis=1).squeeze(0).numpy().astype(np.uint8)

        lab_np = _ensure_hw(label).astype(np.uint8)
        pred_hw = _ensure_hw(pred).astype(np.uint8)

        # 尺寸不一致时，用最近邻把 pred 对齐到 label
        if pred_hw.shape != lab_np.shape:
            import cv2
            pred_hw = cv2.resize(pred_hw, (lab_np.shape[1], lab_np.shape[0]), interpolation=cv2.INTER_NEAREST)

        ious = compute_iou_per_class(pred_hw, lab_np, num_classes)
        miou = np.nanmean(ious)
        results.append((idx, ious, miou))

        if f:
            f.write(f"Image {idx}: IOUs = {['%.4f' % i for i in ious]}, mIoU = {miou:.4f}\n")

    if f:
        f.close()
    return results


# -------- 汇总 precision / recall / F1 / IoU(0/1) / mIoU（安全混淆矩阵） --------
def evaluate_set_metrics(model, dataset, num_classes=2,
                         save_per_image_iou=None, save_summary=None):
    """
    - 逐张前向，累计混淆矩阵（向量化 & 维度健壮）
    - 输出 per-class Precision / Recall / F1 / IoU，以及 mIoU
    - 可选保存逐图 IoU（save_per_image_iou）与汇总（save_summary）
    """
    import cv2

    def fast_hist_safe(pred_hw, lab_hw, ncls):
        # 保证 (H,W) 形状并展平
        p = _ensure_hw(pred_hw).reshape(-1).astype(np.int64)
        l = _ensure_hw(lab_hw).reshape(-1).astype(np.int64)
        m = (l >= 0) & (l < ncls)
        return np.bincount(ncls * l[m] + p[m], minlength=ncls**2).reshape(ncls, ncls)

    model.eval()
    hist = np.zeros((num_classes, num_classes), dtype=np.int64)

    # per-image iou lines
    f_img = open(save_per_image_iou, "w", encoding="utf-8") if save_per_image_iou else None

    for idx in tqdm(range(len(dataset)), desc="Eval (accumulating metrics)"):
        sample = dataset[idx]
        image, label = _extract_img_label(sample)

        if isinstance(image, paddle.Tensor):
            img_t = image.unsqueeze(0)
        else:
            img_t = paddle.to_tensor(np.asarray(image)[np.newaxis, ...], dtype='float32')

        with paddle.no_grad():
            pred = model(img_t)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            pred = paddle.argmax(pred, axis=1).squeeze(0).numpy().astype(np.uint8)

        lab_np = _ensure_hw(label).astype(np.uint8)
        pred_hw = _ensure_hw(pred).astype(np.uint8)

        if pred_hw.shape != lab_np.shape:
            pred_hw = cv2.resize(pred_hw, (lab_np.shape[1], lab_np.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 累计混淆矩阵
        hist += fast_hist_safe(pred_hw, lab_np, num_classes)

        # 逐图 IoU（基于该图）
        h1 = fast_hist_safe(pred_hw, lab_np, num_classes)
        inter1 = np.diag(h1)
        union1 = h1.sum(1) + h1.sum(0) - inter1
        iou1 = np.where(union1 > 0, inter1 / union1, np.nan)
        miou1 = np.nanmean(iou1)
        if f_img:
            f_img.write(f"Image {idx}: IoU(non-forest=0)={iou1[0]:.4f}, IoU(forest=1)={iou1[1]:.4f}, mIoU={miou1:.4f}\n")

    if f_img:
        f_img.close()

    # ---- 汇总指标（基于整体混淆矩阵）----
    TP = np.diag(hist).astype(np.float64)
    FP = hist.sum(0) - TP
    FN = hist.sum(1) - TP
    # TN = hist.sum() - (TP + FP + FN)

    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.where(TP + FP > 0, TP / (TP + FP), np.nan)
        recall    = np.where(TP + FN > 0, TP / (TP + FN), np.nan)
        f1        = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), np.nan)
        iou       = np.where(TP + FP + FN > 0, TP / (TP + FP + FN), np.nan)

    miou = np.nanmean(iou)

    # 打印你关心的指标（class 0=非林地, class 1=林地）
    print("\n===== Metrics (2-class: 0=非林地, 1=林地) =====")
    print(f"Precision: [non-forest, forest] = [{precision[0]:.4f}, {precision[1]:.4f}]")
    print(f"Recall   : [non-forest, forest] = [{recall[0]:.4f}, {recall[1]:.4f}]")
    print(f"F1-score : [non-forest, forest] = [{f1[0]:.4f}, {f1[1]:.4f}]")
    print(f"IoU(non-forest)= {iou[0]:.4f}  IoU(forest)= {iou[1]:.4f}")
    print(f"mIoU = {miou:.4f}")

    if save_summary:
        with open(save_summary, "w", encoding="utf-8") as f:
            f.write("Summary (2-class: 0=non-forest, 1=forest)\n")
            f.write(f"Precision: {precision.tolist()}\n")
            f.write(f"Recall   : {recall.tolist()}\n")
            f.write(f"F1-score : {f1.tolist()}\n")
            f.write(f"IoU      : {iou.tolist()}\n")
            f.write(f"mIoU     : {miou:.6f}\n")

    return {
        "precision": precision, "recall": recall, "f1": f1, "iou": iou, "miou": miou,
        "hist": hist
    }


def main(args):
    assert args.config is not None, \
        'No configuration file specified, please set --config'
    cfg = Config(args.config, opts=args.opts)
    builder = SegBuilder(cfg)
    test_config = merge_test_config(cfg, args)

    # 如需跳过环境打印，可改为 try/except
    utils.show_env_info()
    utils.show_cfg_info(cfg)
    utils.set_device(args.device)

    # Only support for the DeepLabv3+ model with NHWC
    if args.data_format == 'NHWC':
        if cfg.dic['model']['type'] != 'DeepLabV3P':
            raise ValueError(
                'The "NHWC" data format only support the DeepLabV3P model!')
        cfg.dic['model']['data_format'] = args.data_format
        cfg.dic['model']['backbone']['data_format'] = args.data_format
        loss_len = len(cfg.dic['loss']['types'])
        for i in range(loss_len):
            cfg.dic['loss']['types'][i]['data_format'] = args.data_format

    model = builder.model
    if args.model_path:
        utils.load_entire_model(model, args.model_path)
        logger.info('Loaded trained weights successfully.')
    val_dataset = builder.val_dataset

    # 先跑 PaddleSeg 自带 evaluate（打印它的指标）
    evaluate(model, val_dataset, num_workers=args.num_workers, **test_config)

    # 再输出自定义指标 + 逐图 IoU
    per_image_iou_file = "per_image_iou.txt"
    summary_file = "metrics_pick15.txt"
    evaluate_set_metrics(model, val_dataset, num_classes=2,
                         save_per_image_iou=per_image_iou_file,
                         save_summary=summary_file)
    print(f"Per-image IoU saved to {per_image_iou_file}")
    print(f"Summary metrics saved to {summary_file}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
