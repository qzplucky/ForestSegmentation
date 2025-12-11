# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Constant
from paddleseg.models import layers
from paddleseg.models.layers import tensor_fusion_helper as helper


class UAFM(nn.Layer):
    """
    The base of Unified Attention Fusion Module.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__()

        self.conv_x = layers.ConvBNReLU(
            x_ch, y_ch, kernel_size=ksize, padding=ksize // 2, bias_attr=False)
        self.conv_out = layers.ConvBNReLU(
            y_ch, out_ch, kernel_size=3, padding=1, bias_attr=False)
        self.resize_mode = resize_mode

    def check(self, x, y):
        assert x.ndim == 4 and y.ndim == 4
        x_h, x_w = x.shape[2:]
        y_h, y_w = y.shape[2:]
        assert x_h >= y_h and x_w >= y_w

    def prepare(self, x, y):
        x = self.prepare_x(x, y)
        y = self.prepare_y(x, y)
        return x, y

    def prepare_x(self, x, y):
        x = self.conv_x(x)
        return x

    def prepare_y(self, x, y):
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)
        return y_up

    def fuse(self, x, y):
        out = x + y
        out = self.conv_out(out)
        return out

    def forward(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        self.check(x, y)
        x, y = self.prepare(x, y)
        out = self.fuse(x, y)
        return out


class UAFM_ChAtten(UAFM):
    """
    The UAFM with channel attention, which uses mean and max values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            layers.ConvBNAct(
                4 * y_ch,
                y_ch // 2,
                kernel_size=1,
                bias_attr=False,
                act_type="leakyrelu"),
            layers.ConvBN(
                y_ch // 2, y_ch, kernel_size=1, bias_attr=False))

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        atten = helper.avg_max_reduce_hw([x, y], self.training)
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class UAFM_ChAtten_S(UAFM):
    """
    The UAFM with channel attention, which uses mean values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            layers.ConvBNAct(
                2 * y_ch,
                y_ch // 2,
                kernel_size=1,
                bias_attr=False,
                act_type="leakyrelu"),
            layers.ConvBN(
                y_ch // 2, y_ch, kernel_size=1, bias_attr=False))

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        atten = helper.avg_reduce_hw([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class UAFM_SpAtten(UAFM):
    """
    The UAFM with spatial attention, which uses mean and max values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            layers.ConvBNReLU(
                4, 2, kernel_size=3, padding=1, bias_attr=False),
            layers.ConvBN(
                2, 1, kernel_size=3, padding=1, bias_attr=False))
        self._scale = self.create_parameter(
            shape=[1],
            attr=ParamAttr(initializer=Constant(value=1.)),
            dtype="float32")
        self._scale.stop_gradient = True

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        atten = helper.avg_max_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (self._scale - atten)
        out = self.conv_out(out)
        return out


class UAFM_SpAtten_S(UAFM):
    """
    The UAFM with spatial attention, which uses mean values.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_xy_atten = nn.Sequential(
            layers.ConvBNReLU(
                2, 2, kernel_size=3, padding=1, bias_attr=False),
            layers.ConvBN(
                2, 1, kernel_size=3, padding=1, bias_attr=False))

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        atten = helper.avg_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class UAFMMobile(UAFM):
    """
    Unified Attention Fusion Module for mobile.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_x = layers.SeparableConvBNReLU(
            x_ch, y_ch, kernel_size=ksize, padding=ksize // 2, bias_attr=False)
        self.conv_out = layers.SeparableConvBNReLU(
            y_ch, out_ch, kernel_size=3, padding=1, bias_attr=False)


class UAFMMobile_SpAtten(UAFM):
    """
    Unified Attention Fusion Module with spatial attention for mobile.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear'):
        super().__init__(x_ch, y_ch, out_ch, ksize, resize_mode)

        self.conv_x = layers.SeparableConvBNReLU(
            x_ch, y_ch, kernel_size=ksize, padding=ksize // 2, bias_attr=False)
        self.conv_out = layers.SeparableConvBNReLU(
            y_ch, out_ch, kernel_size=3, padding=1, bias_attr=False)

        self.conv_xy_atten = nn.Sequential(
            layers.ConvBNReLU(
                4, 2, kernel_size=3, padding=1, bias_attr=False),
            layers.ConvBN(
                2, 1, kernel_size=3, padding=1, bias_attr=False))

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        atten = helper.avg_max_reduce_channel([x, y])
        atten = F.sigmoid(self.conv_xy_atten(atten))

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


##########################################################################
##########################################################################
##########################################################################
##########################################################################
##########################################################################
def conv(in_channels, out_channels, kernel_size, bias_attr=False, stride=1):
    layer = nn.Conv2D(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias_attr=bias_attr, stride=stride)
    return layer
##########################################################################
## Spatial Attention
class SALayer(nn.Layer):
    def __init__(self, kernel_size=7):
        super(SALayer, self).__init__()
        self.conv1 = nn.Conv2D(2, 1, kernel_size, padding=kernel_size // 2, bias_attr=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = paddle.mean(x, axis=1, keepdim=True)

        max_out = paddle.max(x, axis=1, keepdim=True)
        # torch.cat   -> torch.concat
        y = paddle.concat([avg_out, max_out], axis=1)
        y = self.conv1(y)

        y = self.sigmoid(y)
        return x * y

# Spatial Attention Block (SAB)
class SAB(nn.Layer):
    def __init__(self, n_feat, kernel_size, reduction, bias_attr, act):
        super(SAB, self).__init__()
        modules_body = [conv(n_feat, n_feat, kernel_size, bias_attr=bias_attr), act, conv(n_feat, n_feat, kernel_size, bias_attr=bias_attr)]
        self.body = nn.Sequential(*modules_body)
        self.SA = SALayer(kernel_size=7)

    def forward(self, x):
        res = self.body(x)
        res = self.SA(res)
        res += x
        return res

##########################################################################
## Pixel Attention
##########################################################################

class PALayer(nn.Layer):
    def __init__(self, channel, reduction=16, bias_attr=False):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2D(channel, channel // reduction, 1, padding=0, bias_attr=bias_attr),
            nn.ReLU(name=True),
            nn.Conv2D(channel // reduction, channel, 1, padding=0, bias_attr=bias_attr), # channel <-> 1
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y

## Pixel Attention Block (PAB)

class PAB(nn.Layer):
    def __init__(self, n_feat, kernel_size, reduction, bias_attr, act):
        super(PAB, self).__init__()

        modules_body = [conv(n_feat, n_feat, kernel_size, bias_attr=bias_attr), act, conv(n_feat, n_feat, kernel_size, bias_attr=bias_attr)]
        self.PA = PALayer(n_feat, reduction, bias_attr=bias_attr)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.PA(res)
        res += x
        return res
    
##########################################################################
## Channel Attention Layer
##########################################################################

class CALayer(nn.Layer):
    def __init__(self, channel, reduction=16, bias_attr=False):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv_du = nn.Sequential(
            nn.Conv2D(channel, channel // reduction, 1, padding=0, bias_attr=bias_attr),
            nn.ReLU(name=True),
            nn.Conv2D(channel // reduction, channel, 1, padding=0, bias_attr=bias_attr),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Channel Attention Block (CAB)
class CAB(nn.Layer):
    def __init__(self, n_feat, kernel_size, reduction, bias_attr, act):
        super(CAB, self).__init__()
        modules_body = [conv(n_feat, n_feat, kernel_size, bias_attr=bias_attr), act, conv(n_feat, n_feat, kernel_size, bias_attr=bias_attr)]

        self.CA = CALayer(n_feat, reduction, bias_attr=bias_attr)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res


##########################################################################
# Mixed 
##########################################################################
class Mix(nn.Layer):
    def __init__(self, m=1):
        super(Mix, self).__init__()

        w = paddle.create_parameter(shape=[1], 
                                dtype='float32', 
                                default_initializer=paddle.nn.initializer.Assign(paddle.to_tensor([m], dtype='float32')))

        w = paddle.create_parameter(shape=w.shape, dtype=w.dtype, default_initializer=paddle.nn.initializer.Assign(w))

        
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2, feat3):
        factor = self.mix_block(self.w)
        other = (1 - factor)/2
        # we update it, now there is (1-w)/2 * fea1 + w * fea2 +(1-w)/2 * fea3
        output = fea1 * other.expand_as(fea1) + fea2 * factor.expand_as(fea2) + feat3 * other.expand_as(feat3)
        return output, factor

##########################################################################
# Mixed Residual Module
##########################################################################

class MixPre(nn.Layer):
    def __init__(self, m=1):
        super(MixPre, self).__init__()


        w = paddle.create_parameter(shape=[1], 
                                dtype='float32', 
                                default_initializer=paddle.nn.initializer.Assign(paddle.to_tensor([m], dtype='float32')))

        w = paddle.create_parameter(shape=w.shape, dtype=w.dtype, default_initializer=paddle.nn.initializer.Assign(w))

        
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        factor = self.mix_block(self.w)
        other = (1 - factor)/2
        output = fea1 * other.expand_as(fea1) + fea2 * factor.expand_as(fea2) 
        return output, factor

bn = 1  # block number-1
class UAFM_CMFAttenV01(UAFM):

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear', reduction=4, bias_attr=False):
        super(UAFM_CMFAttenV01,self).__init__(x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear')

        p_act = nn.PReLU()
        n_feat = y_ch
        self.CABatt = [CAB(n_feat, ksize, reduction,  act=p_act, bias_attr=bias_attr) for _ in range(bn)]
        self.PABatt = [PAB(n_feat, ksize, reduction, act=p_act, bias_attr=bias_attr) for _ in range(bn)]
        self.SABatt = [SAB(n_feat, ksize, reduction,  act=p_act, bias_attr=bias_attr) for _ in range(bn)]
        self.CAB_level = nn.Sequential(*self.CABatt)
        self.PAB_level = nn.Sequential(*self.PABatt)
        self.SAB_level = nn.Sequential(*self.SABatt)

        self.add123 = conv(n_feat, out_ch, ksize, bias_attr=bias_attr)
        self.MixPre = MixPre(1)
        self.Mix = Mix(1)

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        mix_o,_ = self.MixPre(x, y)
        
        CAB_xy = self.CAB_level(mix_o)
        PAB_xy = self.PAB_level(mix_o)
        SAB_xy = self.SAB_level(mix_o)

        mixed_img,_ = self.Mix(CAB_xy,PAB_xy,SAB_xy)
        out = self.add123(mixed_img)
        return out


class UAFM_CMFAttenV02(UAFM):

    def __init__(self, x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear', reduction=4, bias_attr=False):
        super(UAFM_CMFAttenV02,self).__init__(x_ch, y_ch, out_ch, ksize=3, resize_mode='bilinear')

        p_act = nn.PReLU()
        n_feat = y_ch
        self.CABatt = [CAB(n_feat, ksize, reduction,  act=p_act, bias_attr=bias_attr) for _ in range(bn)]
        self.PABatt = [PAB(n_feat, ksize, reduction, act=p_act, bias_attr=bias_attr) for _ in range(bn)]
        self.SABatt = [SAB(n_feat, ksize, reduction,  act=p_act, bias_attr=bias_attr) for _ in range(bn)]
        self.CAB_level = nn.Sequential(*self.CABatt)
        self.PAB_level = nn.Sequential(*self.PABatt)
        self.SAB_level = nn.Sequential(*self.SABatt)

        self.add123 = conv(n_feat, out_ch, 3, bias_attr=bias_attr)
        self.MixPre = MixPre(1)
        self.Mix = Mix(1)

    def fuse(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        mix_o,_ = self.MixPre(x, y)
        
        CAB_xy = self.CAB_level(mix_o)
        PAB_xy = self.PAB_level(mix_o)
        SAB_xy = self.SAB_level(mix_o)

        mixed_img,_ = self.Mix(CAB_xy,PAB_xy,SAB_xy)

        mixed_img += mix_o
        out = self.add123(mixed_img)
        return out

