# Copyright (c) OpenMMLab. All rights reserved.
import numpy
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
import torch.nn.functional as F
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from af import myrelu,MyReLU

@HEADS.register_module()
class EAFHead(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self,  **kwargs):
        super(EAFHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        # PSP Module


        self.tfa = TFA(self.in_channels[-1],self.channels)

        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = BasicConv1(in_channels,self.channels,kernel_size=3,stride=1,padding=1)
            fpn_conv =BasicConv1(self.channels,self.channels,kernel_size=3,stride=1,padding=1)


            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.fpn_bottleneck2 = ConvModule(
            self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.sig=nn.Sigmoid()






    def TFA_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        output = self.tfa(x)

        return output



    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.TFA_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs1 = torch.cat(fpn_outs, dim=1)

        fpn_outs2=sum(fpn_outs)
        fpn_outs2=fpn_outs2/4


        feats1 = self.fpn_bottleneck(fpn_outs1)

        feats2=self.fpn_bottleneck2(fpn_outs2)
        feats=feats1*feats2
        feats=self.sig(feats)

        return feats



    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)

        return output


class TFA(nn.ModuleList):
    def __init__(self,in_ch,out_ch):
        super(TFA, self).__init__()
        mid = in_ch


        self.pool_h=nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w=nn.AdaptiveMaxPool2d((1, None))

        self.conv3 =nn.Conv2d(1, mid, 1, 1, 0)
        self.conv4=nn.Conv2d(in_ch*3,mid,1, 1, 0)
        self.conv5=nn.Conv2d(in_ch*4,mid,1, 1, 0)

        self.convf1 = BasicConvsig(in_ch, mid, (1, 3), 1, (0, 1))
        self.convf2 = BasicConvsig(in_ch, mid, (3, 1), 1, (1, 0))
        self.convf3 = BasicConvsig(in_ch, mid, (1, 5), 1, (0, 2))
        self.convf4 = BasicConvsig(in_ch, mid, (5, 1), 1, (2, 0))

        self.daconv1=BasicConv(mid,mid,3,1,1,dilation=1)
        self.daconv2=BasicConv(mid,mid,3,1,1,dilation=2)
        self.daconv3 = BasicConv(mid, out_ch, 3, 1, 1, dilation=3)


    def forward(self,x):
        n, c, h, w = x.size()
        x1=self.pool_h(x)
        x2=self.pool_w(x)

        x11=x1.expand(-1, -1, h, w)
        x22=x2.expand(-1, -1, h, w)

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x3=self.conv3(max_out)

        #x4=torch.cat([x11,x22,x3],dim=1)
        x5=x11+x22+x3


        x_f1=self.convf1(x5)
        x_f2 = self.convf2(x5)
        x_f3 = self.convf3(x5)
        x_f4 = self.convf4(x5)

        x_f5=x_f1+x_f2+x_f3+x_f4


        x_da1=self.daconv1(x_f5)
        x_da2=self.daconv2(x_da1)
        x_da3=self.daconv3(x_da2)
        return x_da3





class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x



class BasicConv1(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,  bn=True, bias=False):
        super(BasicConv1, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None


    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = myrelu(x)
        return x

class BasicConvsig(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,  bn=True, bias=False):
        super(BasicConvsig, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.sig=nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.sig(x)
        return x


