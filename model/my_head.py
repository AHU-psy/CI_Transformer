# Copyright (c) OpenMMLab. All rights reserved.
import pywt
import pywt.data
from functools import partial
from mmcv.cnn import build_norm_layer
from mmengine.model import ModuleList
from mmengine.model.weight_init import (constant_init, trunc_normal_,
                                        trunc_normal_init)
from typing import List, Tuple
from mmseg.models.backbones.vit import TransformerEncoderLayer
from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead
from torch import Tensor
# RAMiT (Reciprocal Attention Mixing Transformer)
from mmseg.utils import OptConfigType, ConfigType, SampleList
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import torch
from torchvision.transforms import functional as TF
import math
from timm.models.fx_features import register_notrace_function
from timm.models.layers import trunc_normal_, to_2tuple
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from mmengine.model import BaseModule
from torch import Tensor
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.losses import accuracy
from mmseg.models.utils import resize
from mmseg.registry import MODELS
from mmseg.structures import SegDataSample


class SAW(nn.Module):
    def __init__(self, dim, relax_denom=0, classifier=None, work=False):
        super(SAW, self).__init__()
        self.work = True
        self.selected_classes = [0, 1]
        self.C = 2
        self.dim = dim
        self.i = torch.eye(self.C, self.C).cuda()
        self.reversal_i = torch.ones(self.C, self.C).triu(diagonal=1).cuda()
        self.classify = classifier
        self.num_off_diagonal = torch.sum(self.reversal_i)
        if relax_denom == 0:
            print("Note relax_denom == 0!")
            self.margin = 0
        else:
            self.margin = self.num_off_diagonal // relax_denom

    def get_mask_matrix(self):
        return self.i, self.reversal_i, self.margin, self.num_off_diagonal

    def get_covariance_matrix(self, x, eye=None):
        eps = 1e-5
        B, C, H, W = x.shape  # i-th feature size (B X C X H X W)
        HW = H * W
        if eye is None:
            eye = torch.eye(C).cuda()
        x = x.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
        x_cor = torch.bmm(x, x.transpose(1, 2)).div(HW - 1) + (eps * eye)  # C X C / HW

        return x_cor, B

    def instance_whitening_loss(self, x, eye, mask_matrix, margin, num_remove_cov):
        x_cor, B = self.get_covariance_matrix(x, eye=eye)
        x_cor_masked = x_cor * mask_matrix

        off_diag_sum = torch.sum(torch.abs(x_cor_masked), dim=(1, 2), keepdim=True) - margin  # B X 1 X 1
        loss = torch.clamp(torch.div(off_diag_sum, num_remove_cov), min=0)  # B X 1 X 1
        loss = torch.sum(loss) / B

        return loss

    def sort_with_idx(self, x, idx, weights):
        b, c, _, _ = x.size()
        after_sort = torch.zeros_like(x)
        weights = F.sigmoid(weights)
        for i in range(b):

            for k in range(int(c / self.C)):
                for j in range(self.C):
                    channel_id = idx[self.selected_classes[j]][k]
                    wgh = weights[self.selected_classes[j]][channel_id]
                    after_sort[i][self.C * k + j][:][:] = wgh * x[i][channel_id][:][:]

        return after_sort

    def forward(self, x):
        if self.work:
            # print('work')
            weights_keys = self.classify.state_dict().keys()
            # print(weights_keys) : odict_keys(['weight', 'bias'])
            selected_keys_classify = []

            for key in weights_keys:

                if "weight" in key:
                    selected_keys_classify.append(key)

            for key in selected_keys_classify:
                weights_t = self.classify.state_dict()[key]

            # print(weights_t.shape)[19, 256, 1, 1]
            classsifier_weights = abs(weights_t.squeeze())
            _, index = torch.sort(classsifier_weights, descending=True, dim=1)
            f_map_lst = []
            B, channel_num, H, W = x.shape
            x = self.sort_with_idx(x, index, classsifier_weights)

            for i in range(int(channel_num / self.C)):
                group = x[:, self.C * i:self.C * (i + 1), :, :]
                f_map_lst.append(group)

            eye, mask_matrix, margin, num_remove_cov = self.get_mask_matrix()
            SAW_loss = torch.FloatTensor([0]).cuda()

            for i in range(int(channel_num / self.C)):
                loss = self.instance_whitening_loss(f_map_lst[i], eye, mask_matrix, margin, num_remove_cov)
                SAW_loss = SAW_loss + loss
        else:
            # print('nowork')
            SAW_loss = torch.FloatTensor([0]).cuda()

        return SAW_loss


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter)
        self.iwt_filter = nn.Parameter(self.iwt_filter)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x




def mean_std(scale, target_mode):
    if scale == 2:
        mean = (0.4485, 0.4375, 0.4045)
        std = (0.2397, 0.2290, 0.2389)
    elif scale == 3:
        mean = (0.4485, 0.4375, 0.4045)
        std = (0.2373, 0.2265, 0.2367)
    elif scale == 4:
        mean = (0.4485, 0.4375, 0.4045)
        std = (0.2352, 0.2244, 0.2349)
    elif target_mode == 'light_dn':  # image normalization with statistics from HQ sets
        mean = (0.4775, 0.4515, 0.4047)
        std = (0.2442, 0.2367, 0.2457)
    elif target_mode == 'light_realdn':  # image normalization with statistics from HQ sets
        mean = (0.0000, 0.0000, 0.0000)
        std = (1.0000, 1.0000, 1.0000)
    elif target_mode == 'light_graydn':  # image normalization with statistics from HQ sets
        mean = (0.4539,)
        std = (0.2326,)
    elif target_mode == 'light_lle':
        mean = (0.1687, 0.1599, 0.1526)
        std = (0.1142, 0.1094, 0.1094)
    elif target_mode == 'light_dr':
        mean = (0.5110, 0.5105, 0.4877)
        std = (0.2313, 0.2317, 0.2397)
    return mean, std


def denormalize(img, mean, std):
    assert isinstance(mean, tuple), 'mean and std should be tuple'
    assert isinstance(std, tuple), 'mean and std should be tuple'

    mean = torch.tensor(mean)
    std = torch.tensor(std)
    if img.ndim == 4:
        mean, std = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).type_as(img), std.unsqueeze(0).unsqueeze(
            -1).unsqueeze(-1).type_as(img)
    elif img.ndim == 3:
        mean, std = mean.unsqueeze(-1).unsqueeze(-1).type_as(img), std.unsqueeze(-1).unsqueeze(-1).type_as(img)
    return img * std + mean


class QKVProjection(nn.Module):
    def __init__(self, dim, num_head, qkv_bias=True):
        super(QKVProjection, self).__init__()
        self.dim = dim
        self.num_head = num_head

        self.qkv = nn.Conv2d(dim, 3 * dim, 1, bias=qkv_bias)

    def forward(self, x):
        B, C, H, W = x.size()
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b (l c) h w -> b l c h w', l=self.num_head)
        return qkv

    def flops(self, resolutions):
        return resolutions[0] * resolutions[1] * 1 * 1 * self.dim * 3 * self.dim


def get_relative_position_index(win_h, win_w):
    # get pair-wise relative position index for each token inside the window
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None,
                                                   :]  # 2, Wh*Ww, Wh*Ww (xaxis matrix & yaxis matrix)
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += win_h - 1  # shift to start from 0
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)  # Wh*Ww, Wh*Ww


class SpatialSelfAttention(nn.Module):
    def __init__(self, dim, num_head, total_head, window_size=8, shift=0, attn_drop=0.0, proj_drop=0.0, helper=True):
        super(SpatialSelfAttention, self).__init__()
        self.dim = dim
        self.num_head = num_head
        self.total_head = total_head
        self.window_size = window_size
        self.window_area = window_size ** 2
        self.shift = shift
        self.helper = helper

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_head, 1, 1))), requires_grad=True)

        # define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        # print(window_size,num_head)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_head))

        # get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(window_size, window_size))

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Conv2d(dim * num_head, dim * num_head, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0)

    def forward(self, qkv):
        B, L, C, H, W = qkv.size()
        # window shift
        # print(qkv.shape)
        # print(self.shift)
        # 对 qkv 进行了平移，使得 qkv 中的特征图在最后两个维度上分别向左和向上平移4个单位，同
        # 时保持数据不丢失，通过循环的方式从另一端进入。
        # 这有助于在图像特征中引入跨窗口的上下文信息，使得模型能学习到更丰富的特征。
        # print(self.shift)
        if self.shift > 0:
            qkv = torch.roll(qkv, shifts=(-self.shift, -self.shift), dims=(-2, -1))

        # window partition
        # print(qkv.shape)#torch.Size([1, 3, 270, 40, 32])
        q, k, v = rearrange(qkv, 'b l c (h wh) (w ww) -> (b h w) l (wh ww) c',
                            wh=self.window_size, ww=self.window_size).chunk(3, dim=-1)  # [B_, L1, hw, C/L] respectively
        # print(q.shape)#([20, 3, 64, 90])
        # print(k.shape)
        # print(ch)

        # a = F.normalize(q, dim=-1)
        # print(a.shape)

        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(2, -1)  # [B_, L1, hw, hw]
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
        attn = attn * logit_scale

        attn = attn + self._get_rel_pos_bias()
        attn = self.attn_drop(F.softmax(attn, dim=-1))

        x = attn @ v  # [B_, L1, hw, C/L]

        # window unpartition + head merge
        x = window_unpartition(x, (H, W), self.window_size)  # [B, L1*C/L, H, W]
        x = self.proj_drop(self.proj(x))

        # window reverse shift
        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(-2, -1))

        return x

    def flops(self, resolutions):
        H, W = resolutions
        num_wins = H // self.window_size * W // self.window_size
        flops = self.num_head * H * W * self.dim if self.helper else 0  # v = v*ch
        flops += num_wins * self.num_head * self.window_area * self.dim * self.window_area  # attn = Q@K^T
        flops += num_wins * self.num_head * self.window_area * self.window_area * self.dim  # attn@V
        flops += H * W * 1 * 1 * self.num_head * self.dim * self.num_head * self.dim  # self.proj
        return flops


@register_notrace_function
def window_unpartition(x, resolutions, window_size):
    return rearrange(x, '(b h w) l (wh ww) c -> b (l c) (h wh) (w ww)',
                     h=resolutions[0] // window_size, w=resolutions[1] // window_size, wh=window_size)


class ChannelSelfAttention(nn.Module):
    def __init__(self, dim, num_head, total_head, attn_drop=0.0, proj_drop=0.0, helper=True):
        super(ChannelSelfAttention, self).__init__()
        self.dim = dim
        self.num_head = num_head
        self.total_head = total_head
        self.helper = helper

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_head, 1, 1))), requires_grad=True)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Conv2d(dim * num_head, dim * num_head, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, qkv):
        B, L, C, H, W = qkv.size()
        # print(sp)
        q, k, v = rearrange(qkv, 'b l c h w -> b l c (h w)').chunk(3, dim=-2)  # [B, L2, C/L, HW]
        # print(q.shape, k.shape)#W=4 1, 1, 192, 1024

        # print(q.shape)
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(2, -1)  # [B, L2, C/L, C/L]
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
        attn = attn * logit_scale

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v  # [B, L2, C/L, HW]

        # head merge
        x = rearrange(x, 'b l c (h w) -> b (l c) h w', h=H)  # [B, L2*C/L, H, W]
        x = self.proj_drop(self.proj(x))  # [B, L2*C/L, H, W]

        return x

    def flops(self, resolutions):
        H, W = resolutions
        flops = self.num_head * self.dim * H * W if self.helper else 0  # v = v*sp
        flops += self.num_head * self.dim * H * W * self.dim  # attn = Q@K^T
        flops += self.num_head * self.dim * self.dim * H * W  # attn@V
        flops += H * W * 1 * 1 * self.num_head * self.dim * self.num_head * self.dim  # self.proj
        return flops


class ReshapeLayerNorm(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super(ReshapeLayerNorm, self).__init__()

        self.dim = dim
        self.norm = norm_layer(dim)

    def forward(self, x):
        B, C, H, W = x.size()
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H)
        return x

    def flops(self, resolutions):
        H, W = resolutions
        flops = 0
        flops += H * W * self.dim
        return flops


class MobiVari1(nn.Module):  # MobileNet v1 Variants
    def __init__(self, dim, kernel_size, stride, act=nn.LeakyReLU, out_dim=None):
        super(MobiVari1, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.out_dim = out_dim or dim

        self.dw_conv = nn.Conv2d(dim, dim, kernel_size, stride, kernel_size // 2, groups=dim)
        self.pw_conv = nn.Conv2d(dim, self.out_dim, 1, 1, 0)
        self.act = act()

    def forward(self, x):
        out = self.act(self.pw_conv(self.act(self.dw_conv(x)) + x))
        return out + x if self.dim == self.out_dim else out

    def flops(self, resolutions):
        H, W = resolutions
        flops = H * W * self.kernel_size * self.kernel_size * self.dim + H * W * 1 * 1 * self.dim * self.out_dim  # self.dw_conv + self.pw_conv
        return flops


class MobiVari2(MobiVari1):  # MobileNet v2 Variants
    def __init__(self, dim, kernel_size, stride, act=nn.LeakyReLU, out_dim=None, exp_factor=1.2, expand_groups=4):
        super(MobiVari2, self).__init__(dim, kernel_size, stride, act, out_dim)
        self.expand_groups = expand_groups
        expand_dim = int(dim * exp_factor)
        expand_dim = expand_dim + (expand_groups - expand_dim % expand_groups)
        self.expand_dim = expand_dim

        self.exp_conv = nn.Conv2d(dim, self.expand_dim, 1, 1, 0, groups=expand_groups)
        self.dw_conv = nn.Conv2d(expand_dim, expand_dim, kernel_size, stride, kernel_size // 2, groups=expand_dim)
        self.pw_conv = nn.Conv2d(expand_dim, self.out_dim, 1, 1, 0)

    def forward(self, x):
        x1 = self.act(self.exp_conv(x))
        out = self.pw_conv(self.act(self.dw_conv(x1) + x1))
        return out + x if self.dim == self.out_dim else out

    def flops(self, resolutions):
        H, W = resolutions
        flops = H * W * 1 * 1 * (self.dim // self.expand_groups) * self.expand_dim  # self.exp_conv
        flops += H * W * self.kernel_size * self.kernel_size * self.expand_dim  # self.dw_conv
        flops += H * W * 1 * 1 * self.expand_dim * self.out_dim  # self.pw_conv
        return flops


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_ratio, act_layer=nn.GELU, bias=True, drop=0.0):
        super(FeedForward, self).__init__()

        self.dim = dim
        self.hidden_ratio = hidden_ratio

        self.hidden = nn.Conv2d(dim, int(dim * hidden_ratio), 1, bias=bias)
        self.drop1 = nn.Dropout(drop)
        self.out = nn.Conv2d(int(dim * hidden_ratio), dim, 1, bias=bias)
        self.drop2 = nn.Dropout(drop)
        self.act = act_layer()

    def forward(self, x):
        return self.drop2(self.out(self.drop1(self.act(self.hidden(x)))))

    def flops(self, resolutions):
        H, W = resolutions
        flops = 2 * H * W * 1 * 1 * self.dim * self.dim * self.hidden_ratio  # self.hidden + self.out
        return flops


class NoLayer(nn.Identity):
    def __init__(self):
        super(NoLayer, self).__init__()

    def flops(self, resolutions):
        return 0

    def forward(self, x, **kwargs):
        return x.flatten(1, 2)


class CATransformer1(nn.Module):  # Reciprocal Attention Transformer Block
    def __init__(self, dim, num_head, chsa_head_ratio=0.25, wtkernel=3, window_size=8, shift=0, head_dim=None,
                 qkv_bias=True, mv_ver=1,
                 hidden_ratio=2.0, act_layer=nn.GELU, norm_layer=ReshapeLayerNorm, attn_drop=0.0, proj_drop=0.0,
                 drop_path=0.0, helper=True,
                 mv_act=nn.LeakyReLU, exp_factor=1.2, expand_groups=4):
        super(CATransformer1, self).__init__()

        self.dim = dim
        self.num_head = num_head
        self.window_size = window_size
        self.chsa_head = int(num_head * chsa_head_ratio)
        self.shift = shift
        self.helper = helper
        # print(num_head,self.chsa_head, window_size, shift)
        self.qkv_proj = QKVProjection(dim, num_head, qkv_bias=qkv_bias)
        self.sp_attn = SpatialSelfAttention(dim // num_head, num_head - self.chsa_head, num_head,
                                            window_size, shift, attn_drop, proj_drop,
                                            helper) if num_head - self.chsa_head != 0 else NoLayer()
        self.ch_attn = ChannelSelfAttention(dim // num_head, self.chsa_head, num_head, attn_drop, proj_drop,
                                            helper) if self.chsa_head != 0 else NoLayer()
        if mv_ver == 1:
            self.mobivari = MobiVari1(dim, 3, 1, act=mv_act)
        elif mv_ver == 2:
            self.mobivari = MobiVari2(dim, 3, 1, act=mv_act, out_dim=None, exp_factor=exp_factor,
                                      expand_groups=expand_groups)

        self.norm1 = norm_layer(dim)
        self.PReLU = nn.PReLU()
        self.ffn = FeedForward(dim, hidden_ratio, act_layer=act_layer)
        self.norm2 = norm_layer(dim)

        self.Convcat = nn.Conv2d(2 * dim, dim, kernel_size=1, stride=1, padding=0)
        self.WTconv = WTConv2d(dim, dim, wtkernel)
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, t2):
        B, C, H, W = x.size()

        # QKV projection + head split
        qkv = self.qkv_proj(x)  # [B, L, C, H, W]

        # SP-SA / CH-SA
        sp = self.sp_attn(qkv[:, :self.num_head - self.chsa_head])  # [B, L1*C/L, H, W]

        ch = self.ch_attn(qkv[:, self.num_head - self.chsa_head:])  # [B, L2*C/L, H, W]

        # print(t2)
        if t2 is not None:
            # print(t2.shape)
            attn0 = self.Convcat(torch.cat([sp, ch, t2], dim=1))
            # print(attn0.shape)
            attn0 = self.WTconv(attn0)  # merge [B, C, H, W]
            attn0 = self.PReLU(attn0)
            # print(attn0)
            # print(sp.shape, ch.shape, t2.shape)
        else:
            attn0 = self.WTconv(torch.cat([sp, ch], dim=1))
            attn0 = self.PReLU(attn0)

        attn = self.norm1(attn0) + x  # LN, skip connection [B, C, H, W]

        # print(attn.shape)
        # FFN
        out = self.norm2(self.ffn(attn)) + attn  # FFN, LN, skip connection [B, C, H, W]

        return out, attn0


class CATransformer2(nn.Module):  # Reciprocal Attention Transformer Block
    def __init__(self, dim, num_head, chsa_head_ratio=0.25, wtkernel=3, window_size=8, shift=0, head_dim=None,
                 qkv_bias=True, mv_ver=1,
                 hidden_ratio=2.0, act_layer=nn.GELU, norm_layer=ReshapeLayerNorm, attn_drop=0.0, proj_drop=0.0,
                 drop_path=0.0, helper=True,
                 mv_act=nn.LeakyReLU, exp_factor=1.2, expand_groups=4):
        super(CATransformer2, self).__init__()

        self.dim = dim
        self.num_head = num_head
        self.window_size = window_size
        self.chsa_head = int(num_head * chsa_head_ratio)
        self.shift = shift
        self.helper = helper
        # print(num_head,self.chsa_head, window_size, shift)
        self.qkv_proj = QKVProjection(dim, num_head, qkv_bias=qkv_bias)
        self.sp_attn = SpatialSelfAttention(dim // num_head, num_head - self.chsa_head, num_head,
                                            window_size, shift, attn_drop, proj_drop,
                                            helper) if num_head - self.chsa_head != 0 else NoLayer()
        self.ch_attn = ChannelSelfAttention(dim // num_head, self.chsa_head, num_head, attn_drop, proj_drop,
                                            helper) if self.chsa_head != 0 else NoLayer()
        if mv_ver == 1:
            self.mobivari = MobiVari1(dim, 3, 1, act=mv_act)
        elif mv_ver == 2:
            self.mobivari = MobiVari2(dim, 3, 1, act=mv_act, out_dim=None, exp_factor=exp_factor,
                                      expand_groups=expand_groups)

        self.norm1 = norm_layer(dim)

        self.ffn = FeedForward(dim, hidden_ratio, act_layer=act_layer)
        self.norm2 = norm_layer(dim)
        self.PReLU = nn.PReLU()
        self.Convcat = nn.Conv2d(2 * dim, dim, kernel_size=1, stride=1, padding=0)
        self.WTconv = WTConv2d(dim, dim, wtkernel)

        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, t1):
        B, C, H, W = x.size()
        # print(t1)
        # QKV projection + head split
        qkv = self.qkv_proj(x)  # [B, L, C, H, W]

        # SP-SA / CH-SA
        sp = self.sp_attn(qkv[:, :self.num_head - self.chsa_head])  # [B, L1*C/L, H, W]

        ch = self.ch_attn(qkv[:, self.num_head - self.chsa_head:])  # [B, L2*C/L, H, W]
        # print(t1)
        if t1 is not None:
            # print(t1.shape)
            attn0 = self.Convcat(torch.cat([sp, ch, t1], dim=1))
            # print(attn0.shape)
            attn0 = self.WTconv(attn0)  # merge [B, C, H, W]
            attn0 = self.PReLU(attn0)
            # print(sp.shape, ch.shape, t1.shape)
        else:
            attn0 = self.WTconv(torch.cat([sp, ch], dim=1))
            attn0 = self.PReLU(attn0)

        attn = self.norm1(attn0) + x  # LN, skip connection [B, C, H, W]

        # print(attn.shape)
        # FFN
        out = self.norm2(self.ffn(attn)) + attn  # FFN, LN, skip connection [B, C, H, W]

        return out, attn0


class DT(nn.Module):
    def __init__(self, depth, dim, num_head, chsa_head_ratio, wtkernel, window_size=8, head_dim=None,
                 qkv_bias=True, mv_ver=1, hidden_ratio=2.0, act_layer=nn.GELU, norm_layer=ReshapeLayerNorm,
                 attn_drop=0.0, proj_drop=0.0, drop_path=0.0, helper=True, mv_act=nn.LeakyReLU):
        super(DT, self).__init__()
        self.depth = depth
        self.dim = dim
        self.num_head = num_head

        self.window_size_1 = window_size
        self.window_size_2 = window_size // 2
        self.shift1 = self.window_size_1 // 2
        self.shift2 = self.window_size_2 // 2
        self.wtkernel1 = wtkernel
        self.wtkernel2 = wtkernel - 2

        self.t1 = CATransformer1(dim, num_head, chsa_head_ratio, self.wtkernel1, self.window_size_1, self.shift1,
                                    head_dim, qkv_bias,
                                    mv_ver, hidden_ratio, act_layer, norm_layer, attn_drop, proj_drop, drop_path,
                                    helper, mv_act)
        self.t2 = CATransformer2(dim, num_head, chsa_head_ratio, self.wtkernel2, self.window_size_2, self.shift2,
                                    head_dim, qkv_bias,
                                    mv_ver, hidden_ratio, act_layer, norm_layer, attn_drop, proj_drop, drop_path,
                                    helper, mv_act)
        self.convk1 = nn.Conv2d(2 * dim, dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x, t1=None, t2=None):
        # CATransformer1 的输出作为 t2 的输入
        out1, attn1 = self.t1(x, t2)

        # CATransformer2 的输出作为 t1 的输入
        out2, attn2 = self.t2(x, t1)

        # 合并输出
        out = out1 + out2
        attn = torch.cat([attn1, attn2], dim=1)
        attn = self.convk1(attn)
        attn = F.softmax(attn, dim=1)

        # 应用权重
        out = out * attn
        # print(out.shape,attn1.shape,attn2.shape)
        # 返回更新后的 t1 和 t2
        return out, attn1, attn2, attn


class Encoder(nn.Module):
    def __init__(self, depth, dim, num_head, chsa_head_ratio, wtkernel, window_size=8, head_dim=None,
                 qkv_bias=True, mv_ver=1, hidden_ratio=2.0, act_layer=nn.GELU, norm_layer=ReshapeLayerNorm,
                 attn_drop=0.0, proj_drop=0.0, drop_path=0.0, helper=True, mv_act=nn.LeakyReLU):
        super(Encoder, self).__init__()
        self.block = DT(depth, dim, num_head, chsa_head_ratio, wtkernel, window_size, head_dim,
                        qkv_bias, mv_ver, hidden_ratio, act_layer, norm_layer,
                        attn_drop, proj_drop, drop_path, helper, mv_act)

    def forward(self, x, num_iterations=3):
        # 初始化 t1 和 t2 为 None
        t1, t2 = None, None

        for i in range(num_iterations):  # 设置多次迭代进行交互
            if i == 0:
                out, t1, t2, attn = self.block(x, t1, t2)  # 逐步更新 t1 和 t2
            else:
                out, t1, t2, attn = self.block(out, t1, t2)

        return out, t1, t2, attn


@MODELS.register_module()
class CITransformerHead(BaseDecodeHead):

    def __init__(
            self,
            in_channels,
            num_layers,
            num_heads,
            embed_dims,
            mlp_ratio=4,
            drop_path_rate=0.1,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            num_fcs=2,
            qkv_bias=True,
            act_cfg=dict(type='GELU'),
            norm_cfg=dict(type='LN'),
            init_std=0.02,
            **kwargs,
    ):
        super().__init__(in_channels=in_channels, **kwargs)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = ModuleList()
        for i in range(1):
            self.layers.append(
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * embed_dims,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=num_fcs,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    batch_first=True,
                ))

        self.dec_proj = nn.Linear(in_channels, embed_dims)

        ############
        self.window_size = 8
        shift = self.window_size // 2
        self.head_dim = None
        self.CStrans = Encoder(depth=2, dim=embed_dims, num_head=4, chsa_head_ratio=0.25, window_size=8, qkv_bias=True,
                               mv_ver=1, hidden_ratio=2.0, act_layer=nn.GELU, norm_layer=ReshapeLayerNorm,
                               attn_drop=0.0, proj_drop=0.0, drop_path=0.0, helper=True, mv_act=nn.LeakyReLU,
                               wtkernel=5)
        ###########

        self.selected_classes = [0, 1]
        self.cls_emb = nn.Parameter(
            torch.randn(1, self.num_classes, embed_dims))
        self.patch_proj = nn.Linear(embed_dims, embed_dims, bias=False)
        self.classes_proj = nn.Linear(embed_dims, embed_dims, bias=False)

        self.decoder_norm = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)[1]
        self.mask_norm = build_norm_layer(
            norm_cfg, self.num_classes, postfix=2)[1]

        self.BatchNorm2d = nn.BatchNorm2d(embed_dims)

        self.init_std = init_std
        ############################
        self.convbd = nn.Conv2d(embed_dims, embed_dims // 2, kernel_size=3, padding=1)
        self.normbd = nn.InstanceNorm2d(embed_dims // 2)
        self.actbd = nn.PReLU()
        self.convbd2 = nn.Conv2d(embed_dims // 2, 1, kernel_size=1)

        # self.cls_seg = nn.Conv2d(in_channels // 4, 1, kernel_size=1)
        # self.GN = nn.GroupNorm(num_groups=8, num_channels=embed_dims)
        self.classifier_1 = nn.Conv2d(embed_dims, 2, kernel_size=1, stride=1, bias=True)
        # self.convforsig = nn.Conv2d(3, 1, kernel_size=7, stride=1, padding=3, bias=False)
        # self.sigmoid = nn.Sigmoid()
        #######################

        self.feat = nn.Conv2d(in_channels, embed_dims, kernel_size=1, stride=1, bias=True)
        self.classifier_feat = nn.Conv2d(embed_dims, 2, kernel_size=1, stride=1, bias=True)
        self.SAW_stage = SAW(embed_dims, relax_denom=2.0, classifier=self.classifier_feat)

        #######################
        delattr(self, 'conv_seg')

    def init_weights(self):
        trunc_normal_(self.cls_emb, std=self.init_std)
        trunc_normal_init(self.patch_proj, std=self.init_std)
        trunc_normal_init(self.classes_proj, std=self.init_std)
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=self.init_std, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

    def forward(self, inputs):
        # print(inputs[0].shape)
        # [1, 768, 32, 32]
        if self.training:
            x = self._transform_inputs(inputs)
            ########
            feat1 = self.feat(x)
            feat = self.classifier_feat(feat1)
            SAW_loss = self.SAW_stage(feat1)
            ############

            # print(bd.shape)
            x, t1, t2, attn = self.CStrans(feat1)

            bd = self.convbd(attn)
            bd = self.normbd(bd)
            bd = self.actbd(bd)
            bd = self.convbd2(bd)

            # print(x.shape)
            x = self.BatchNorm2d(x)

            masks = self.classifier_1(x)

            return masks, feat, bd, SAW_loss

        else:

            x = self._transform_inputs(inputs)
            ########
            feat1 = self.feat(x)

            ############

            # print(bd.shape)
            x, t1, t2, attn = self.CStrans(feat1)

            # print(x.shape)
            x = self.BatchNorm2d(x)

            masks = self.classifier_1(x)

            return masks

    def _stack_batch_gt(self, batch_data_samples: SampleList) -> Tuple[Tensor]:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in batch_data_samples
        ]
        gt_edge_segs = [
            data_sample.gt_edge_map.data for data_sample in batch_data_samples
        ]

        # print(batch_data_samples)
        gt_sem_segs = torch.stack(gt_semantic_segs, dim=0)
        gt_edge_segs = torch.stack(gt_edge_segs, dim=0)
        return gt_sem_segs, gt_edge_segs

    #
    def loss_by_feat(self, seg_logits: Tuple[Tensor],
                     batch_data_samples: SampleList) -> dict:
        loss = dict()

        masks, feat, bd, SAW_loss = seg_logits
        sem_label, bd_label = self._stack_batch_gt(batch_data_samples)
        #############################
        # print(name)
        # import os
        # filename = os.path.basename(name[0])
        # #print(filename)
        # #
        # from PIL import Image
        # image_data = bd_label.squeeze().cpu().numpy()
        # image_data = (image_data * 255).astype('uint8')
        # image = Image.fromarray(image_data, mode='L')
        # name = 'bianyuan/'+filename
        # image.save(name)

        ###############################
        # print(bd_label.shape)

        masks_logit = resize(
            masks,
            size=sem_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        feat_logit = resize(
            feat,
            size=sem_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        bd_logit = resize(
            bd,
            size=bd_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)

        sem_label = sem_label.squeeze(1)
        bd_label = bd_label.squeeze(1)

        loss['loss_smg_mask'] = self.loss_decode[0](masks_logit, sem_label)
        loss['loss_smg_feat'] = self.loss_decode[0](feat_logit, sem_label)
        loss['loss_bd'] = self.loss_decode[1](bd_logit, bd_label)
        loss['SAW_loss'] = (0.01 * SAW_loss).detach().data

        loss['acc_seg'] = accuracy(masks_logit, sem_label, ignore_index=self.ignore_index)

        return loss

# python tools/train.py configs/segmenter/mysementer.py