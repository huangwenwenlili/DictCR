## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from einops import rearrange

from .ops_utils import *
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
##########################################################################

def img2windows(img, H_sp, W_sp):
    """
    Input: Image (B, C, H, W)
    Output: Window Partition (B', N, C)
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    Input: Window Partition (B', N, C)
    Output: Image (B, H, W, C)
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img

class DynamicPosBias(nn.Module):
    # The implementation builds on Crossformer code https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
    """ Dynamic Relative Position Bias.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        residual (bool):  If True, use residual strage to connect conv.
    """

    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases)  # 2Gh-1 * 2Gw-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos


class WindowAttention(nn.Module):
    def __init__(self, dim, idx, split_size=[8, 8], dim_out=None, num_heads=6, attn_drop=0., proj_drop=0.,
                 qk_scale=None, position_bias=True):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        if idx == 0:
            H_sp, W_sp = self.split_size[0], self.split_size[1]
        elif idx == 1:
            W_sp, H_sp = self.split_size[0], self.split_size[1]
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp

        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 2, self.num_heads, residual=False)
            # generate mother-set
            position_bias_h = torch.arange(1 - self.H_sp, self.H_sp)
            position_bias_w = torch.arange(1 - self.W_sp, self.W_sp)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()
            self.register_buffer('rpe_biases', biases)

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.H_sp)
            coords_w = torch.arange(self.W_sp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.H_sp - 1
            relative_coords[:, :, 1] += self.W_sp - 1
            relative_coords[:, :, 0] *= 2 * self.W_sp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer('relative_position_index', relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2win(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, qkv, H, W, mask=None):
        """
        Input: qkv: (B, 3*L, C), H, W, mask: (B, N, N), N is the window size
        Output: x (B, H, W, C)
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        # partition the q,k,v, image to window
        q = self.im2win(q, H, W)
        k = self.im2win(k, H, W)
        v = self.im2win(v, H, W)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N

        # calculate drpe
        if self.position_bias:
            pos = self.pos(self.rpe_biases)
            # select position bias
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        N = attn.shape[3]

        # use mask for shift window
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C

        # merge the window, window to image
        x = windows2img(x, self.H_sp, self.W_sp, H, W)  # B H' W' C

        return x


class L_SA(nn.Module):
    # The implementation builds on CAT code https://github.com/zhengchen1999/CAT/blob/main/basicsr/archs/cat_arch.py
    def __init__(self, dim, num_heads,
                 split_size=[8, 32], shift_size=[4, 16], qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., idx=0, reso=64, rs_id=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.shift_size = shift_size
        self.idx = idx
        self.rs_id = rs_id
        self.patches_resolution = reso
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        assert 0 <= self.shift_size[0] < self.split_size[0], "shift_size must in 0-split_size0"
        assert 0 <= self.shift_size[1] < self.split_size[1], "shift_size must in 0-split_size1"

        self.branch_num = 2

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.attns = nn.ModuleList([
            WindowAttention(
                dim // 2, idx=i,
                split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, position_bias=True)
            for i in range(self.branch_num)])

        if (self.rs_id % 2 == 0 and self.idx > 0 and (self.idx - 1) % 2 == 0) or (
                self.rs_id % 2 != 0 and self.idx % 2 == 0):
            attn_mask = self.calculate_mask(self.patches_resolution, self.patches_resolution)

            self.register_buffer("attn_mask_0", attn_mask[0])
            self.register_buffer("attn_mask_1", attn_mask[1])
        else:
            attn_mask = None

            self.register_buffer("attn_mask_0", None)
            self.register_buffer("attn_mask_1", None)

        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)  # DW Conv

    def calculate_mask(self, H, W):
        # The implementation builds on Swin Transformer code https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
        # calculate attention mask for Rwin
        img_mask_0 = torch.zeros((1, H, W, 1))  # 1 H W 1 idx=0
        img_mask_1 = torch.zeros((1, H, W, 1))  # 1 H W 1 idx=1
        h_slices_0 = (slice(0, -self.split_size[0]),
                      slice(-self.split_size[0], -self.shift_size[0]),
                      slice(-self.shift_size[0], None))
        w_slices_0 = (slice(0, -self.split_size[1]),
                      slice(-self.split_size[1], -self.shift_size[1]),
                      slice(-self.shift_size[1], None))

        h_slices_1 = (slice(0, -self.split_size[1]),
                      slice(-self.split_size[1], -self.shift_size[1]),
                      slice(-self.shift_size[1], None))
        w_slices_1 = (slice(0, -self.split_size[0]),
                      slice(-self.split_size[0], -self.shift_size[0]),
                      slice(-self.shift_size[0], None))
        cnt = 0
        for h in h_slices_0:
            for w in w_slices_0:
                img_mask_0[:, h, w, :] = cnt
                cnt += 1
        cnt = 0
        for h in h_slices_1:
            for w in w_slices_1:
                img_mask_1[:, h, w, :] = cnt
                cnt += 1

        # calculate mask for H-Shift
        img_mask_0 = img_mask_0.view(1, H // self.split_size[0], self.split_size[0], W // self.split_size[1],
                                     self.split_size[1], 1)
        img_mask_0 = img_mask_0.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size[0], self.split_size[1],
                                                                            1)  # nW, sw[0], sw[1], 1
        mask_windows_0 = img_mask_0.view(-1, self.split_size[0] * self.split_size[1])
        attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
        attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 != 0, float(-100.0)).masked_fill(attn_mask_0 == 0, float(0.0))

        # calculate mask for V-Shift
        img_mask_1 = img_mask_1.view(1, H // self.split_size[1], self.split_size[1], W // self.split_size[0],
                                     self.split_size[0], 1)
        img_mask_1 = img_mask_1.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size[1], self.split_size[0],
                                                                            1)  # nW, sw[1], sw[0], 1
        mask_windows_1 = img_mask_1.view(-1, self.split_size[1] * self.split_size[0])
        attn_mask_1 = mask_windows_1.unsqueeze(1) - mask_windows_1.unsqueeze(2)
        attn_mask_1 = attn_mask_1.masked_fill(attn_mask_1 != 0, float(-100.0)).masked_fill(attn_mask_1 == 0, float(0.0))

        return attn_mask_0, attn_mask_1

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """

        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3)  # 3, B, HW, C
        # v without partition
        v = qkv[2].transpose(-2, -1).contiguous().view(B, C, H, W)

        max_split_size = max(self.split_size[0], self.split_size[1])
        pad_l = pad_t = 0
        pad_r = (max_split_size - W % max_split_size) % max_split_size
        pad_b = (max_split_size - H % max_split_size) % max_split_size

        qkv = qkv.reshape(3 * B, H, W, C).permute(0, 3, 1, 2)  # 3B C H W
        qkv = F.pad(qkv, (pad_l, pad_r, pad_t, pad_b)).reshape(3, B, C, -1).transpose(-2, -1)  # l r t b
        _H = pad_b + H
        _W = pad_r + W
        _L = _H * _W

        if (self.rs_id % 2 == 0 and self.idx > 0 and (self.idx - 1) % 2 == 0) or (
                self.rs_id % 2 != 0 and self.idx % 2 == 0):
            qkv = qkv.view(3, B, _H, _W, C)
            # H-Shift
            qkv_0 = torch.roll(qkv[:, :, :, :, :C // 2], shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(2, 3))
            qkv_0 = qkv_0.view(3, B, _L, C // 2)
            # V-Shift
            qkv_1 = torch.roll(qkv[:, :, :, :, C // 2:], shifts=(-self.shift_size[1], -self.shift_size[0]), dims=(2, 3))
            qkv_1 = qkv_1.view(3, B, _L, C // 2)

            if self.patches_resolution != _H or self.patches_resolution != _W:
                mask_tmp = self.calculate_mask(_H, _W)
                # H-Rwin
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=mask_tmp[0].to(x.device))
                # V-Rwin
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=mask_tmp[1].to(x.device))

            else:
                # H-Rwin
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=self.attn_mask_0)
                # V-Rwin
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=self.attn_mask_1)

            x1 = torch.roll(x1_shift, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
            x2 = torch.roll(x2_shift, shifts=(self.shift_size[1], self.shift_size[0]), dims=(1, 2))
            x1 = x1[:, :H, :W, :].reshape(B, L, C // 2)
            x2 = x2[:, :H, :W, :].reshape(B, L, C // 2)
            # Concat
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            # V-Rwin
            x1 = self.attns[0](qkv[:, :, :, :C // 2], _H, _W)[:, :H, :W, :].reshape(B, L, C // 2)
            # H-Rwin
            x2 = self.attns[1](qkv[:, :, :, C // 2:], _H, _W)[:, :H, :W, :].reshape(B, L, C // 2)
            # Concat
            attened_x = torch.cat([x1, x2], dim=2)

        # mix
        lcm = self.get_v(v)
        lcm = lcm.permute(0, 2, 3, 1).contiguous().view(B, L, C)

        x = attened_x + lcm

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Gate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)  # DW Conv

    def forward(self, x, H, W):
        # Split
        x1, x2 = x.chunk(2, dim=-1)
        B, N, C = x.shape
        x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C // 2, H, W)).flatten(2).transpose(-1,
                                                                                                              -2).contiguous()

        return x1 * x2


class MLP_gate_rgt(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = Gate(hidden_features // 2)
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.sg(x, H, W)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x

class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, H, W):
        x_size = (H, W)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x

##########################################################################
class Window_Attn(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio,
                 convffn_kernel_size = 5,
                 reso=64, idx=0,split_size = [8, 32],
                 qkv_bias=True,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Window_Attn, self).__init__()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.convffn_kernel_size = convffn_kernel_size
        # win and shift win 交替
        split_size = split_size
        shift_size = [split_size[0]//2, split_size[1]//2]
        reso = reso
        self.attn = L_SA(
            dim, split_size=split_size, shift_size=shift_size, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=None, attn_drop=0.,
            drop=0., idx=idx, reso=reso
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_hidden_dim = mlp_hidden_dim + mlp_hidden_dim%2
        self.convffn = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, kernel_size=convffn_kernel_size,
                               act_layer=act_layer)
        # self.convffn = MLP_gate_rgt(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim,
        #                             act_layer=act_layer)


    def forward(self, x):
        b, c, H, W = x.shape
        # h, w = x_size
        # b, n, c = x.shape
        x = rearrange(x, 'b c h w-> b (h w) c')
        res = x
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.convffn(self.norm2(x), H, W)
        x = rearrange(x, 'b (h w ) c-> b c h w', h=H, w=W)

        return x


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.reduction = nn.Linear(4 * in_dim, out_dim, bias=False)

    def forward(self, x, x_size):
        """
        x: B, H*W, C
        """
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)

        return x

class Hourglass(nn.Module):
    r"""Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
    """

    def __init__(self, in_dim, out_dim,down_factor,output_size=7):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.reduction = nn.Linear(4 * in_dim, out_dim, bias=False)
        dim_hg = in_dim * down_factor * down_factor
        hk = 3
        self.hourglass_embedding = nn.Sequential(
            Rearrange('b (h p1) (w p2) c -> b h w (c p1 p2)', p1=down_factor, p2=down_factor),
            nn.LayerNorm(dim_hg),
            Rearrange('b h w c -> b c h w'),
            nn.Conv2d(dim_hg, in_dim, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(in_dim, out_dim, kernel_size=hk, stride=1, padding=hk // 2, groups=out_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(output_size=(output_size, output_size)),
            # nn.Conv2d(dim_hg_e, dim_hg, kernel_size=1, padding=0),
            Rearrange('b c h w -> b h w c'),
            nn.LayerNorm(out_dim),
        )


    def forward(self, x, x_size):
        """
        x: B, H*W, C
        """
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        # x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        # x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        # x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        # x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        # x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        # x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        # x = self.reduction(x)

        x = self.hourglass_embedding(x)

        return x

class AnchorLinear(nn.Module):
    r"""Linear anchor projection layer
    Args:
        dim (int): Number of input channels.
    """

    def __init__(self, in_channels, out_channels, down_factor, pooling_mode, bias):
        super().__init__()
        self.down_factor = down_factor
        if pooling_mode == "maxpool":
            self.pooling = nn.MaxPool2d(down_factor, down_factor)
        elif pooling_mode == "avgpool":
            self.pooling = nn.AvgPool2d(down_factor, down_factor)
        self.reduction = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x, x_size):
        """
        x: B, H*W, C
        """
        x = blc_to_bchw(x, x_size)
        x = bchw_to_blc(self.pooling(x))
        x = blc_to_bhwc(self.reduction(x), [s // self.down_factor for s in x_size])
        return x


class AnchorProjection(nn.Module):
    def __init__(self, dim, proj_type, one_stage, anchor_window_down_factor, downc_ratio=1,pool_size=7):
        super(AnchorProjection, self).__init__()
        self.proj_type = proj_type
        self.body = nn.ModuleList([])
        if one_stage:
            if proj_type == "patchmerging":
                m = PatchMerging(dim, dim // downc_ratio)
            elif proj_type == "conv2d":
                kernel_size = anchor_window_down_factor + 1
                stride = anchor_window_down_factor
                padding = kernel_size // 2
                m = nn.Conv2d(dim, dim // downc_ratio, kernel_size, stride, padding)
            elif proj_type == "separable_conv":
                kernel_size = anchor_window_down_factor + 1
                stride = anchor_window_down_factor
                m = SeparableConv(dim, dim // downc_ratio, kernel_size, stride, True)
            elif proj_type.find("adapool") >= 0:
                m = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
            elif proj_type.find("pool") >= 0:
                m = AnchorLinear(
                    dim, dim // downc_ratio, anchor_window_down_factor, proj_type, True
                )
            elif proj_type.find("hourglass") >= 0:
                m = Hourglass(dim, dim // downc_ratio, anchor_window_down_factor,output_size=pool_size)

            self.body.append(m)
        else:
            for i in range(int(math.log2(anchor_window_down_factor))):
                cin = dim if i == 0 else dim // downc_ratio
                if proj_type == "patchmerging":
                    m = PatchMerging(cin, dim // downc_ratio)
                elif proj_type == "conv2d":
                    m = nn.Conv2d(cin, dim // downc_ratio, 3, 2, 1)
                elif proj_type == "separable_conv":
                    m = SeparableConv(cin, dim // downc_ratio, 3, 2, True)
                self.body.append(m)

    def forward(self, x, x_size):
        if self.proj_type.find("conv") >= 0:
            x = blc_to_bchw(x, x_size)
            for m in self.body:
                x = m(x)
            x = bchw_to_bhwc(x)
        elif self.proj_type.find("adapool") >= 0:
            x = blc_to_bchw(x, x_size)
            for m in self.body:
                x = m(x)
            x = bchw_to_bhwc(x)
        elif self.proj_type.find("pool") >= 0:
            for m in self.body:
                x = m(x, x_size)
        elif self.proj_type.find("hourglass") >= 0:
            for m in self.body:
                x = m(x, x_size)
        else:
            for i, m in enumerate(self.body):
                x = m(x, [s // 2**i for s in x_size])
            x = blc_to_bhwc(x, [s // 2 ** (i + 1) for s in x_size])
        return x

class AgentAttention(nn.Module):
    def __init__(self, dim, resolution,  num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 agent_num=49, anchor_proj_type="separable_conv",
        anchor_one_stage=True,
        anchor_window_down_factor=1, **kwargs):
        super().__init__()
        self.dim = dim
        self.resolution = resolution
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.agent_num = agent_num
        self.scale = head_dim ** -0.5
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        # self.scale = qk_scale or head_dim ** -0.5

        H_sp, W_sp = self.resolution, self.resolution

        self.H_sp = H_sp
        self.W_sp = W_sp
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        # self.dwc = nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=1, groups=dim)

        # self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        # self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        # self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, H_sp, 1))
        # self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, W_sp))
        # self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, H_sp, 1, agent_num))
        # self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, W_sp, agent_num))
        # trunc_normal_(self.an_bias, std=.02)
        # trunc_normal_(self.na_bias, std=.02)
        # trunc_normal_(self.ah_bias, std=.02)
        # trunc_normal_(self.aw_bias, std=.02)
        # trunc_normal_(self.ha_bias, std=.02)
        # trunc_normal_(self.wa_bias, std=.02)
        pool_size = int(agent_num ** 0.5)
        # self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self.agentProj = AnchorProjection(
            dim, anchor_proj_type, anchor_one_stage, anchor_window_down_factor,pool_size=pool_size
        )

    def forward(self, x, H, W):
        """
        x: B n C
        """

        ### Img2Window
        b, n, c = x.shape
        # H = W = self.resolution
        # assert n == H * W, "flatten img_tokens has wrong size"

        num_heads = self.num_heads
        head_dim = c // num_heads
        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # q, k, v: b, n, c

        # h, w = self.H_sp, self.W_sp
        # h = int(n ** 0.5)
        # w = h
        h = H
        w = W

        # agent_tokens = self.pool(q.reshape(b, h, w, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        agent_tokens = self.agentProj(x, (h,w)).reshape(b, -1, c)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        # self.agent_num = agent_tokens.shape[1]
        agent_tokens = agent_tokens.reshape(b, -1 , num_heads, head_dim).permute(0, 2, 1, 3)

        # position_bias1 = nn.functional.interpolate(self.an_bias, size=(self.H_sp, self.W_sp), mode='bilinear',align_corners=True)
        # position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        # position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        # position_bias = position_bias1 + position_bias2
        # agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1))
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        # agent_bias1 = nn.functional.interpolate(self.na_bias, size=(self.H_sp, self.W_sp), mode='bilinear',align_corners=True)
        # agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        # agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        # agent_bias = agent_bias1 + agent_bias2
        # q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1))
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)
        # v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        # x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

##########################################################################
class Agent_Attn(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio,
                 convffn_kernel_size = 5,
                 reso=64, idx=0,
                 qkv_bias=True,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 agent_num=49,
                 anchor_proj_type="separable_conv",
                 anchor_one_stage=True,
                 anchor_window_down_factor=1
                 ):
        super(Agent_Attn, self).__init__()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.convffn_kernel_size = convffn_kernel_size
        # win and shift win 交替
        reso = reso
        self.attn = AgentAttention(dim, reso,  num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=0., proj_drop=0.,
                 agent_num=agent_num,anchor_proj_type=anchor_proj_type,
                 anchor_one_stage=anchor_one_stage,
                 anchor_window_down_factor=anchor_window_down_factor)

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_hidden_dim = mlp_hidden_dim + mlp_hidden_dim%2
        self.convffn = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, kernel_size=convffn_kernel_size,
                               act_layer=act_layer)
        # self.convffn = MLP_gate_rgt(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim,
        #                             act_layer=act_layer)


    def forward(self, x):
        b, c, H, W = x.shape
        # h, w = x_size
        # b, n, c = x.shape
        x = rearrange(x, 'b c h w-> b (h w) c')
        x = x + self.attn(self.norm1(x))
        x = x + self.convffn(self.norm2(x), H, W)
        x = rearrange(x, 'b (h w) c-> b c h w', h=H, w=W)

        return x


##########################################################################
class Parallel_Agent_Window_Attn(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio,
                 convffn_kernel_size = 5,
                 reso=64, idx=0, rs_id=0, split_size = [8, 32],
                 qkv_bias=True,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 agent_num=49,
                 anchor_proj_type="separable_conv",
                 anchor_one_stage=True,
                 anchor_window_down_factor=1,
                 out_proj_type ="linear",
                 layerscale_value=1e-4
                 ):
        super(Parallel_Agent_Window_Attn, self).__init__()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.convffn_kernel_size = convffn_kernel_size
        # win and shift win 交替
        reso = reso
        self.attn_agent = AgentAttention(dim, reso,  num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=0., proj_drop=0.,
                 agent_num=agent_num,anchor_proj_type=anchor_proj_type,
                 anchor_one_stage=anchor_one_stage,
                 anchor_window_down_factor=anchor_window_down_factor)

        split_size = split_size
        shift_size = [split_size[0] // 2, split_size[1] // 2]
        self.attn_win = L_SA(
            dim, split_size=split_size, shift_size=shift_size, num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=None, attn_drop=0.,
            drop=0., idx=idx, reso=reso, rs_id=rs_id
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_hidden_dim = mlp_hidden_dim + mlp_hidden_dim%2
        self.convffn = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, kernel_size=convffn_kernel_size,
                               act_layer=act_layer)
        # self.convffn = MLP_gate_rgt(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim,
        #                             act_layer=act_layer)
        self.out_proj_type = out_proj_type
        if self.out_proj_type == "linear":
            self.proj = nn.Linear(dim*2, dim)
        else:
            self.proj = nn.Conv2d(dim*2, dim, 3, 1, 1)

        # HAI
        # self.gamma = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)


    def forward(self, x):
        b, c, H, W = x.shape
        # h, w = x_size
        # b, n, c = x.shape
        x_res = rearrange(x, 'b c h w-> b (h w) c')
        x = self.norm1(x_res)
        # x_window, x_agent = torch.split(x, c // 2, dim=-1)
        x_agent = self.attn_agent(x, H, W)
        x_window = self.attn_win(x, H, W)
        x = torch.cat([x_window, x_agent], dim=-1)
        if self.out_proj_type == "linear":
            x = self.proj(x)
        else:
            x = blc_to_bchw(x, (H,W))
            x = bchw_to_blc(self.proj(x))

        x = x_res + x
        x = x + self.convffn(self.norm2(x), H, W)

        # HAI
        # x = x + (x_res * self.gamma)

        x = rearrange(x, 'b (h w) c-> b c h w', h=H, w=W)

        return x

class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self,x,x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class TransformerBlock(nn.Module):
    def __init__(self, img_size, dim, num_heads, depth, ffn_expansion_factor, split_size = [8, 32],agent_num=49,
                 anchor_proj_type="separable_conv",
                 anchor_one_stage=True,
                 anchor_window_down_factor=1,
                 out_proj_type= "linear",
                 rs_id = 0
                 ):
        super(TransformerBlock, self).__init__()
        self.dim = dim
        self.input_resolution = to_2tuple(img_size)

        num_tokens = 128
        self.blocks = nn.ModuleList()

        for i in range(depth):
            block = Parallel_Agent_Window_Attn(dim, num_heads, mlp_ratio=ffn_expansion_factor,
                         convffn_kernel_size=5,
                         reso=img_size, idx=i, rs_id= rs_id, split_size=split_size,
                         qkv_bias=True,
                         norm_layer=nn.LayerNorm,
                         agent_num=agent_num,
                         anchor_proj_type=anchor_proj_type,
                         anchor_one_stage=anchor_one_stage,
                         anchor_window_down_factor=anchor_window_down_factor,
                         out_proj_type=out_proj_type
                         )

            self.blocks.append(block)
    
    def forward(self, x):
        x_size = x.shape[2]

        for blk in self.blocks:
            x = blk(x)

        return x
       
##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
##---------- Restormer -----------------------
class Restormer(nn.Module):
    def __init__(self, 
        img_size=256,
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        split_size=[8,32],
        agent_nums=[784, 196, 49, 196, 784],
        anchor_proj_type="hourglass",
        anchor_one_stage=True,
        anchor_window_down_factors=[4,4,4,4,4],
        out_proj_type="linear",
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        
        self.encoder_level1 = TransformerBlock(img_size, dim=dim, depth=num_blocks[0], num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, split_size=split_size,
                 agent_num=agent_nums[0],  anchor_proj_type=anchor_proj_type,
                 anchor_one_stage=anchor_one_stage,
                 anchor_window_down_factor=anchor_window_down_factors[0],rs_id=0,out_proj_type= out_proj_type)
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = TransformerBlock(img_size//2, dim=int(dim*2**1), depth=num_blocks[1], num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, split_size=split_size,
                 agent_num=agent_nums[1], anchor_proj_type=anchor_proj_type,
                 anchor_one_stage=anchor_one_stage,
                 anchor_window_down_factor=anchor_window_down_factors[1],rs_id=1,out_proj_type= out_proj_type)
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = TransformerBlock(img_size//4, dim=int(dim*2**2), depth=num_blocks[2], num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, split_size=split_size,
                agent_num=agent_nums[2],anchor_proj_type=anchor_proj_type,
                 anchor_one_stage=anchor_one_stage,
                 anchor_window_down_factor=anchor_window_down_factors[2],rs_id=2,out_proj_type= out_proj_type)

        # self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        # self.latent = TransformerBlock(img_size//8, dim=int(dim*2**3), depth=num_blocks[3], num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        
        # self.up4_3 = Upsample(int(dim*2**2)) ## From Level 4 to Level 3
        # self.reduce_chan_level3 = nn.Conv2d(int(dim*2**2), int(dim*2**2), kernel_size=1, bias=bias)
        # self.decoder_level3 = TransformerBlock(img_size//4, dim=int(dim*2**2), depth=num_blocks[2], num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = TransformerBlock(img_size//2, dim=int(dim*2**1), depth=num_blocks[1], num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, split_size=split_size,
                 agent_num=agent_nums[1],anchor_proj_type=anchor_proj_type,
                 anchor_one_stage=anchor_one_stage,
                 anchor_window_down_factor=anchor_window_down_factors[1],rs_id=3,out_proj_type= out_proj_type)
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = TransformerBlock(img_size, dim=int(dim*2**1), depth=num_blocks[0], num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, split_size=split_size,
                 agent_num=agent_nums[0], anchor_proj_type=anchor_proj_type,
                 anchor_one_stage=anchor_one_stage,
                 anchor_window_down_factor=anchor_window_down_factors[0],rs_id=4,out_proj_type= out_proj_type)
        
        # self.refinement = TransformerBlock(img_size, dim=int(dim*2**1), depth=num_refinement_blocks,  num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        # import pdb 
        # pdb.set_trace()
       
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        # inp_enc_level4 = self.down3_4(out_enc_level3)
        # latent = self.latent(inp_enc_level4)
                        
        # inp_dec_level3 = self.up4_3(latent)
        # inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        # inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        # out_dec_level3 = self.decoder_level3(inp_dec_level3)

        inp_dec_level2 = self.up3_2(out_enc_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        # out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1
