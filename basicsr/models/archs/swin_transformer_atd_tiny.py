## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange

from .ops_utils import *
from timm.models.layers import to_2tuple
##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class AdaNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
      super(AdaNorm, self).__init__()
      self.a_2 = nn.Parameter(torch.ones(features))
      self.b_2 = nn.Parameter(torch.zeros(features))
      self.eps = eps

    def forward(self, x):
      mean = x.mean(-1, keepdim=True)
      std = x.std(-1, keepdim=True)
      x=x-mean
      mean=x.mean(-1,keepdim=True)
      granorm=(1/10*(x-mean)/(std+self.eps)).detach()
      x_norm=(x-x*granorm)/(std+self.eps)
      return x_norm

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



#



##########################################################################
class Window_Attn(nn.Module):
    def __init__(self, dim, num_heads, window_shift, ffn_expansion_factor, bias, LayerNorm_type):
        super(Window_Attn, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type) #
        # self.norm1 = AdaNorm(dim)
        # win and shift win 交替
        self.window_attn = WindowAttention_v2((8,8), num_heads, window_shift, attn_drop=0.0)
        
        self.proj = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1)
      

        self.norm2 = LayerNorm(dim, LayerNorm_type) #AdaNorm(dim)
        # self.norm2 = AdaNorm(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        self.query = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1,groups=dim,bias=bias),)
          

        self.key = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1,groups=dim,bias=bias),
        )

        self.value = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0,bias=bias),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1,groups=dim,bias=bias),
        )#n
    
 

    def forward(self, x, table, index, mask):

        b,c,h,w = x.shape
        x_size = (h,w)
        res = x
        x =  self.norm1(x)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        qkv = torch.stack([q, k, v], dim=0)
    
        #窗口注意力
        
        x_window = self.window_attn(qkv, x_size, table, index, mask) #b, c//2, h,w
        x_out = res + self.proj(x_window)
        x_ffn = x_out + self.ffn(self.norm2(x_out))

        return x_ffn


# Shuffle operation for Categorization and UnCategorization operations.
def index_reverse(index):
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r

def feature_shuffle(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim-1, index=index)
    return shuffled_x

class ATD_CA(nn.Module):
    r"""
    Adaptive Token Dictionary Cross-Attention.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_tokens (int): Number of tokens in external token dictionary. Default: 64
        reducted_dim (int, optional): Reducted dimension number for query and key matrix. Default: 4
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, input_resolution, num_tokens=64, reducted_dim=10, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_tokens = num_tokens
        self.rc = reducted_dim
        self.qkv_bias = qkv_bias

        self.wq = nn.Linear(dim, reducted_dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, reducted_dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)

        self.scale = nn.Parameter(torch.ones([self.num_tokens]) * 0.5, requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, td, x_size):
        r"""
        Args:
            x: input features with shape of (b, n, c)
            td: token dicitionary with shape of (b, m, c)
            x_size: size of the input x (h, w)
        """
        h, w = x_size
        b, n, c = x.shape
        b, m, c = td.shape
        rc = self.rc

        # Q: b, n, c
        q = self.wq(x)
        # K: b, m, c
        k = self.wk(td)
        # V: b, m, c
        v = self.wv(td)

        # Q @ K^T
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))  # b, n, n_tk
        scale = torch.clamp(self.scale, 0, 1)
        attn = attn * (1 + scale * np.log(self.num_tokens))
        attn = self.softmax(attn)

        # Attn * V
        x = (attn @ v).reshape(b, n, c)

        return x, attn


class AC_MSA(nn.Module):
    r"""
    Adaptive Category-based Multihead Self-Attention.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_tokens (int): Number of tokens in external dictionary. Default: 64
        num_heads (int): Number of attention heads. Default: 4
        category_size (int): Number of tokens in each group for global sparse attention. Default: 128
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, input_resolution, num_tokens=64, num_heads=4, category_size=128, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.category_size = category_size

        # self.wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((1, 1))), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, sim, x_size):
        """
        Args:
            x: input features with shape of (b, HW, c)
            mask: similarity map with shape of (b, HW, m)
            x_size: size of the input x
        """

        H, W = x_size
        b, n, c3 = qkv.shape
        c = c3 // 3
        b, n, m = sim.shape
        gs = min(n, self.category_size)  # group size
        ng = (n + gs - 1) // gs

        # classify features into groups based on similarity map (sim)
        tk_id = torch.argmax(sim, dim=-1, keepdim=False)
        # sort features by type
        # x_sort_values, x_sort_indices = torch.sort(tk_id, dim=-1, stable=False)
        x_sort_values, x_sort_indices = torch.sort(tk_id, dim=-1)
        x_sort_indices_reverse = index_reverse(x_sort_indices)
        shuffled_qkv = feature_shuffle(qkv, x_sort_indices)  # b, n, c3
        pad_n = ng * gs - n
        paded_qkv = torch.cat((shuffled_qkv, torch.flip(shuffled_qkv[:, n - pad_n:n, :], dims=[1])), dim=1)
        y = paded_qkv.reshape(b, -1, gs, c3)

        qkv = y.reshape(b, ng, gs, 3, self.num_heads, c // self.num_heads).permute(3, 0, 1, 4, 2,
                                                                                   5)  # 3, b, ng, nh, gs, c//nh
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Q @ K^T
        attn = (q @ k.transpose(-2, -1))  # b, ng, nh, gs, gs

        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01)).to(qkv.device)).exp()
        attn = attn * logit_scale

        # softmax
        attn = self.softmax(attn)  # b, ng, nh, gs, gs

        # Attn * V
        y = (attn @ v).permute(0, 1, 3, 2, 4).reshape(b, n + pad_n, c)[:, :n, :]

        x = feature_shuffle(y, x_sort_indices_reverse)
        x = self.proj(x)

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

class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x

class ATD_Attn(nn.Module):
    r"""
    ATD Transformer Layer

    Args:
        dim (int): Number of input channels.
        idx (int): Layer index.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        category_size (int): Category size for AC-MSA.
        num_tokens (int): Token number for each token dictionary.
        reducted_dim (int): Reducted dimension number for query and key matrix.
        convffn_kernel_size (int): Convolutional kernel size for ConvFFN.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        is_last (bool): True if this layer is the last of a ATD Block. Default: False
    """

    def __init__(self,
                 dim,
                 idx,
                 input_resolution,
                 num_heads,
                 category_size,
                 num_tokens,
                 reducted_dim,
                 convffn_kernel_size,
                 mlp_ratio,
                 qkv_bias=True,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 is_last=False,
                 ):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.convffn_kernel_size = convffn_kernel_size
        self.num_tokens = num_tokens
        self.softmax = nn.Softmax(dim=-1)
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.reducted_dim = reducted_dim
        self.is_last = is_last

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        if not is_last:
            self.norm3 = nn.InstanceNorm1d(num_tokens, affine=True)
            self.sigma = nn.Parameter(torch.zeros([num_tokens, 1]), requires_grad=True)

        self.wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)


        self.attn_atd = ATD_CA(
            self.dim,
            input_resolution=input_resolution,
            qkv_bias=qkv_bias,
            num_tokens=num_tokens,
            reducted_dim=reducted_dim
        )
        self.attn_aca = AC_MSA(
            self.dim,
            input_resolution=input_resolution,
            num_tokens=num_tokens,
            num_heads=num_heads,
            category_size=category_size,
            qkv_bias=qkv_bias,
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, kernel_size=convffn_kernel_size,
                               act_layer=act_layer)

    def forward(self, x, td, x_size):
        b, c, h, w = x.shape
        # h, w = x_size
        # b, n, c = x.shape
        c3 = 3 * c
        x = rearrange(x, 'b c h w-> b (h w) c')

        shortcut = x
        x = self.norm1(x)
        qkv = self.wqkv(x)

        # ATD_CA
        x_atd, sim_atd = self.attn_atd(x, td, x_size)  # x_atd: (b, n, c)  sim_atd: (b, n,)

        # AC_MSA
        x_aca = self.attn_aca(qkv, sim_atd, x_size)


        x = shortcut + x_atd + x_aca

        # FFN
        x = x + self.convffn(self.norm2(x), x_size)

        b, N, c = x.shape
        b, n, c = td.shape

        # Adaptive Token Refinement
        if not self.is_last:
            mask_soft = self.softmax(self.norm3(sim_atd.transpose(-1, -2)))
            mask_x = x.reshape(b, N, c)
            s = self.sigmoid(self.sigma)
            td = s * td + (1 - s) * torch.einsum('btn,bnc->btc', mask_soft, mask_x)
        x = rearrange(x, 'b (h w ) c-> b c h w', h=h,w=w)
        return x, td


class TransformerBlock(nn.Module):
    def __init__(self, img_size, dim, num_heads, depth, ffn_expansion_factor, bias, LayerNorm_type, window_size=8):
        super(TransformerBlock, self).__init__()
        self.dim = dim
        self.input_resolution = to_2tuple(img_size)
        self.window_size = to_2tuple(window_size)
        self.shift_size = [w // 2 for w in self.window_size]

        num_tokens = 128
        self.blocks = nn.ModuleList()

        for i in range(depth):
            if i % 2 == 0:
                for k, v in self.set_table_index_mask(self.input_resolution).items():
                    self.register_buffer(k, v)

                # window_shift = i % 4 ==2
                window_shift = False
                block = Window_Attn(dim, num_heads, window_shift, ffn_expansion_factor, bias, LayerNorm_type)
            else:
                block = ATD_Attn(dim=dim, idx=i,input_resolution=self.input_resolution, num_heads=num_heads,
                                 category_size=256,
                                 num_tokens=num_tokens,
                                 convffn_kernel_size=5,
                                 reducted_dim=20,
                                 mlp_ratio=ffn_expansion_factor,
                                 qkv_bias=True,
                                 norm_layer=nn.LayerNorm,
                                 is_last=i == depth - 1)

            self.blocks.append(block)

        # Token Dictionary
        self.td = nn.Parameter(torch.randn([num_tokens, dim]), requires_grad=True)

    def set_table_index_mask(self, x_size):
        """
        Two used cases:
        1) At initialization: set the shared buffers.
        2) During forward pass: get the new buffers if the resolution of the input changes
        """
        table_w = get_relative_coords_table_all(self.window_size)
        index_w = get_relative_position_index_simple(self.window_size)
        mask_w = calculate_mask(x_size, self.window_size, self.shift_size)
        return {
            "table_w": table_w,
            "index_w": index_w,
             "mask_w": mask_w}
    
    def get_table_index_mask(self, device=None, input_resolution=None):
        # Used during forward pass
        if input_resolution == self.input_resolution:
            
            return {
                "table_w": self.table_w,         
                "index_w": self.index_w,    
                "mask_w": self.mask_w,  
            }
        else:
            table_index_mask = self.set_table_index_mask(input_resolution)
            for k, v in table_index_mask.items():
                table_index_mask[k] = v.to(device)
            return table_index_mask
    
    def forward(self, x):
        x_size = x.shape[2]
        x_size = to_2tuple(x_size)

        i = 0

        b, c, h, w = x.shape
        td = self.td.repeat([b, 1, 1])
        for blk in self.blocks:
            if i % 2 == 0:
                table_index_mask = self.get_table_index_mask(x.device, x_size)
                # window_shift = i % 4 ==2
                window_shift = False
                if window_shift:
                    mask = table_index_mask["mask_w"]
                else:
                    mask = None
                x = blk(x, table_index_mask["table_w"], table_index_mask["index_w"], mask)
            else:
                x, td = blk(x, td, x_size)
            i += 1
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
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(Restormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        
        self.encoder_level1 = TransformerBlock(img_size, dim=dim, depth=num_blocks[0], num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) 
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = TransformerBlock(img_size//2, dim=int(dim*2**1), depth=num_blocks[1], num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) 
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = TransformerBlock(img_size//4, dim=int(dim*2**2), depth=num_blocks[2], num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)

        # self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        # self.latent = TransformerBlock(img_size//8, dim=int(dim*2**3), depth=num_blocks[3], num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        
        # self.up4_3 = Upsample(int(dim*2**2)) ## From Level 4 to Level 3
        # self.reduce_chan_level3 = nn.Conv2d(int(dim*2**2), int(dim*2**2), kernel_size=1, bias=bias)
        # self.decoder_level3 = TransformerBlock(img_size//4, dim=int(dim*2**2), depth=num_blocks[2], num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = TransformerBlock(img_size//2, dim=int(dim*2**1), depth=num_blocks[1], num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) 
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = TransformerBlock(img_size, dim=int(dim*2**1), depth=num_blocks[0], num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) 
        
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

