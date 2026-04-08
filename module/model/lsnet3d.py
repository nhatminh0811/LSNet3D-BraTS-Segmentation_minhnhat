import torch
import torch.nn as nn
import itertools

class Conv3d_BN(nn.Sequential):
    """Conv2d + BatchNorm wrapper (following official LSNet pattern)"""
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', nn.Conv3d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', nn.BatchNorm3d(b))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps)**0.5
        m = nn.Conv3d(w.size(1) * c.groups, w.size(0), w.shape[2:], 
                     stride=c.stride, padding=c.padding, dilation=c.dilation, 
                     groups=c.groups, device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class Residual(nn.Module):
    """Residual wrapper with optional stochastic depth"""
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class FFN3D(nn.Module):
    """Feed-forward network for 3D"""
    def __init__(self, ed, h):
        super().__init__()
        self.pw1 = Conv3d_BN(ed, h, ks=1)
        self.act = nn.ReLU()
        self.pw2 = Conv3d_BN(h, ed, ks=1, bn_weight_init=0)

    def forward(self, x):
        return self.pw2(self.act(self.pw1(x)))


class LKP3D(nn.Module):
    """Large Kernel Perception - generates dynamic kernels"""
    def __init__(self, dim, lks=5, sks=3, groups=8):
        super().__init__()
        self.cv1 = Conv3d_BN(dim, dim // 2, ks=1)
        self.act = nn.ReLU()
        self.cv2 = Conv3d_BN(dim // 2, dim // 2, ks=lks, pad=(lks - 1) // 2, groups=dim // 2)
        self.cv3 = Conv3d_BN(dim // 2, dim // 2, ks=1)
        self.cv4 = nn.Conv3d(dim // 2, sks ** 3 * dim // groups, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=dim // groups, num_channels=sks ** 3 * dim // groups)
        
        self.sks = sks
        self.groups = groups
        self.dim = dim
        
    def forward(self, x):
        x = self.act(self.cv3(self.cv2(self.act(self.cv1(x)))))
        w = self.norm(self.cv4(x))
        b, _, d, h, width = w.size()
        w = w.view(b, self.dim // self.groups, self.sks ** 3, d, h, width)
        return w


class SKA3D(nn.Module):
    """Small Kernel Aggregation - applies dynamic kernels efficiently"""
    def __init__(self, dim, sks=3, groups=8):
        super().__init__()
        self.dim = dim
        self.sks = sks
        self.groups = groups
        self.pad = (sks - 1) // 2

    def forward(self, x, w):
        # x: (B, dim, D, H, W)
        # w: (B, dim//groups, sks³, D, H, W)
        B, C, D, H, W = x.shape
        G = self.groups
        out = torch.zeros_like(x)
        
        for g in range(G):
            patches_list = []
            x_g = x[:, g*C//G:(g+1)*C//G]  # (B, C/G, D, H, W)
            
            # Extract sks³ neighbors using torch.roll (efficient, cyclic padding)
            for di in range(-self.pad, self.pad + 1):
                for hi in range(-self.pad, self.pad + 1):
                    for wi in range(-self.pad, self.pad + 1):
                        patch = torch.roll(x_g, (-di, -hi, -wi), dims=(2, 3, 4))  # (B, C/G, D, H, W)
                        patches_list.append(patch)
            
            patches = torch.stack(patches_list, dim=2)  # (B, C/G, 27, D, H, W)
            w_g = w[:, g]  # (B, 27, D, H, W)
            w_g_exp = w_g.unsqueeze(1)  # (B, 1, 27, D, H, W)
            out_g = (patches * w_g_exp).sum(dim=2)  # (B, C/G, D, H, W)
            out[:, g*C//G:(g+1)*C//G] = out_g
        
        return out


class LSConv3D(nn.Module):
    """LS Convolution - core LSNet module"""
    def __init__(self, dim):
        super().__init__()
        self.lkp = LKP3D(dim, lks=5, sks=3, groups=8)
        self.ska = SKA3D(dim, sks=3, groups=8)
        self.bn = nn.BatchNorm3d(dim)

    def forward(self, x):
        w = self.lkp(x)
        sk = self.ska(x, w)
        return self.bn(sk) + x


class RepVGGDW3D(nn.Module):
    """RepVGG-style depthwise block for even depths"""
    def __init__(self, ed):
        super().__init__()
        self.conv = Conv3d_BN(ed, ed, ks=3, pad=1, groups=ed)
        self.conv1 = Conv3d_BN(ed, ed, ks=1, groups=ed)
        self.dim = ed
    
    def forward(self, x):
        return self.conv(x) + self.conv1(x) + x
    
    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1.fuse()
        
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        
        # Pad 1x1 to 3x3
        conv1_w = nn.functional.pad(conv1_w, [1,1,1,1,1,1])
        identity = nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, 1, 
                                                device=conv1_w.device), [1,1,1,1,1,1])
        
        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b
        
        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
        return conv


class Attention3D(nn.Module):
    """3D Attention with sparse relative position bias (official LSNet style)"""
    def __init__(self, dim, key_dim, num_heads=8, attn_ratio=4, resolution=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        
        h = self.dh + nh_kd * 2
        self.qkv = Conv3d_BN(dim, h, ks=1)
        self.proj = nn.Sequential(nn.ReLU(), Conv3d_BN(self.dh, dim, bn_weight_init=0))
        self.dw = Conv3d_BN(nh_kd, nh_kd, 3, 1, 1, groups=nh_kd)
        
        # Sparse position bias: only store unique offsets
        points = list(itertools.product(range(resolution), range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]), abs(p1[2] - p2[2]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        
        self.attention_biases = nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs', torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):
        B, _, D, H, W = x.shape
        N = D * H * W
        
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, -1, D, H, W).split([self.nh_kd, self.nh_kd, self.dh], dim=1)
        q = self.dw(q)
        
        q = q.view(B, self.num_heads, -1, N)
        k = k.view(B, self.num_heads, -1, N)
        v = v.view(B, self.num_heads, -1, N)
        
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn + (self.attention_biases[:, self.attention_bias_idxs] 
                       if self.training else self.ab)
        attn = attn.softmax(dim=-1)
        
        x = (v @ attn.transpose(-2, -1)).reshape(B, -1, D, H, W)
        x = self.proj(x)
        return x


class Block3D(nn.Module):
    """LSNet block: even depth → RepVGG, odd depth → LSConv or Attention"""
    def __init__(self, ed, kd, nh=8, ar=4, resolution=8, stage=-1, depth=-1):
        super().__init__()
        
        if depth % 2 == 0:
            # Even: RepVGG depthwise
            self.mixer = RepVGGDW3D(ed)
            self.se = nn.Identity()  # Could add SE here
        else:
            # Odd: LSConv or Attention
            if stage == 3:  # Last stage only
                self.mixer = Attention3D(ed, kd, nh, ar, resolution=resolution)
            else:
                self.mixer = LSConv3D(ed)
            self.se = nn.Identity()
        
        self.ffn = Residual(FFN3D(ed, int(ed * 2)))

    def forward(self, x):
        return self.ffn(self.se(self.mixer(x)))


class LSNet3D(nn.Module):
    """3D LSNet for volumetric segmentation"""
    def __init__(self, in_chans=4, embed_dim=[64, 128, 192, 256], 
                 key_dim=[16, 16, 16, 16], depth=[1, 2, 3, 4], 
                 num_heads=[4, 4, 4, 4], patch_size=2):
        super().__init__()
        
        # Patch embedding (3 stride-2 convs = 8x downsampling)
        self.patch_embed = nn.Sequential(
            Conv3d_BN(in_chans, embed_dim[0] // 4, 3, 2, 1), nn.ReLU(),
            Conv3d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1), nn.ReLU(),
            Conv3d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1)
        )
        
        resolution = 8  # 64 / 8
        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]
        
        self.blocks1 = nn.Sequential()
        self.blocks2 = nn.Sequential()
        self.blocks3 = nn.Sequential()
        self.blocks4 = nn.Sequential()
        blocks = [self.blocks1, self.blocks2, self.blocks3, self.blocks4]
        
        for i, (ed, kd, dpth, nh, ar) in enumerate(zip(embed_dim, key_dim, depth, num_heads, attn_ratio)):
            for d in range(dpth):
                blocks[i].append(Block3D(ed, kd, nh, ar, resolution, stage=i, depth=d))
            
            if i != len(depth) - 1:
                blk = blocks[i+1]
                resolution = (resolution - 1) // 2 + 1  # stride-2
                blk.append(Conv3d_BN(embed_dim[i], embed_dim[i], ks=3, stride=2, pad=1, groups=embed_dim[i]))
                blk.append(Conv3d_BN(embed_dim[i], embed_dim[i+1], ks=1, stride=1, pad=0))
    
    def forward(self, x):
        x = self.patch_embed(x)
        outs = []
        
        x = self.blocks1(x)
        outs.append(x)
        
        x = self.blocks2(x)
        outs.append(x)
                
        x = self.blocks3(x)
        outs.append(x)
        
        x = self.blocks4(x)
        outs.append(x)
        
        return outs
