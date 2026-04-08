import torch
import torch.nn as nn
from .lsnet3d import LSNet3D, Conv3d_BN


class ChannelAttention3D(nn.Module):
    """3D Channel Attention Module - squeeze spatial dims, emphasize important channels"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction, channels, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention3D(nn.Module):
    """3D Spatial Attention Module - emphasize important spatial regions (ET regions)"""
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return x * self.sigmoid(out)


class CBAM3D(nn.Module):
    """3D Convolutional Block Attention Module - combines channel and spatial attention"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention3D(channels, reduction)
        self.spatial_attention = SpatialAttention3D(channels, kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class BraTS_Segmentation_Decoder3D(nn.Module):
    """Decoder for BraTS: upsamples from 8³ → 64³ with spatial attention for ET focus"""
    def __init__(self, embed_dim=[64, 128, 192, 256], num_classes=4):
        super().__init__()
        
        # Skip connections from encoder (reverse order)
        self.skip_dim = list(reversed(embed_dim))  # [256, 192, 128, 64]
        
        # Upsample stages: 1³ → 2³ → 4³ → 8³ → 16³ → 32³ → 64³
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            Conv3d_BN(self.skip_dim[0], self.skip_dim[1], ks=1),
        )
        self.attn1 = CBAM3D(self.skip_dim[1])
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            Conv3d_BN(self.skip_dim[1] * 2, self.skip_dim[2], ks=1),
        )
        self.attn2 = CBAM3D(self.skip_dim[2])
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            Conv3d_BN(self.skip_dim[2] * 2, self.skip_dim[3], ks=1),
        )
        self.attn3 = CBAM3D(self.skip_dim[3])
        
        # Additional upsample stages to reach 64³
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            Conv3d_BN(self.skip_dim[3] * 2, 64, ks=1),
        )
        self.attn4 = CBAM3D(64)
        
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            Conv3d_BN(64, 32, ks=1),
        )
        self.attn5 = CBAM3D(32)
        
        self.up6 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            Conv3d_BN(32, 16, ks=1),
        )
        self.attn6 = CBAM3D(16)
        
        # Final segmentation head: 64³ → 4 classes
        self.seg_head = nn.Sequential(
            Conv3d_BN(16, 16, ks=3, pad=1),
            nn.ReLU(),
            nn.Conv3d(16, num_classes, kernel_size=1)
        )
        
        # Multi-scale auxiliary classifiers for hierarchical supervision (LSNet "see large focus small")
        # aux_head_3: From 8³ level (large features, coarse ET detection)
        self.aux_head_3 = nn.Sequential(
            Conv3d_BN(self.skip_dim[3], 32, ks=3, pad=1),
            nn.ReLU(),
            nn.Conv3d(32, num_classes, kernel_size=1)
        )
        
        # aux_head_4: From 16³ level (intermediate features)
        self.aux_head_4 = nn.Sequential(
            Conv3d_BN(64, 24, ks=3, pad=1),
            nn.ReLU(),
            nn.Conv3d(24, num_classes, kernel_size=1)
        )
        
        # aux_head_5: From 32³ level (fine features, small ET emphasis)
        self.aux_head_5 = nn.Sequential(
            Conv3d_BN(32, 16, ks=3, pad=1),
            nn.ReLU(),
            nn.Conv3d(16, num_classes, kernel_size=1)
        )

    def forward(self, encoder_outs):
        """
        encoder_outs: list of 4 outputs from encoder stages
        [outs1(B, 64, 8, 8, 8), outs2(B, 128, 4, 4, 4), 
         outs3(B, 192, 2, 2, 2), outs4(B, 256, 1, 1, 1)]
        
        Returns:
        - Mode 'train' (training): dict with multi-scale outputs for hierarchical supervision
          {'main': (B, 4, 64, 64, 64), 'aux_8': (B, 4, 8, 8, 8), 
           'aux_16': (B, 4, 16, 16, 16), 'aux_32': (B, 4, 32, 32, 32)}
        - Mode 'test' (inference): (B, 4, 64, 64, 64) - only main output
        """
        out4 = encoder_outs[3]  # (B, 256, 1, 1, 1)
        out3 = encoder_outs[2]  # (B, 192, 2, 2, 2)
        out2 = encoder_outs[1]  # (B, 128, 4, 4, 4)
        out1 = encoder_outs[0]  # (B, 64, 8, 8, 8)
        
        # Upsample from 1³ → 2³ + attention
        x = self.up1(out4)  # (B, 192, 2, 2, 2)
        x = self.attn1(x)  # Channel+spatial attention for feature refinement
        x = torch.cat([x, out3], dim=1)  # (B, 384, 2, 2, 2)
        
        # Upsample from 2³ → 4³ + attention
        x = self.up2(x)  # (B, 192, 4, 4, 4)
        x = self.attn2(x)  # Focus on important spatial regions
        x = torch.cat([x, out2], dim=1)  # (B, 320, 4, 4, 4)
        
        # Upsample from 4³ → 8³ + attention (coarse ET detection - aux output 1)
        x = self.up3(x)  # (B, 256, 8, 8, 8)
        x = self.attn3(x)  # Enhanced for ET detection at 8³ resolution
        aux_out_8 = self.aux_head_3(x)  # Auxiliary output at coarse scale
        x = torch.cat([x, out1], dim=1)  # (B, 320, 8, 8, 8)
        
        # Upsample from 8³ → 16³ + attention (intermediate - aux output 2)
        x = self.up4(x)  # (B, 64, 16, 16, 16)
        x = self.attn4(x)  # Attention at intermediate resolution
        aux_out_16 = self.aux_head_4(x)  # Auxiliary output at intermediate scale
        
        # Upsample from 16³ → 32³ + attention (fine - aux output 3)
        x = self.up5(x)  # (B, 32, 32, 32, 32)
        x = self.attn5(x)  # Pre-final attention
        aux_out_32 = self.aux_head_5(x)  # Auxiliary output at fine scale
        
        # Upsample from 32³ → 64³ + attention (final high-res)
        x = self.up6(x)  # (B, 16, 64, 64, 64)
        x = self.attn6(x)  # Final spatial attention for precise boundary
        
        # Final segmentation head: (B, 16, 64, 64, 64) → (B, 4, 64, 64, 64)
        logits = self.seg_head(x)  # (B, 4, 64, 64, 64)
        
        # Return multi-scale outputs for hierarchical supervision during training
        return {
            'main': logits,
            'aux_8': aux_out_8,      # Coarse scale (B, 4, 8, 8, 8)
            'aux_16': aux_out_16,    # Intermediate scale (B, 4, 16, 16, 16)
            'aux_32': aux_out_32     # Fine scale (B, 4, 32, 32, 32)
        }


class LSNet3D_Seg(nn.Module):
    """Complete LSNet3D segmentation model for BraTS"""
    def __init__(self, in_chans=4, num_classes=4, embed_dim=[64, 128, 192, 256],
                 key_dim=[16, 16, 16, 16], depth=[1, 2, 3, 4], num_heads=[4, 4, 4, 4]):
        super().__init__()
        
        self.backbone = LSNet3D(in_chans=in_chans, embed_dim=embed_dim, 
                               key_dim=key_dim, depth=depth, num_heads=num_heads)
        self.decoder = BraTS_Segmentation_Decoder3D(embed_dim=embed_dim, num_classes=num_classes)
    
    def forward(self, x):
        """
        x: input volume (B, 4, 64, 64, 64)
        returns: segmentation logits (B, 4, 8, 8, 8)
        """
        encoder_outs = self.backbone(x)
        logits = self.decoder(encoder_outs)
        return logits
