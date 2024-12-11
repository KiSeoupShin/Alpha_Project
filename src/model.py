import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Conv Layer 1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=stride, padding=1, bias=True
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Conv Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=1, padding=1, bias=True
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
    
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=(1, 1), stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out

class ResNetBigger(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.5, linear_layer_size=192, filter_sizes=[64, 32, 16, 16]):
        super(ResNetBigger, self).__init__()
        print(f"training with dropout={dropout_rate}")
        # Initial input conv
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=(3, 3),
            stride=1, padding=1, bias=False
        )

        self.bn1 = nn.BatchNorm2d(64)

        self.linear_layer_size = linear_layer_size

        self.filter_sizes = filter_sizes

        self.block1 = self._create_block(64, filter_sizes[0], stride=1)
        self.block2 = self._create_block(filter_sizes[0], filter_sizes[1], stride=2)
        self.block3 = self._create_block(filter_sizes[1], filter_sizes[2], stride=2)
        self.block4 = self._create_block(filter_sizes[2], filter_sizes[3], stride=2)
        self.bn2 = nn.BatchNorm1d(linear_layer_size)
        self.bn3 = nn.BatchNorm1d(32)
        self.linear1 = nn.Linear(linear_layer_size, 32)
        self.linear2 = nn.Linear(32, num_classes)

        self.dropout = nn.Dropout(dropout_rate)

        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = np.inf

    # A block is just two residual blocks for ResNet18
    def _create_block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels, 1)
        )

    def forward(self, x):
        # Output of one layer becomes input to the next
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = nn.AvgPool2d(4)(out)
        out = out.view(out.size(0), -1)
        out = self.bn2(out)
        out = self.dropout(out)
        out = self.linear1(out)
        out = self.bn3(out)
        out = self.dropout(out)
        out = F.relu(out)
        out = self.linear2(out)
        out = torch.sigmoid(out)
        return out

    def set_device(self, device):
        for b in [self.block1, self.block2, self.block3, self.block4]:
            b.to(device)
        self.to(device)


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels, height, width, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pos_h = torch.arange(height).unsqueeze(1).expand(height, width)
        pos_w = torch.arange(width).unsqueeze(0).expand(height, width)
        
        pe = torch.zeros(1, channels, height, width)
        div_term = torch.exp(torch.arange(0, channels, 2).float() * -(np.log(10000.0) / channels))
        
        # Encoding for height dimension
        pe[0, 0::2, :, :] = torch.sin(pos_h.unsqueeze(0) * div_term.view(-1, 1, 1))
        pe[0, 1::2, :, :] = torch.cos(pos_h.unsqueeze(0) * div_term.view(-1, 1, 1))
        
        # Encoding for width (time) dimension
        if channels > 2:
            pe[0, 2::2, :, :] = torch.sin(pos_w.unsqueeze(0) * div_term.view(-1, 1, 1))
            pe[0, 3::2, :, :] = torch.cos(pos_w.unsqueeze(0) * div_term.view(-1, 1, 1))
            
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        return self.dropout(x)

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 is_pos_encoding=True, height=128, width=160,
                 is_backbone=True,
                 ):
        super().__init__()

        self.is_backbone = is_backbone
        self.is_pos_encoding = is_pos_encoding
        if is_pos_encoding:
            self.pos_encoding = PositionalEncoding2D(channels=in_chans, height=height, width=width, dropout=0.1)

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        if self.is_pos_encoding:
            x = self.pos_encoding(x)

        if self.is_backbone:
            outputs = []
            for i in range(0, x.shape[-1] - 32 + 1, 16):
                sliced_x = x[..., i:i + 32]
                output = self.forward_features(sliced_x)
                outputs.append(output)
            x = torch.stack(outputs, dim=-2)
            out = self.head(x)
            out = out.max(dim=-2).values
            return x, out
        else:
            x = self.forward_features(x)
            x = self.head(x)
        return x, None

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Attention(nn.Module):
    def __init__(self, dim, num_classes, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 클래스 예측을 위한 head 추가
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        # 클래스 예측
        out = self.head(out)
        out = out.max(dim=1).values
        return out


class MelClassifier(nn.Module):
    def __init__(self,
                 backbone_parameters,
                 attention_parameters,
                 ):
        super().__init__()
        self.backbone = convnext_small(**backbone_parameters)
        self.attention = Attention(**attention_parameters)

    def forward(self, x):
        x, out = self.backbone(x)
        x = self.attention(x)
        return x, out


def convnext_tiny(**kwargs):
    return ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)

def convnext_small(**kwargs):
    return ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)

def convnext_base(**kwargs):
    return ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)

def convnext_large(**kwargs):
    return ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)

def convnext_xlarge(**kwargs):
    return ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)