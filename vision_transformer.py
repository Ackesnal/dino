# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial

import torch
import torch.nn as nn

from utils import trunc_normal_


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
   
        

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            dim_in,
            dim_hidden=None,
            dim_out=None,
            bias=True,
            drop_path=0.,
            use_conv=False,
            channel_idle=False,
            act_layer=nn.GELU,
            feature_norm="LayerNorm",
            idle_ratio=0.75):
            
        super().__init__()
        
        ######################## ↓↓↓↓↓↓ ########################
        # Hyperparameters
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden or dim_in
        self.dim_out = dim_out or dim_in
        ######################## ↑↑↑↑↑↑ ########################
        
        ######################## ↓↓↓↓↓↓ ########################
        # Self-attention projections
        self.fc1 = nn.Linear(self.dim_in, self.dim_hidden, bias=bias)
        self.fc2 = nn.Linear(self.dim_hidden, self.dim_out, bias=bias)
        self.act = act_layer()
        ######################## ↑↑↑↑↑↑ ########################
        
        ######################## ↓↓↓↓↓↓ ########################
        # Channel-idle
        self.channel_idle = channel_idle
        self.act_channels = int(dim_hidden * (1-idle_ratio))
        ######################## ↑↑↑↑↑↑ ########################
        
        ######################## ↓↓↓↓↓↓ ########################
        self.feature_norm = feature_norm
        if self.feature_norm == "LayerNorm":
            self.norm = nn.LayerNorm(self.dim_in)
        elif self.feature_norm == "BatchNorm":
            self.norm1 = nn.BatchNorm1d(self.dim_in)
            self.norm2 = nn.BatchNorm1d(self.dim_hidden)
        ######################## ↑↑↑↑↑↑ ########################
        
        ######################## ↓↓↓↓↓↓ ########################
        # Drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else None
        ######################## ↑↑↑↑↑↑ ########################
        
    def forward(self, x):
        B, N, C = x.shape
        ######################## ↓↓↓ 2-layer MLP ↓↓↓ ########################
        shortcut = x # B, N, C
        
        # 1st Feature normalization
        if self.feature_norm == "LayerNorm":
            x = self.norm(x)
        elif self.feature_norm == "BatchNorm":
            x = self.norm1(x.transpose(-1,-2)).transpose(-1, -2)
        
        # FFN in
        x = self.fc1(x) # B, N, 4C
        
        # Activation
        if self.channel_idle:
            mask = torch.zeros_like(x, dtype=torch.bool)
            mask[:, :, :self.act_channels] = True
            x = torch.where(mask, self.act(x), x)
        else:
            x = self.act(x)
        
        # 2nd Feature normalization
        if self.feature_norm == "BatchNorm":
            x = self.norm2(x.transpose(-1,-2)).transpose(-1, -2)
            
        # FFN out
        x = self.fc2(x)
         
        # Add DropPath
        x = self.drop_path(x) if self.drop_path is not None else x
        
        x = x + shortcut
        ######################## ↑↑↑ 2-layer MLP ↑↑↑ ########################
        #if x.get_device() == 0:
            #print("x after ffn:", x.std(-1).mean().item(), x.mean().item(), x.max().item(), x.min().item())
        return x
        
    def reparam(self):
        self.eval()
        with torch.no_grad():
            mean = self.norm1.running_mean
            std = torch.sqrt(self.norm1.running_var + self.norm1.eps)
            weight = self.norm1.weight
            bias = self.norm1.bias
            
            fc1_bias = self.fc1(-mean/std*weight+bias)
            fc1_weight = self.fc1.weight / std[None, :] * weight[None, :]
            
            mean = self.norm2.running_mean
            std = torch.sqrt(self.norm2.running_var + self.norm2.eps)
            weight = self.norm2.weight
            bias = self.norm2.bias
            
            fc2_bias = self.fc2(-mean/std*weight+bias)
            fc2_weight = self.fc2.weight / std[None, :] * weight[None, :]
        
        return fc1_bias, fc1_weight, fc2_bias, fc2_weight, self.act_channels



class RePaMlp(nn.Module):
    def __init__(self, 
                 fc1_bias, 
                 fc1_weight, 
                 fc2_bias, 
                 fc2_weight,
                 act_channels, 
                 act_layer):
        super().__init__()
        
        dim = fc1_weight.shape[1]
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, act_channels)
        self.fc3 = nn.Linear(act_channels, dim, bias=False)
        self.act = act_layer()
        
        with torch.no_grad():
            weight1 = fc1_weight[act_channels:, :].T @ fc2_weight[:, act_channels:].T + torch.eye(dim).to(fc1_weight.device)
            weight2 = fc1_weight[:act_channels, :]
            weight3 = fc2_weight[:, :act_channels] 
            bias1 = (fc1_bias[act_channels:].unsqueeze(0) @ fc2_weight[:, act_channels:].T).squeeze() + fc2_bias
            bias2 = fc1_bias[:act_channels]
            
            self.fc1.weight.copy_(weight1.T)
            self.fc1.bias.copy_(bias1)
            self.fc2.weight.copy_(weight2)
            self.fc2.bias.copy_(bias2)
            self.fc3.weight.copy_(weight3)
        
    def forward(self, x):
        with torch.no_grad():
            x = self.fc3(self.act(self.fc2(x))) + self.fc1(x)
            return x
            


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn



class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, feature_norm="LayerNorm", channel_idle=False,
                 idle_ratio=0.75):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        if channel_idle:
            self.mlp = Mlp(dim_in=dim, dim_hidden=mlp_hidden_dim, bias=qkv_bias, act_layer=act_layer, 
                           drop_path=drop_path, feature_norm=feature_norm,
                           channel_idle=channel_idle, idle_ratio=idle_ratio)
        else:
            self.mlp = Mlp(dim_in=dim, dim_hidden=mlp_hidden_dim, bias=qkv_bias, 
                           act_layer=act_layer, drop_path=drop_path)
                           
        self.act_layer = act_layer

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = self.mlp(x)
        return x
        
    def reparam(self):
        fc1_bias, fc1_weight, fc2_bias, fc2_weight, act_channels = self.mlp.reparam()
        del self.mlp
        self.mlp = RePaMlp(fc1_bias, fc1_weight, fc2_bias, fc2_weight, act_channels, self.act_layer)
        return
        


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, feature_norm="LayerNorm", channel_idle=False, idle_ratio=0.75,
                 **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                feature_norm=feature_norm, channel_idle=channel_idle, idle_ratio=idle_ratio)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output
        
    def reparam(self):
        for blk in self.blocks:
            blk.reparam()


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
