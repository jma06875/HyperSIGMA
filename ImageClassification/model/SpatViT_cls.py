# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, mmseg, setr, xcit and swin code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/fudan-zvg/SETR
# https://github.com/facebookresearch/xcit/
# https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------
import warnings
import math
import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import drop_path, to_2tuple, trunc_normal_

from mmengine.dist import get_dist_info
from torch.nn.init import constant_, xavier_uniform_

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Import LoRA
try:
    from model.lora import LoRALinear
except ImportError:
    LoRALinear = None
    print("Warning: LoRALinear not found. Using standard Linear.")
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


def get_reference_points(spatial_shapes, device):
    H_, W_ = spatial_shapes[0], spatial_shapes[1]
    ref_y, ref_x = torch.meshgrid(
        torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
        torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
    ref_y = ref_y.reshape(-1)[None] / H_
    ref_x = ref_x.reshape(-1)[None] / W_
    ref = torch.stack((ref_x, ref_y), -1)
    return ref


def deform_inputs_func(x, patch_size):
    B, c, h, w = x.shape
    spatial_shapes = torch.as_tensor([h // patch_size, w // patch_size],
                                    dtype=torch.long, device=x.device)
    reference_points = get_reference_points([h // patch_size, w // patch_size], x.device)
    deform_inputs = [reference_points, spatial_shapes]
    return deform_inputs

class ConditionalLoRA(nn.Module):
    def __init__(self, in_dim, out_dim, r=8, cond_dim=64):
        super().__init__()
        self.r = r
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Shared low-rank matrices
        self.A = nn.Parameter(torch.zeros(out_dim, r))
        self.B = nn.Parameter(torch.zeros(r, in_dim))

        # C generator: from DSM feature → (r, r) matrix per sample
        self.c_net = nn.Sequential(
            nn.Linear(cond_dim, r * r),
            nn.ReLU(),
            nn.Linear(r * r, r * r)
        )

        self.reset_parameters()

    def reset_parameters(self):
        # Initialize A with Kaiming, B as zero (standard LoRA init)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x, cond_feat):
        """
        x: (B, N, in_dim)
        cond_feat: (B, cond_dim) — global DSM feature
        Returns: (B, N, out_dim) delta to add to main linear output
        """
        B_, N, D = x.shape
        assert D == self.in_dim

        # Generate C for each sample: (B, r*r) → (B, r, r)
        C_flat = self.c_net(cond_feat)  # (B, r*r)
        C = C_flat.view(B_, self.r, self.r)  # (B, r, r)

        # Compute: x @ B^T → (B, N, r)
        x_B = torch.matmul(x, self.B.t())  # (B, N, r)

        # Multiply by C: (B, N, r) @ (B, r, r) → use einsum for batched matmul
        x_C = torch.einsum('bnr,brs->bns', x_B, C)  # (B, N, r)

        # Multiply by A: (B, N, r) @ (out_dim, r)^T → (B, N, out_dim)
        delta = torch.matmul(x_C, self.A.t())  # (B, N, out_dim)

        return delta

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SampleAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None, n_points=4,
            use_lora=False, lora_r=8, lora_alpha=16, lora_dropout=0.1,
            # ↓↓↓ 新增参数 ↓↓↓
            use_cond_lora=False,
            cond_dim=64
    ):
        super().__init__()
        self.n_points = n_points
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # >>>>>>>>>> Replace qkv with LoRA if enabled <<<<<<<<<<
        qkv_linear = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)
        if use_lora and LoRALinear is not None:
            self.qkv = LoRALinear(qkv_linear, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        else:
            self.qkv = qkv_linear
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        self.sampling_offsets = nn.Linear(all_head_dim, self.num_heads * n_points * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_cond_lora = use_cond_lora
        if use_cond_lora:
            self.cond_lora_proj = ConditionalLoRA(
                in_dim=all_head_dim,   # 注意：输入是 all_head_dim（=dim）
                out_dim=dim,
                r=lora_r,
                cond_dim=cond_dim
            )
        else:
            self.cond_lora_proj = None

    def forward(self, x, H, W, deform_inputs,dsm_feat=None):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, -1).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        reference_points, input_spatial_shapes = deform_inputs
        sampling_offsets = self.sampling_offsets(q).reshape(
            B, N, self.num_heads, self.n_points, 2).transpose(1, 2)
        _, _, L = q.shape
        q = q.reshape(B, N, self.num_heads, L // self.num_heads).transpose(1, 2)
        offset_normalizer = torch.stack([input_spatial_shapes[1], input_spatial_shapes[0]])
        sampling_locations = reference_points[:, None, :, None, :] \
                             + sampling_offsets / offset_normalizer[None, None, None, None, :]
        sampling_locations = 2 * sampling_locations - 1

        k = k.reshape(B, N, self.num_heads, L // self.num_heads).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, L // self.num_heads).transpose(1, 2)
        k = k.flatten(0,1).transpose(1,2).reshape(B*self.num_heads, L // self.num_heads, input_spatial_shapes[0], input_spatial_shapes[1])
        v = v.flatten(0,1).transpose(1,2).reshape(B*self.num_heads, L // self.num_heads, input_spatial_shapes[0], input_spatial_shapes[1])
        sampling_locations = sampling_locations.flatten(0,1).reshape(B*self.num_heads, N, self.n_points, 2)
        q = q[:,:,:,None,:]

        sampled_k = F.grid_sample(k, sampling_locations, mode='bilinear',
                                  padding_mode='zeros', align_corners=False).reshape(B, self.num_heads, L // self.num_heads, N, self.n_points).permute(0,1,3,4,2)
        sampled_v = F.grid_sample(v, sampling_locations, mode='bilinear',
                                  padding_mode='zeros', align_corners=False).reshape(B, self.num_heads, L // self.num_heads, N, self.n_points).permute(0,1,3,4,2)
        
        attn = (q * sampled_k).sum(-1) * self.scale
        attn = attn.softmax(dim=-1)[:, :, :, :, None]
        x = (attn * sampled_v).sum(-2).transpose(1, 2).reshape(B, N, -1)
        x_main = self.proj(x)
        if self.use_cond_lora and dsm_feat is not None:
            lora_delta = self.cond_lora_proj(x, dsm_feat)  # x: (B, N, all_head_dim)
            x = x_main + lora_delta
        else:
            x = x_main
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None, n_points=4,
            use_lora=False, lora_r=8, lora_alpha=16, lora_dropout=0.1,
            # ↓↓↓ 新增参数 ↓↓↓
            use_cond_lora=False,
            cond_dim=None
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # >>>>>>>>>> Replace qkv with LoRA if enabled <<<<<<<<<<
        qkv_linear = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)
        if use_lora and LoRALinear is not None:
            self.qkv = LoRALinear(qkv_linear, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        else:
            self.qkv = qkv_linear
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_cond_lora = use_cond_lora
        if use_cond_lora:
            self.cond_lora_proj = ConditionalLoRA(
                in_dim=all_head_dim,   # 注意：输入是 all_head_dim（=dim）
                out_dim=dim,
                r=lora_r,
                cond_dim=cond_dim
            )
        else:
            self.cond_lora_proj = None

    def forward(self, x, H, W, rel_pos_bias=None,dsm_feat=None):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x_main = self.proj(x)
        if self.use_cond_lora and dsm_feat is not None:
            lora_delta = self.cond_lora_proj(x, dsm_feat)  # x: (B, N, all_head_dim)
            x = x_main + lora_delta
        else:
            x = x_main
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None, sample=False, restart_regression=True, n_points=None,
                 use_lora=False, lora_r=8, lora_alpha=16, lora_dropout=0.1,use_cond_lora=False, cond_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.sample = sample

        if not sample:
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim,
                use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                
                use_cond_lora=use_cond_lora,
                cond_dim=cond_dim)
        else:
            self.attn = SampleAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim, n_points=n_points,
                use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                
                use_cond_lora=use_cond_lora,
                cond_dim=cond_dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, H, W, deform_inputs,dsm_feat=None):
        if self.gamma_1 is None:
            if not self.sample:
                x = x + self.drop_path(self.attn(self.norm1(x), H, W))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x), H, W, deform_inputs))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            if not self.sample:
                x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), H, W))
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), H, W, deform_inputs))
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)


class HybridEmbed(nn.Module):
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class SpatViT(nn.Module):
    def __init__(self, img_size=224, patch_size=1, in_chans=3, num_classes=80, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=None, init_values=None, use_checkpoint=False, 
                 use_abs_pos_emb=False, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 out_indices=[11], interval=3, pretrained=None, restart_regression=True, n_points=4,
                 use_lora=False, lora_r=8, lora_alpha=16, lora_dropout=0.1,
                 use_cond_lora=False,cond_dim=None):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_cond_lora = use_cond_lora
        self.cond_dim = cond_dim if cond_dim is not None else embed_dim // 4
        self.in_chans = in_chans
        self.out_channels = (3, embed_dim, embed_dim, embed_dim, embed_dim)
        self.patch_size = patch_size
        self.DR = nn.Conv1d(embed_dim*4, 128, kernel_size=1, bias=False)
        self.cls = nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=embed_dim*4, out_features=128),
            nn.Linear(in_features=128, out_features=64),
            nn.Linear(in_features=64, out_features=num_classes)
        )

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches
        self.out_indices = out_indices

        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            self.pos_embed = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.use_rel_pos_bias = use_rel_pos_bias
        self.use_checkpoint = use_checkpoint

        # Pass LoRA args to Blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, sample=((i + 1) % interval != 0), 
                restart_regression=restart_regression, n_points=n_points,
                use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                use_cond_lora=self.use_cond_lora,cond_dim=self.cond_dim)
            for i in range(depth)])
         
        self.interval = interval

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        self.norm = norm_layer(embed_dim)

        # FPN layers (unchanged)
        if patch_size == 16:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                Norm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )
            self.fpn2 = nn.Sequential(nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2))
            self.fpn3 = nn.Identity()
            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            # Simplified: you can restore other patch_size logic if needed
            self.fpn1 = self.fpn2 = self.fpn3 = self.fpn4 = nn.Identity()
        self.dsm_to_weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),        # (B, 1, H, W) -> (B, 1, 1, 1)
            nn.Flatten(),                   # (B, 1)
            nn.Linear(1, embed_dim // 4),   # bottleneck
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim),
            nn.Sigmoid()
        )
        self.apply(self._init_weights)
        self.fix_init_weight()
        self.pretrained = pretrained
        self.use_lora = use_lora

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained):
        # ... (your existing init_weights code unchanged) ...
        # We won't repeat it here for brevity — keep your original implementation
        pass  # <<< Keep your original `init_weights` function body!

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x, patch_size,dsm_feat=None):
        img = [x]
        deform_inputs = deform_inputs_func(x, patch_size)
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        features = []
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
            # 注意：torch.utils.checkpoint 要求所有输入都是 Tensor 或 None
            # dsm_feat 是 (B, cond_dim) 或 None，符合要求
                x = checkpoint.checkpoint(
                    blk, 
                    x, Hp, Wp, deform_inputs, dsm_feat
                    )
            else:
                x = blk(x, Hp, Wp, deform_inputs, dsm_feat=dsm_feat)  # ← 关键：传入 dsm_feat
            if i in self.out_indices:
                features.append(x)
        features = list(map(lambda x: x.permute(0, 2, 1).reshape(B, -1, Hp, Wp), features))
        return img + features
    
    def forward(self, x, dsm=None):
        B, C, H, W = x.shape  
    # 如果提供了 dsm，生成通道调制权重 C
        if dsm is not None:
            dsm_input = dsm.unsqueeze(1)  # (B, 1, H, W)
            channel_weights = self.dsm_to_weight(dsm_input)  # (B, embed_dim)
        else:
            channel_weights = torch.ones(B, self.embed_dim, device=x.device)

        features = self.forward_features(x, self.patch_size)
        feture1 = features[4]
        feture2 = features[3]
        feture3 = features[2]
        feture4 = features[1]

        y1 = F.avg_pool2d(feture1, feture1.size()[2:]).view(B, -1)
        y2 = F.avg_pool2d(feture2, feture2.size()[2:]).view(B, -1)
        y3 = F.avg_pool2d(feture3, feture3.size()[2:]).view(B, -1)
        y4 = F.avg_pool2d(feture4, feture4.size()[2:]).view(B, -1)

        output = torch.cat((y1, y2, y3, y4), 1)

        if dsm is not None:
            channel_weights = self.dsm_to_weight(dsm_input).repeat(1, 4)  # (B, 4*D)
            output = output * channel_weights

        output = self.classifier(output)
        return output

    # 应用 DSM 调制
        
        D = self.embed_dim
        mod_weights = channel_weights.repeat(1, 4)  # (B, 4*D)
        output = output * mod_weights

        output = self.classifier(output)
        return output

# Model builders with LoRA support
def replace_attention_with_lora(module, r=8, lora_alpha=16, lora_dropout=0.1, prefix=""):
    """Replace both 'qkv' and 'proj' Linear layers in Attention with LoRA."""
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and ("qkv" in full_name or "proj" in full_name):
            print(f"🔧 Replacing {full_name} with LoRALinear")
            setattr(module, name, LoRALinear(child, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout))
        else:
            replace_attention_with_lora(child, r, lora_alpha, lora_dropout, full_name)

def spat_vit_b_rvsa(args, inchannels=3, use_lora=False, lora_r=8, lora_alpha=16, lora_dropout=0.1,use_cond_lora=False):
    num_classes = getattr(args, 'num_classes', 16)
    
    backbone = SpatViT(
        img_size=args.image_size,
        in_chans=inchannels,
        num_classes=num_classes,
        patch_size=16,
        drop_path_rate=0.1,
        out_indices=[3, 5, 7, 11],
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        use_checkpoint=(args.use_ckpt == 'True'),
        use_abs_pos_emb=False,
        interval=3,
        use_lora=use_lora,          # Keep original LoRA if needed
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        
        use_cond_lora=use_cond_lora,
        cond_dim=768 // 4,    
    )
    
    if use_lora:
        # ✅ 调用新函数
        replace_attention_with_lora(backbone, r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        for param in backbone.parameters():
            param.requires_grad = False
        for name, param in backbone.named_parameters():
            if "lora_" in name or "classifier" in name:
                param.requires_grad = True
    
    return backbone