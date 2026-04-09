# model/spat_only_lora.py
"""
改动说明（相对原文件）：
  1. LoRALinear (AB)  →  ACBLoRALinear (ACB)
       output = W·x  +  (α/r) · A · C(mod2) · B · x
     C 由第二模态全局特征动态生成，每个样本独立的 (r, r) 矩阵
 
  2. 新增 Mod2Encoder：将第二模态 (B, C2, H, W) 压缩为 (B, cond_dim)
 
  3. inject_lora_into_vit_qkv → inject_acb_lora：注入 ACBLoRALinear
 
  4. SpatOnlyLoRA.forward(x)  →  forward(x, mod2=None)
     - mod2 可选，为 None 时退化为标准 LoRA（C 路径跳过）
 
  5. 修复原 forward 中的 Bug：
     - encoder(x) 返回 logits，不是特征图列表
     - 现在改为调用 encoder.forward_features() 取真正的特征图
     - outputs[0] 取的是 batch 第一个样本（错误）→ 已修正为取 features[-1]
 
不需要改 SpatViT_cls.py。
"""
 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.SpatViT_cls import SpatViT   # ← 保持你原来的 import 路径不变
 
 
# ══════════════════════════════════════════════════════════
# 1. 第二模态编码器
#    输入: (B, C2, H, W)
#    输出: (B, cond_dim)  —— 每个样本一个全局条件向量
# ══════════════════════════════════════════════════════════
 
class Mod2Encoder(nn.Module):
    def __init__(self, in_channels: int, cond_dim: int = 64):
        super().__init__()
        mid = max(cond_dim, 32)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.GELU(),
            nn.Conv2d(mid, cond_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cond_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),   # (B, cond_dim, 1, 1)
            nn.Flatten(),              # (B, cond_dim)
        )
 
    def forward(self, mod2: torch.Tensor) -> torch.Tensor:
        return self.encoder(mod2)
 
 
# ══════════════════════════════════════════════════════════
# 2. ACBLoRALinear：将 C 插入 A 和 B 之间
#
#    标准 LoRA:   output = W·x + (α/r) · A · B · x
#    ACB   LoRA:  output = W·x + (α/r) · A · C(mod2) · B · x
#
#    数据流（以 qkv 为例，out_features = 3 * embed_dim）：
#      x       : (B, N, in_features)
#      x @ B   : (B, N, r)           ← 投影到低秩空间
#      @ C      : (B, N, r)           ← 第二模态调制（per-sample）
#      @ A      : (B, N, out_features) ← 投影回输出空间
# ══════════════════════════════════════════════════════════
 
class ACBLoRALinear(nn.Module):
    def __init__(
        self,
        in_features:  int,
        out_features: int,
        rank:         int   = 8,
        alpha:        float = 1.0,
        cond_dim:     int   = 64,
        bias:         bool  = True,
    ):
        super().__init__()
        self.rank    = rank
        self.scaling = alpha / rank
 
        # 原始权重（冻结）
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        for p in self.linear.parameters():
            p.requires_grad = False
 
        # LoRA 矩阵（可训练）
        # B: (in_features, r)  — 先降维
        # A: (r, out_features) — 再升维
        self.lora_B = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_A = nn.Parameter(torch.zeros(rank, out_features))
 
        # C 生成网络：cond_feat (B, cond_dim) → C (B, r, r)
        # 这是第二模态融入 LoRA 的核心
        self.c_net = nn.Sequential(
            nn.Linear(cond_dim, rank * rank),
            nn.ReLU(),
            nn.Linear(rank * rank, rank * rank),
        )
 
        self._reset_lora_parameters()
 
    def _reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 主路径（预训练权重）
        out = self.linear(x)                            # (B, N, out_features)
 
        # ACB 旁路（只在 _cond_feat 已注册时激活）
        cond_feat = getattr(self, '_cond_feat', None)

        B_, N, D = x.shape

        # Step 1: x → 低秩空间
        xB = x @ self.lora_B   # (B, N, r)

        if cond_feat is not None:
            # ===== 双模态：ACB =====
            C = self.c_net(cond_feat).view(B_, self.rank, self.rank)
            C = torch.tanh(C)

            xBC = torch.einsum('bnr,brs->bns', xB, C)

        else:
            # ===== 单模态：标准 LoRA（关键！！！）=====
            xBC = xB

        # Step 3: 回投
        delta = xBC @ self.lora_A

        # Dropout 保留（单模态也能用）
        delta = F.dropout(delta, p=0.2, training=self.training)

        out = out + self.scaling * delta
 
        return out
 
 
# ══════════════════════════════════════════════════════════
# 3. 注入函数：将 SpatViT 中所有 attn.qkv 替换为 ACBLoRALinear
# ══════════════════════════════════════════════════════════
 
def inject_acb_lora(
    model:    nn.Module,
    rank:     int   = 8,
    alpha:    float = 1.0,
    cond_dim: int   = 64,
) -> nn.Module:
    """
    遍历 model，将所有名称含 'attn.qkv' 的 nn.Linear
    替换为 ACBLoRALinear，并拷贝原始权重。
    """
    # 先收集，避免在遍历中修改迭代器
    # ⚠️ 严格匹配：必须是 blocks.N.attn.qkv 这个确切路径
    #    防止误匹配 dsm_to_weight 等其他含 Linear 的模块
    import re
    _TARGET_PATTERN = re.compile(r'^blocks\.\d+\.attn\.qkv$')
 
    replacements = []
    for full_name, module in model.named_modules():
        # 只匹配 blocks.{数字}.attn.qkv，精确路径
        if not _TARGET_PATTERN.search(full_name):
            continue
        if not isinstance(module, nn.Linear):
            continue  # 已经被替换过，跳过
        parts = full_name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        replacements.append((full_name, parent, parts[-1], module))
 
    for full_name, parent, attr, orig in replacements:
        acb = ACBLoRALinear(
            in_features  = orig.in_features,
            out_features = orig.out_features,
            rank         = rank,
            alpha        = alpha,
            cond_dim     = cond_dim,
            bias         = orig.bias is not None,
        )
        # 拷贝预训练权重
        acb.linear.weight.data.copy_(orig.weight.data)
        if orig.bias is not None:
            acb.linear.bias.data.copy_(orig.bias.data)
 
        setattr(parent, attr, acb)
        print(f"  ✅ {full_name} → ACBLoRALinear")
 
    return model
 
 
# ══════════════════════════════════════════════════════════
# 4. 主模型：SpatOnlyLoRA
# ══════════════════════════════════════════════════════════
 
class SpatOnlyLoRA(nn.Module):
    """
    双模态 SpatViT + ACB LoRA。
 
    forward(x, mod2=None)
      x    : (B, C1, H, W)  — 高光谱 Indian_pines_corrected
      mod2 : (B, C2, H, W)  — 第二模态 HST_10_rep1（可选）
 
    mod2=None 时 C 路径跳过，等价于普通 LoRA（单模态）。
    """
    def __init__(
        self,
        img_size:      int   = 128,
        in_channels:   int   = 200,   # 高光谱波段数
        mod2_channels: int   = 10,    # 第二模态通道数
        patch_size:    int   = 8,
        classes:       int   = 16,
        model_size:    str   = 'base',
        lora_rank:     int   = 8,
        lora_alpha:    float = 1.0,
        cond_dim:      int   = 64,    # 第二模态压缩后的维度
    ):
        super().__init__()
        self.img_size  = img_size
        self.classes   = classes
        self.cond_dim  = cond_dim
 
        if model_size == 'base':
            embed_dim = 768
            depth     = 12
        else:
            raise NotImplementedError("Only base model supported for now")
 
        # ── 第一模态骨干（SpatViT） ──────────────────────────
        self.encoder = SpatViT(
            img_size        = img_size,
            num_classes     = classes,
            in_chans        = in_channels,
            patch_size      = patch_size,
            drop_path_rate  = 0.1,
            out_indices     = [2, 5, 8, 11],
            embed_dim       = embed_dim,
            depth           = depth,
            num_heads       = embed_dim // 64,   # 768//64 = 12 heads
            mlp_ratio       = 4,
            qkv_bias        = True,
            use_abs_pos_emb = True,
            interval        = 3,
            n_points        = 8,
        )
 
        # ── 第二模态编码器 ───────────────────────────────────
        self.mod2_encoder = Mod2Encoder(
            in_channels = mod2_channels,
            cond_dim    = cond_dim,
        )
 
        # ── 注入 ACB LoRA（替换所有 attn.qkv） ──────────────
        print("Injecting ACB LoRA into SpatViT...")
        self.encoder = inject_acb_lora(
            self.encoder,
            rank     = lora_rank,
            alpha    = lora_alpha,
            cond_dim = cond_dim,
        )
 
        # ── 冻结骨干，只解冻 LoRA 参数 ──────────────────────
        self._setup_trainable_params()
 
        # ── 分类头 ──────────────────────────────────────────
        # forward_features 返回 (B, embed_dim, Hp, Wp)
        # 接 GAP → (B, embed_dim) → Linear → (B, classes)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, classes),
        )
 
    def _setup_trainable_params(self):
        """冻结所有参数，只解冻 ACB LoRA 相关层 + 分类头。"""
        for param in self.encoder.parameters():
            param.requires_grad = False
 
        for name, param in self.encoder.named_parameters():
            if any(k in name for k in ['lora_A', 'lora_B', 'c_net']):
                param.requires_grad = True
 
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"  可训练参数: {trainable:,} / {total:,} "
              f"({100 * trainable / total:.2f}%)")
 
    def _set_cond_feat(self, cond_feat):
        """
        在所有 ACBLoRALinear 上注册当前 batch 的条件特征。
        forward 开始时设置，结束后清除。
        """
        for module in self.encoder.modules():
            if isinstance(module, ACBLoRALinear):
                module._cond_feat = cond_feat
 
    def forward(self, x: torch.Tensor, mod2: torch.Tensor = None) -> torch.Tensor:
        """
        参数:
            x    : (B, C1, H, W)  高光谱输入
            mod2 : (B, C2, H, W)  第二模态输入，可为 None
 
        返回:
            logits : (B, classes)
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"输入尺寸 {H}×{W} 与期望 {self.img_size}×{self.img_size} 不符"
 
        # Step 1: 编码第二模态 → 条件向量 (B, cond_dim)
        cond_feat = self.mod2_encoder(mod2) if mod2 is not None else None
 
        # Step 2: 把条件向量注册到每个 ACBLoRALinear
        self._set_cond_feat(cond_feat)
 
        # Step 3: 骨干前向，取特征图
        # forward_features 返回 [原图, f_idx2, f_idx5, f_idx8, f_idx11]
        # 最后一个元素对应最深层 out_index=11
        features = self.encoder.forward_features(x, self.encoder.patch_size)
        feat_map = features[-1]   # (B, embed_dim, Hp, Wp)
 
        # Step 4: 分类头
        logits = self.cls_head(feat_map)   # (B, classes)
 
        # Step 5: 清除条件向量，防止残留影响下次前向
        self._set_cond_feat(None)
 
        return logits
 
 
# ══════════════════════════════════════════════════════════
# 5. 工厂函数（可选，方便外部调用）
# ══════════════════════════════════════════════════════════
 
def build_spat_only_lora(
    img_size:      int   = 128,
    in_channels:   int   = 200,
    mod2_channels: int   = 10,
    patch_size:    int   = 8,
    classes:       int   = 16,
    lora_rank:     int   = 8,
    lora_alpha:    float = 1.0,
    cond_dim:      int   = 64,
) -> SpatOnlyLoRA:
    return SpatOnlyLoRA(
        img_size      = img_size,
        in_channels   = in_channels,
        mod2_channels = mod2_channels,
        patch_size    = patch_size,
        classes       = classes,
        lora_rank     = lora_rank,
        lora_alpha    = lora_alpha,
        cond_dim      = cond_dim,
    )
 
 
# ══════════════════════════════════════════════════════════
# 6. 快速验证（python model/spat_only_lora.py 直接运行）
# ══════════════════════════════════════════════════════════
 
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    model = build_spat_only_lora(
        img_size      = 128,
        in_channels   = 200,   # Indian_pines_corrected: 200 bands
        mod2_channels = 10,    # HST_10_rep1: 10 channels
        classes       = 16,
        lora_rank     = 8,
        lora_alpha    = 1.0,
        cond_dim      = 64,
    ).to(device)
 
    hsi  = torch.randn(2, 200, 128, 128).to(device)
    mod2 = torch.randn(2,  10, 128, 128).to(device)
 
    # 双模态
    out = model(hsi, mod2)
    print(f"✅ 双模态: {hsi.shape} + {mod2.shape} → {out.shape}")
    assert out.shape == (2, 16)
 
    # 单模态（mod2=None，退化为普通 LoRA）
    out_single = model(hsi, mod2=None)
    print(f"✅ 单模态: {hsi.shape} → {out_single.shape}")
    assert out_single.shape == (2, 16)
 
    print("\n可训练参数：")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print(f"  {name:55s} {tuple(p.shape)}")