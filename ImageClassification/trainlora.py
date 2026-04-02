# train.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # 确保能 import model

from model.SpatViT_cls import spat_vit_b_rvsa

# 模拟 argparse 参数（实际项目中应从命令行或 config 读取）
class Args:
    image_size = 224
    use_ckpt = False  # 注意：这里应该是布尔值，不是字符串 'False'

args = Args()

# 创建带 LoRA 的模型
model = spat_vit_b_rvsa(
    args,
    inchannels=3,
    use_lora=True,      # 启用 LoRA
    lora_r=8,           # LoRA 秩
    lora_alpha=16       # LoRA 缩放因子
)
# 🔒 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 🔓 解冻 LoRA 参数（lora_A 和 lora_B）
for name, param in model.named_parameters():
    if "lora_" in name:
        param.requires_grad = True
        print(f"✅ Trainable: {name}")

trainable_params = [p for p in model.parameters() if p.requires_grad]
total_trainable = sum(p.numel() for p in trainable_params)
print(f"Trainable parameters: {total_trainable:,}")  # 例如：24,576

import torch.optim as optim

optimizer = optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)
print("PASS >>>>>>>>>>>>>>>>>>>>>>>>>>>> ")