# cls_spat.py  —— 修复版（支持 batch_size=32，不再 OOM Killed）
#
# 核心改动：
#   1. 删除 create_patches 全量提取，改为 Dataset.__getitem__ 里实时切 patch
#   2. 修复 num_heads = embed_dim // 4 → 12（ViT-Base 标准配置）
#   3. img_size=16, patch_size=4（整除，序列长度合理）
#   4. SpatOnlyLoRA 接口对齐：forward(hsi, mod2)，mod2 形状 (B,1,H,W)

import os
import numpy as np
import torch
import torch.nn as nn
import h5py
import scipy.io as sio
import psutil

from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader

from model.spat_only_lora import SpatOnlyLoRA
from model import split_data, utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("RAM before loading data:", psutil.virtual_memory().percent, "%")

# ══════════════════════════════════════════════════════════
# 超参数（集中管理）
# ══════════════════════════════════════════════════════════
IMG_SIZE       = 8     # patch 空间尺寸，必须能被 patch_size 整除
PATCH_SIZE     = 4       # ViT patch size，序列长度 = (16/4)² = 16
PCA_COMPONENTS = 20      # 高光谱 PCA 降维
BATCH_SIZE     = 32
MAX_EPOCH      = 100
LR             = 2e-4
LORA_RANK      = 8
LORA_ALPHA     = 16
COND_DIM       = 64      # 第二模态压缩维度
TRAIN_RATIO    = 0.05
VAL_RATIO      = 0.05   
TEST_RATIO     = 0.90
EVAL_INTERVAL  = 10
PATIENCE       = 10
USE_DSM = True
PATH_WEIGHT = "weights/"
PATH_RESULT = "result/"
os.makedirs(PATH_WEIGHT, exist_ok=True)
os.makedirs(PATH_RESULT, exist_ok=True)

DATA_PATH = '/root/projects/HyperSIGMA/ImageClassification/data/HST_10_rep1.mat'


# ══════════════════════════════════════════════════════════
# 1. 数据加载（只加载原始矩阵，不提取 patch）
# ══════════════════════════════════════════════════════════

def load_data(data_path):
    with h5py.File(data_path, 'r') as mat:
        data_HS_HR = np.array(mat['data_HS_HR']).astype(np.float32)  # (C, W, H) MATLAB 顺序
        TR  = np.array(mat['TR']).astype(np.int64)
        TE  = np.array(mat['TE']).astype(np.int64)
        DSM = np.array(mat['DSM']).astype(np.float32)

    # 转置为 (H, W, C) / (H, W)
    data_cube = data_HS_HR.transpose(2, 1, 0)   # (H, W, C)
    TR  = TR.T                                   # (H, W)
    TE  = TE.T
    DSM = DSM.T

    g_truth = TR + TE   # TR 和 TE 无重叠

    print(f"data_cube : {data_cube.shape}")
    print(f"g_truth   : {g_truth.shape}, unique={np.unique(g_truth)}")
    print(f"DSM       : {DSM.shape}")
    return data_cube, g_truth, DSM


def hybrid_spatial_split(gt, train_ratio=0.1, block_size=80, random_seed=42):
    np.random.seed(random_seed)
    H, W = gt.shape

    train_mask = np.zeros_like(gt, dtype=bool)
    test_mask  = np.zeros_like(gt, dtype=bool)

    # ✅ block 划分
    blocks = [(i, j)
              for i in range(0, H, block_size)
              for j in range(0, W, block_size)]
    np.random.shuffle(blocks)

    n_train_blk = int(len(blocks) * train_ratio)

    for k, (i, j) in enumerate(blocks):
        ie, je = min(i + block_size, H), min(j + block_size, W)
        if k < n_train_blk:
            train_mask[i:ie, j:je] = gt[i:ie, j:je] > 0
        else:
            test_mask[i:ie, j:je]  = gt[i:ie, j:je] > 0

    # ✅ 防止遗漏
    leftover = (gt > 0) & (~train_mask) & (~test_mask)
    test_mask |= leftover

    # 🔥 关键：返回“绝对索引”
    flat_indices = np.where(gt.flatten() > 0)[0]

    train_mask_flat = train_mask.flatten()
    test_mask_flat  = test_mask.flatten()

    train_index = flat_indices[train_mask_flat[flat_indices]]
    test_index  = flat_indices[test_mask_flat[flat_indices]]

    return train_index.astype(int), np.array([], dtype=int), test_index.astype(int)


# ══════════════════════════════════════════════════════════
# 2. Dataset：实时切 patch（核心内存优化）
#
#    不再预先提取所有 patch 存入内存，
#    而是在 __getitem__ 里现切，内存占用降低 10-100x
# ══════════════════════════════════════════════════════════

class OnTheFlyHSIDataset(Dataset):
    """
    参数：
        hs_data  : (H, W, C) numpy float32，已做 PCA
        dsm_data : (H, W)    numpy float32，已归一化
        gt       : (H, W)    numpy int，标签（0=背景）
        indices  : 1D array，展平后的绝对像素索引（只含非零标签）
        window   : patch 空间尺寸（IMG_SIZE）
        is_train : 是否做数据增强
    """
    def __init__(self, hs_data, dsm_data, gt, indices,
                 window=16, is_train=False,training=True,
                 aug_noise_hsi=0.02, aug_noise_dsm=0.2):
        super().__init__()
        self.hs   = hs_data       # (H, W, C) — 只存引用，不复制
        self.dsm  = dsm_data      # (H, W)
        self.gt   = gt            # (H, W)
        self.indices   = indices
        self.window    = window
        self.pad       = window // 2
        self.is_train  = is_train
        self.noise_hsi = aug_noise_hsi
        self.noise_dsm = aug_noise_dsm
        self.training = training
        H, W, C = hs_data.shape
        # 预先 pad，pad 后的数组仍共享内存基础数据
        # pad = window//2，切片 [ip : ip+window] 恰好是 window 个像素
        self.hs_pad  = np.pad(
            hs_data,
            ((self.pad, self.pad), (self.pad, self.pad), (0, 0)),
            mode='reflect'
        ) 

        self.dsm_pad = np.pad(
            dsm_data,
            ((self.pad, self.pad), (self.pad, self.pad)),
            mode='reflect'
        )
        self.H, self.W = H, W
        # pad 大小：确保切出来恰好是 window × window
        # 切法：padded[ip : ip+window, jp : jp+window]
        # 所以只需要在左/上方向 pad = window//2，右/下不需要额外 pad
        # 但为了对称，两侧都 pad window//2，切片从 ip 开始取 window 个
        self.pad = window // 2

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        flat_idx = self.indices[idx]
        i, j = divmod(flat_idx, self.W)

        ip = i + self.pad
        jp = j + self.pad

        half = self.window // 2

        hs_patch = self.hs_pad[
            ip - half : ip - half + self.window,
            jp - half : jp - half + self.window,
        :
        ]


        if self.training:
            noise = np.random.normal(0, 0.01, hs_patch.shape)
            hs_patch = hs_patch + noise



        dsm_patch = self.dsm_pad[
            ip - half : ip - half + self.window,
            jp - half : jp - half + self.window
        ]

        if not USE_DSM:
            dsm_patch = np.zeros_like(dsm_patch)
        dsm_patch = (dsm_patch - dsm_patch.mean()) / (dsm_patch.std() + 1e-6)

        
        hs_t = torch.from_numpy(
            np.ascontiguousarray(hs_patch.transpose(2, 0, 1))
        ).float()

        dsm_t = torch.from_numpy(
            np.ascontiguousarray(dsm_patch[None])
        ).float()

        label = int(self.gt.flat[flat_idx]) - 1
        label_t = torch.tensor(label, dtype=torch.long)

        if self.is_train:
            hs_t  = hs_t  + torch.randn_like(hs_t)  * self.noise_hsi
            dsm_t = dsm_t + torch.randn_like(dsm_t) * 0.05

        return hs_t, dsm_t, label_t

# ══════════════════════════════════════════════════════════
# 3. 主流程
# ══════════════════════════════════════════════════════════

# --- 加载原始数据 ---
data_cube, g_truth, dsm_raw = load_data(DATA_PATH)
H, W, C_orig = data_cube.shape
num_classes = int(g_truth.max())
print(f"num_classes = {num_classes}")

# --- PCA 降维（只做一次，结果是 (H, W, PCA_COMPONENTS)） ---
data_pca, pca_model = split_data.apply_PCA(data_cube, num_components=PCA_COMPONENTS)
print(f"data after PCA: {data_pca.shape}")   # (H, W, 30)

# --- 归一化 DSM ---
dsm_norm = (dsm_raw - dsm_raw.min()) / (dsm_raw.max() - dsm_raw.min() + 1e-8)

# --- 划分索引 ---
gt_flat = g_truth.reshape(-1)
nonzero_indices = np.where(gt_flat > 0)[0]
nonzero_labels  = gt_flat[nonzero_indices]

gt_flat = g_truth.reshape(-1)
indices = np.where(gt_flat > 0)[0]


from sklearn.model_selection import train_test_split

gt_flat = g_truth.reshape(-1)
indices = np.where(gt_flat > 0)[0]
labels  = gt_flat[indices]

train_index, test_index = train_test_split(
    indices,
    train_size=TRAIN_RATIO,
    stratify=labels,   # 🔥 关键：保证每个类都有
    random_state=42
)

# ===== 从 train 划 val =====
val_portion = 0.2
num_train = len(train_index)

np.random.seed(42)
perm = np.random.permutation(num_train)

val_size = int(num_train * val_portion)

val_index = train_index[perm[:val_size]]
train_index = train_index[perm[val_size:]]

# ================== 🔥 加在这里（唯一必须修改） ==================

pad = IMG_SIZE // 2   # 和 Dataset 保持一致
H, W = g_truth.shape

coords = np.array(np.where(g_truth > 0)).T  # (N,2)

def filter_safe(indices):
    selected_coords = coords[np.isin(np.where(g_truth.flatten() > 0)[0], indices)]

    mask = (
        (selected_coords[:, 0] >= pad) &
        (selected_coords[:, 0] < H - pad) &
        (selected_coords[:, 1] >= pad) &
        (selected_coords[:, 1] < W - pad)
    )

    return indices[mask]


# ===============================================================


train_label = g_truth.flat[train_index].astype(int) - 1
test_label  = g_truth.flat[test_index].astype(int)  - 1
val_label = g_truth.flat[val_index].astype(int) - 1
print(f"train={len(train_index)}, val={len(val_index)}, test={len(test_index)}")
assert np.min(train_label) >= 0 and np.max(train_label) < num_classes
assert np.min(test_label)  >= 0 and np.max(test_label)  < num_classes

# --- Dataset & DataLoader ---
train_dataset = OnTheFlyHSIDataset(
    data_pca, dsm_norm, g_truth, train_index,training=True,
    window=IMG_SIZE, is_train=True)

val_dataset = OnTheFlyHSIDataset(
    data_pca, dsm_norm, g_truth, val_index,training=False,
    window=IMG_SIZE, is_train=False)

test_dataset = OnTheFlyHSIDataset(
    data_pca, dsm_norm, g_truth, test_index,training=False,
    window=IMG_SIZE, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=4, pin_memory=True,)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=4, pin_memory=True)

print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

# ══════════════════════════════════════════════════════════
# 4. 模型
# ══════════════════════════════════════════════════════════

model = SpatOnlyLoRA(
    img_size      = IMG_SIZE,        # 16
    in_channels   = PCA_COMPONENTS,  # 30
    mod2_channels = 1,               # DSM 是单通道
    patch_size    = PATCH_SIZE,      # 4，序列长度 = (16/4)² = 16
    classes       = num_classes,
    lora_rank     = LORA_RANK,
    lora_alpha    = LORA_ALPHA,
    cond_dim      = COND_DIM,
).to('cpu')

# --- 加载预训练权重 ---
script_dir = os.path.dirname(os.path.abspath(__file__))
ckpt_path  = os.path.join(script_dir, "spat-base.pth")

if False:
    print(f"Loading pretrained weights from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    pretrained = ckpt.get('model', ckpt)

    model_params = model.state_dict()
    first_key = next(iter(pretrained.keys()))
    needs_prefix = ('encoder.' not in first_key and
                    any('encoder.' in k for k in model_params))

    loaded, skipped = 0, 0
    for k, v in pretrained.items():
        if any(s in k for s in ['pos_embed', 'patch_embed.proj', 'head', 'classifier']):
            skipped += 1
            continue
        target_k = ('encoder.' + k) if needs_prefix else k
        if target_k in model_params and v.shape == model_params[target_k].shape:
            model_params[target_k] = v
            loaded += 1
        else:
            skipped += 1

    model.load_state_dict(model_params, strict=False)
    print(f"Loaded {loaded} layers, skipped {skipped}")
else:
    print(f"No checkpoint found at {ckpt_path}, training from scratch.")

model = model.to(device)

# --- 参数冻结：只训练 LoRA 相关 + 分类头 ---
for p in model.parameters():
    p.requires_grad = False
for name, p in model.named_parameters():
    if USE_DSM:
        if any(k in name for k in ['lora_A', 'lora_B', 'c_net',
                                'mod2_encoder', 'cls_head']):
            p.requires_grad = True
    else:
        if any(k in name for k in ['lora_A', 'lora_B', 'cls_head']):
            p.requires_grad = True
        

trainable = [(n, p.shape) for n, p in model.named_parameters() if p.requires_grad]
print(f"Trainable params: {len(trainable)}")
for n, s in trainable:
    print(f"  {n:60s} {s}")

# ══════════════════════════════════════════════════════════
# 5. 优化器 & 损失
# ══════════════════════════════════════════════════════════
# ⭐ 参数分组（关键）
def get_param_groups(model):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or "norm" in name.lower()
        ):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": 5e-4},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# ⭐ optimizer
optimizer = torch.optim.AdamW(
    get_param_groups(model),
    lr=LR
)


# ⭐ scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=MAX_EPOCH,
    eta_min=1e-6
)


# ⭐ class weight
# ===== FIX START =====

unique_classes = np.unique(train_label)

class_weights = np.ones(num_classes, dtype=np.float32)

if len(unique_classes) > 0:
    weights_partial = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=train_label
    )
    class_weights[unique_classes] = weights_partial

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

print("train_label unique:", np.unique(train_label))
# ===== FIX END =====

criterion = nn.CrossEntropyLoss(weight=class_weights)
# ══════════════════════════════════════════════════════════
# 6. 训练循环
# ══════════════════════════════════════════════════════════

best_val_oa      = 0.0
patience_counter = 0

for epoch in range(MAX_EPOCH):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for hsi, dsm, labels in train_loader:
        hsi, dsm, labels = hsi.to(device), dsm.to(device), labels.to(device)

        optimizer.zero_grad()
        if USE_DSM:
            outputs = model(hsi, dsm)
        else:
            outputs = model(hsi, None)          # forward(hsi, mod2)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total   += labels.size(0)

    scheduler.step()
    train_acc = correct / total
    print(f"Epoch {epoch+1:3d} | loss={running_loss/len(train_loader):.4f} "
          f"| train_acc={train_acc:.4f}")

    # --- 验证 ---
    if (epoch + 1) % EVAL_INTERVAL == 0:
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for hsi, dsm, lab in val_loader:
                hsi = hsi.to(device)
                dsm = dsm.to(device)

                if USE_DSM:
                    pred = model(hsi, dsm).argmax(1).cpu()
                else:
                    pred = model(hsi, None).argmax(1).cpu()

                all_preds.append(pred)
                all_labels.append(lab)

        all_preds  = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        oa = (all_preds == all_labels).mean()
        print(f"  → Val OA = {oa:.4f} (best={best_val_oa:.4f})")

        if oa > best_val_oa:
            best_val_oa      = oa
            patience_counter = 0
            torch.save(model.state_dict(), PATH_WEIGHT + "best_model.pth")
            print(f"  ✅ Saved best model (OA={best_val_oa:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"🛑 Early stopping at epoch {epoch+1}")
                break

# ══════════════════════════════════════════════════════════
# 7. 测试
# ══════════════════════════════════════════════════════════

model.load_state_dict(torch.load(PATH_WEIGHT + "best_model.pth"))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for hsi, dsm, lab in test_loader:
        hsi = hsi.to(device)
        dsm = dsm.to(device)

        if USE_DSM:
            pred = model(hsi, dsm).argmax(1).cpu().numpy()
        else:
            pred = model(hsi, None).argmax(1).cpu().numpy()

        all_preds.append(pred)
        all_labels.append(lab.numpy())

y_pred = np.concatenate(all_preds)
y_true = np.concatenate(all_labels)

oa    = metrics.accuracy_score(y_true, y_pred)
kappa = metrics.cohen_kappa_score(y_true, y_pred)
cm    = metrics.confusion_matrix(y_true, y_pred)
each_acc, aa = utils.aa_and_each_accuracy(cm)

print(f"\nTest OA    = {oa:.4f}")
print(f"Kappa      = {kappa:.4f}")
print(f"AA         = {aa:.4f}")
print(f"Each class = {each_acc}")
  


