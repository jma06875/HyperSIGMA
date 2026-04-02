import scipy.io as sio
import numpy as np

# 加载 .mat 文件
mat = sio.loadmat('data/HST_10_rep1.mat')

# 查看变量
print("Keys:", mat.keys())
print("TR shape:", mat['TR'].shape)
print("TR dtype:", mat['TR'].dtype)
print("Unique values in TR:", np.unique(mat['TR']))
print("Max label in TR:", np.max(mat['TR']))