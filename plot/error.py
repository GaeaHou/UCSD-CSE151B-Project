import numpy as np
import matplotlib.pyplot as plt

# 加载训练集预测值和真实值（形状应为 N × 2 × 48 × 72）
val_preds = np.load("val_preds.npy")
val_trues = np.load("val_trues.npy")

# 确保数据形状一致
assert val_preds.shape == val_trues.shape
n_samples = val_preds.shape[0]

# 每个样本的 MSE
mse_per_sample = np.mean((val_preds - val_trues) ** 2, axis=(1, 2, 3))

# 找出 top 3 最大误差的样本索引
topk = 3
top_indices = np.argsort(mse_per_sample)[-topk:][::-1]

# 可视化每个高误差样本的 tas 和 pr 误差图
for i, idx in enumerate(top_indices):
    pred = val_preds[idx]
    true = val_trues[idx]
    error = pred - true  # shape: (2, 48, 72)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axs[0].imshow(error[0], cmap='coolwarm')
    axs[0].set_title(f"Sample {idx} - tas error")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(error[1], cmap='coolwarm')
    axs[1].set_title(f"Sample {idx} - pr error")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    plt.suptitle(f"Top-{i+1} Highest MSE Sample (MSE={mse_per_sample[idx]:.2f})")
    plt.tight_layout()
    plt.show()