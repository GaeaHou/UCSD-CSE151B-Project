import re
import matplotlib.pyplot as plt

# 读取日志文件
with open("loss.txt", "r") as f:
    lines = f.readlines()

# 提取 Train 和 Validation 的 MSE
train_mse = []
val_mse = []
epochs = []

for line in lines:
    train_match = re.search(r"\[Epoch (\d+)\] Train Loss \(MSE\): ([\d.]+)", line)
    val_match = re.search(r"\[Epoch (\d+)\] Validation MSE \(manual\): ([\d.]+)", line)

    if train_match:
        epoch = int(train_match.group(1))
        train_mse.append(float(train_match.group(2)))
        if epoch not in epochs:
            epochs.append(epoch)

    if val_match:
        val_mse.append(float(val_match.group(2)))

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(epochs[:len(train_mse)], train_mse, label='Train MSE Loss',
         color='tab:blue', linewidth=2, marker='o', markersize=5)
plt.plot(epochs[:len(val_mse)], val_mse, label='Validation MSE Loss',
         color='tab:orange', linewidth=2, marker='o', markersize=5)

# 图形美化
plt.title("Training and Validation MSE Loss per Epoch", fontsize=15)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("MSE Loss", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
plt.tight_layout()
plt.show()