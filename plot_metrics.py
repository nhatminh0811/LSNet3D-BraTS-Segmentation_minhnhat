import pandas as pd
import matplotlib.pyplot as plt

# 1. Point to the path of your log file
log_path = 'lightning_logs/version_2/metrics.csv' 
df = pd.read_csv(log_path)

# PyTorch Lightning logs per step, we need to group by epoch for smoother plots
df_epoch = df.groupby('epoch').mean(numeric_only=True).reset_index()

# 2. Create a figure layout like your sample (2 rows, 3 columns)
plt.figure(figsize=(18, 10))

# Plot Loss
if 'train_loss' in df_epoch.columns:
    plt.subplot(2, 3, 1)
    plt.plot(df_epoch['epoch'], df_epoch['train_loss'], color='red')
    plt.title("Epoch Average Loss")
    plt.xlabel("epoch")
else:
    print("Column 'train_loss' not found in metrics.csv")

# Plot Overall Mean Dice
if 'val_dice_avg' in df_epoch.columns:
    plt.subplot(2, 3, 2)
    plt.plot(df_epoch['epoch'], df_epoch['val_dice_avg'], color='green')
    plt.title("Val Mean Dice")
    plt.xlabel("epoch")
else:
    print("Column 'val_dice_avg' not found in metrics.csv")

# Plot specific regions (TC, WT, ET)
metrics = ['val_dice_tc', 'val_dice_wt', 'val_dice_et']
colors = ['blue', 'brown', 'purple']

for i, m in enumerate(metrics):
    plt.subplot(2, 3, i+4)
    if m in df_epoch.columns:
        plt.plot(df_epoch['epoch'], df_epoch[m], color=colors[i])
        plt.title(f"Val Mean Dice {m.split('_')[-1].upper()}")
        plt.ylim(0, 1)
        plt.xlabel("epoch")
    else:
        plt.title(f"No {m}")

plt.tight_layout()
plt.savefig("my_lsnet_3d_results.png")
plt.show()