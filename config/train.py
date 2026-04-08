import sys
sys.path.append('.')

from config.segmentor import Segmentor
import pytorch_lightning as pl
import os
import torch
from module.model.decoder3d import LSNet3D_Seg
from datasets.datasets import create_dataloaders, create_nnunet_dataloaders

def main():
    # Auto-detect GPU if available, fallback to CPU
    device = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    # Model setup
    if device == "gpu":
        model = LSNet3D_Seg().cuda()
    else:
        model = LSNet3D_Seg()

    os.makedirs('./weight/BraTS/', exist_ok=True)
    check_point = pl.callbacks.model_checkpoint.ModelCheckpoint(
        './weight/BraTS/',
        filename="best_model_epoch={epoch:02d}_dice={val_dice_avg:.4f}",
        monitor="val_dice_avg",
        mode="max",
        save_top_k=1,
        verbose=True,
        save_weights_only=True,
        auto_insert_metric_name=False
    )
    progress_bar = pl.callbacks.TQDMProgressBar()
    params = {
        "accelerator": device,
        "devices": 1,
        "benchmark": True,
        "enable_progress_bar": True,
        "logger": True,
        "callbacks": [check_point, progress_bar],
        "log_every_n_steps": 1,
        "num_sanity_val_steps": 0,
        "max_epochs": 200,
        "precision": "16-mixed" if device == "gpu" else "32",
    }

    trainer = pl.Trainer(**params)

    base_dir = os.getenv("BRATS_DATA_DIR", r"E:\\3D SEGMENTATION BraTs\\datasets\\BraTS")

    # Check if it's nnU-Net format
    dataset_json = os.path.join(base_dir, 'dataset.json')
    if os.path.exists(dataset_json):
        train_dataset, val_dataset, test_dataset = create_nnunet_dataloaders(
            dataset_json,
            batch_size=4,
            val_split=0.1,
            test_split=0.1,
            num_workers=0,
            pin_memory=(device == "gpu"),
            crop_size=64,
            seed=42,
            output_test_list="test_patients_list.txt"
        )
    else:
        train_dataset, val_dataset, test_dataset = create_dataloaders(
            base_dir,
            batch_size=4,
            val_split=0.1,
            test_split=0.1,
            num_workers=0,
            pin_memory=(device == "gpu"),
            crop_size=64,
            seed=42,
            output_test_list="test_patients_list.txt"
        )

    segmentor = Segmentor(model=model)
    trainer.fit(segmentor, train_dataset, val_dataset)


if __name__ == '__main__':
    main()