import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from sklearn.model_selection import train_test_split
import json

class BraTSLoader(Dataset):
    def __init__(self, patient_dirs, transform=True, crop_size=64):
        self.patient_dirs = patient_dirs
        self.transform = transform
        self.crop_size = crop_size

    def __len__(self):
        return len(self.patient_dirs)

    def foreground_crop(self, image, seg, target_size):
        """
        Smart Crop Mechanism: Ensures 70% of crop blocks contain tumor regions (labels 1, 2, 3).
        Helps the model learn faster and achieve high Dice scores on small tumor regions.
        """
        if isinstance(target_size, int):
            target_size = (target_size,) * 3
            
        current_size = image.shape[:-1]
        fg_indices = np.argwhere(seg > 0)
        
        # 70% xác suất lấy tâm là vùng u, 30% crop ngẫu nhiên toàn bộ
        if len(fg_indices) > 0 and np.random.rand() < 0.7:
            center = fg_indices[np.random.randint(len(fg_indices))]
            crop_start = []
            for i in range(3):
                start = center[i] - target_size[i] // 2
                start = max(0, min(start, current_size[i] - target_size[i]))
                crop_start.append(start)
        else:
            crop_start = [np.random.randint(0, c - t + 1) if c > t else 0 
                          for c, t in zip(current_size, target_size)]
        
        image_crop = image[crop_start[0]:crop_start[0]+target_size[0], 
                           crop_start[1]:crop_start[1]+target_size[1], 
                           crop_start[2]:crop_start[2]+target_size[2], :]
        seg_crop = seg[crop_start[0]:crop_start[0]+target_size[0], 
                         crop_start[1]:crop_start[1]+target_size[1], 
                         crop_start[2]:crop_start[2]+target_size[2]]
        
        return image_crop, seg_crop

    def __getitem__(self, idx):
        patient_dir = self.patient_dirs[idx]
        patient_name = os.path.basename(patient_dir)
        
        # 1. Load 4 types of MRI images (Modalities)
        modalities = ['t1', 't1ce', 't2', 'flair']
        imgs = []
        for m in modalities:
            img_path = os.path.join(patient_dir, f'{patient_name}_{m}.nii')
            imgs.append(nib.load(img_path).get_fdata())
        
        image = np.stack(imgs, axis=-1).astype(np.float32)
        
        # 2. Load & Map Label: BraTS (0, 1, 2, 4) -> (0, 1, 2, 3)
        seg_path = os.path.join(patient_dir, f'{patient_name}_seg.nii')
        seg = nib.load(seg_path).get_fdata()
        seg_mapped = np.zeros_like(seg, dtype=np.int64)
        seg_mapped[seg == 1] = 1 # NCR/NET
        seg_mapped[seg == 2] = 2 # ED
        seg_mapped[seg == 3] = 3 # ET (BraTS with 0/1/2/3 labels)
        seg_mapped[seg == 4] = 3 # ET (BraTS with 0/1/2/4 labels)
        
        # 3. Z-score Normalization (Only calculated on brain regions > 0)
        for i in range(4):
            mask = image[..., i] > 0
            if mask.any():
                mean = image[..., i][mask].mean()
                std = image[..., i][mask].std()
                image[..., i][mask] = (image[..., i][mask] - mean) / (std + 1e-8)
        
        # 4. Apply Smart Crop
        image, seg = self.foreground_crop(image, seg_mapped, self.crop_size)
        
        # 5. Convert to Tensor (C, D, H, W)
        image = torch.from_numpy(image).permute(3, 0, 1, 2).float()
        seg = torch.from_numpy(seg).long()
        
        # 6. Data Augmentation (Only applied when transform=True)
        if self.transform:
            image, seg = self.augment(image, seg)
            
        return image, seg

    def augment(self, image, seg):
        # Random flip 
        if torch.rand(1) < 0.5:
            axis = torch.randint(1, 4, (1,)).item()
            image = torch.flip(image, [axis])
            seg = torch.flip(seg, [axis-1])
        
        # Add light Gaussian noise (Increase generalization)
        if torch.rand(1) < 0.2:
            image += torch.randn_like(image) * 0.02
            
        return image, seg


class BraTSNnUnetLoader(Dataset):
    def __init__(self, image_label_pairs, transform=True, crop_size=64):
        self.image_label_pairs = image_label_pairs
        self.transform = transform
        self.crop_size = crop_size

    def __len__(self):
        return len(self.image_label_pairs)

    def foreground_crop(self, image, seg, target_size):
        """
        Smart Crop Mechanism: Ensures 70% of crop blocks contain tumor regions (labels 1, 2, 3).
        Helps the model learn faster and achieve high Dice scores on small tumor regions.
        """
        if isinstance(target_size, int):
            target_size = (target_size,) * 3
            
        current_size = image.shape[:-1]
        fg_indices = np.argwhere(seg > 0)
        
        # 70% xác suất lấy tâm là vùng u, 30% crop ngẫu nhiên toàn bộ
        if len(fg_indices) > 0 and np.random.rand() < 0.7:
            center = fg_indices[np.random.randint(len(fg_indices))]
            crop_start = []
            for i in range(3):
                start = center[i] - target_size[i] // 2
                start = max(0, min(start, current_size[i] - target_size[i]))
                crop_start.append(start)
        else:
            crop_start = [np.random.randint(0, c - t + 1) if c > t else 0 
                          for c, t in zip(current_size, target_size)]
        
        image_crop = image[crop_start[0]:crop_start[0]+target_size[0], 
                           crop_start[1]:crop_start[1]+target_size[1], 
                           crop_start[2]:crop_start[2]+target_size[2], :]
        seg_crop = seg[crop_start[0]:crop_start[0]+target_size[0], 
                         crop_start[1]:crop_start[1]+target_size[1], 
                         crop_start[2]:crop_start[2]+target_size[2]]
        
        return image_crop, seg_crop

    def __getitem__(self, idx):
        img_path, label_path = self.image_label_pairs[idx]
        
        # 1. Load 4D image (Modalities stacked)
        img_nii = nib.load(img_path)
        image = img_nii.get_fdata().astype(np.float32)  # Shape: (D, H, W, 4)
        
        # 2. Load label
        seg_nii = nib.load(label_path)
        seg = seg_nii.get_fdata().astype(np.int64)  # Shape: (D, H, W)
        
        # 3. Z-score Normalization (Only calculated on brain regions > 0)
        for i in range(4):
            mask = image[..., i] > 0
            if mask.any():
                mean = image[..., i][mask].mean()
                std = image[..., i][mask].std()
                image[..., i][mask] = (image[..., i][mask] - mean) / (std + 1e-8)
        
        # 4. Apply Smart Crop
        image, seg = self.foreground_crop(image, seg, self.crop_size)
        
        # 5. Convert to Tensor (C, D, H, W)
        image = torch.from_numpy(image).permute(3, 0, 1, 2).float()
        seg = torch.from_numpy(seg).long()
        
        # 6. Data Augmentation (Only applied when transform=True)
        if self.transform:
            image, seg = self.augment(image, seg)
            
        return image, seg

    def augment(self, image, seg):
        # Random flip 
        if torch.rand(1) < 0.5:
            axis = torch.randint(1, 4, (1,)).item()
            image = torch.flip(image, [axis])
            seg = torch.flip(seg, [axis-1])
        
        # Add light Gaussian noise (Increase generalization)
        if torch.rand(1) < 0.2:
            image += torch.randn_like(image) * 0.02
            
        return image, seg

# --- HÀM HỖ TRỢ CHIA DỮ LIỆU ---

def get_valid_patient_dirs(base_dir):
    valid_dirs = []
    if os.path.exists(base_dir):
        for d in os.listdir(base_dir):
            patient_path = os.path.join(base_dir, d)
            if os.path.isdir(patient_path):
                patient_name = d
                # Kiểm tra đủ 5 file mới nạp
                required_files = [f'{patient_name}_{m}.nii' for m in ['t1', 't1ce', 't2', 'flair', 'seg']]
                if all(os.path.exists(os.path.join(patient_path, f)) for f in required_files):
                    valid_dirs.append(patient_path)
    return valid_dirs


def create_dataloaders(
    base_dir,
    batch_size=4,
    val_split=0.1,
    test_split=0.1,
    num_workers=4,
    pin_memory=True,
    crop_size=64,
    seed=42,
    output_test_list=None,
):
    all_dirs = get_valid_patient_dirs(base_dir)
    if len(all_dirs) == 0:
        raise ValueError(
            f"No valid patient directories found in '{base_dir}'. "
            "Check your path and BraTS data structure."
        )

    test_val_ratio = val_split + test_split
    if test_val_ratio >= 1.0:
        raise ValueError("val_split + test_split must be less than 1.0")

    train_dirs, temp_dirs = train_test_split(
        all_dirs, test_size=test_val_ratio, random_state=seed
    )
    if len(temp_dirs) == 0:
        raise ValueError("Split ratios too small; no data left for validation/testing")

    val_ratio = val_split / test_val_ratio
    val_dirs, test_dirs = train_test_split(
        temp_dirs, test_size=test_split / test_val_ratio, random_state=seed
    )

    if output_test_list:
        with open(output_test_list, "w") as f:
            for path in test_dirs:
                f.write(os.path.basename(path) + "\n")

    train_loader = DataLoader(
        BraTSLoader(train_dirs, transform=True, crop_size=crop_size),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        BraTSLoader(val_dirs, transform=False, crop_size=crop_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        BraTSLoader(test_dirs, transform=False, crop_size=crop_size),
        batch_size=1,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def create_nnunet_dataloaders(
    dataset_json_path,
    batch_size=4,
    val_split=0.1,
    test_split=0.1,
    num_workers=4,
    pin_memory=True,
    crop_size=64,
    seed=42,
    output_test_list=None,
):
    # Load dataset.json
    with open(dataset_json_path, 'r') as f:
        dataset_info = json.load(f)
    
    training_data = dataset_info['training']
    
    # Get base directory
    base_dir = os.path.dirname(dataset_json_path)
    imagesTr_dir = os.path.join(base_dir, 'imagesTr')
    labelsTr_dir = os.path.join(base_dir, 'labelsTr')
    
    # Create image-label pairs
    all_pairs = []
    for item in training_data:
        img_path = os.path.join(base_dir, item['image'])
        label_path = os.path.join(base_dir, item['label'])
        if os.path.exists(img_path) and os.path.exists(label_path):
            all_pairs.append((img_path, label_path))
    
    if len(all_pairs) == 0:
        raise ValueError(
            f"No valid image-label pairs found in dataset.json. "
            "Check your paths and dataset structure."
        )

    test_val_ratio = val_split + test_split
    if test_val_ratio >= 1.0:
        raise ValueError("val_split + test_split must be less than 1.0")

    train_pairs, temp_pairs = train_test_split(
        all_pairs, test_size=test_val_ratio, random_state=seed
    )
    if len(temp_pairs) == 0:
        raise ValueError("Split ratios too small; no data left for validation/testing")

    val_ratio = val_split / test_val_ratio
    val_pairs, test_pairs = train_test_split(
        temp_pairs, test_size=test_split / test_val_ratio, random_state=seed
    )

    if output_test_list:
        with open(output_test_list, "w") as f:
            for img_path, _ in test_pairs:
                patient_name = os.path.basename(img_path).replace('.nii.gz', '').replace('BRATS_', '')
                f.write(patient_name + "\n")

    train_loader = DataLoader(
        BraTSNnUnetLoader(train_pairs, transform=True, crop_size=crop_size),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        BraTSNnUnetLoader(val_pairs, transform=False, crop_size=crop_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        BraTSNnUnetLoader(test_pairs, transform=False, crop_size=crop_size),
        batch_size=1,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create BraTS dataloaders")
    parser.add_argument("--data-dir", type=str, required=True, help="BraTS training data root folder")
    parser.add_argument("--test-patient-list", type=str, default="test_patients_list.txt")
    args = parser.parse_args()

    # Check if it's nnU-Net format
    dataset_json = os.path.join(args.data_dir, 'dataset.json')
    if os.path.exists(dataset_json):
        train_loader, val_loader, test_loader = create_nnunet_dataloaders(
            dataset_json,
            output_test_list=args.test_patient_list,
        )
    else:
        train_loader, val_loader, test_loader = create_dataloaders(
            args.data_dir,
            output_test_list=args.test_patient_list,
        )

    print("-" * 30)
    print(f"Total valid cases:    {len(train_loader.dataset.image_label_pairs) if hasattr(train_loader.dataset, 'image_label_pairs') else len(train_loader.dataset.patient_dirs) + len(val_loader.dataset.patient_dirs) + len(test_loader.dataset.patient_dirs)}")
    print(f"Training set:         {len(train_loader.dataset.image_label_pairs) if hasattr(train_loader.dataset, 'image_label_pairs') else len(train_loader.dataset.patient_dirs)} cases")
    print(f"Validation set:       {len(val_loader.dataset.image_label_pairs) if hasattr(val_loader.dataset, 'image_label_pairs') else len(val_loader.dataset.patient_dirs)} cases")
    print(f"Testing set:          {len(test_loader.dataset.image_label_pairs) if hasattr(test_loader.dataset, 'image_label_pairs') else len(test_loader.dataset.patient_dirs)} cases")
    print(f"Test patient list saved to: {args.test_patient_list}")
    print("-" * 30)