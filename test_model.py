import sys
sys.path.append('.')

import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from config.segmentor import Segmentor
from module.model.decoder3d import LSNet3D_Seg
from metrics.metrics import compute_brats_metrics, brats_hausdorff_distance

# ==========================================================
# 1. UTILITY: Sliding Window Inference (Logits bug fixed)
# ==========================================================
def sliding_window_inference(image, model, window_size=128, overlap=0.5):
    batch, channels, d, h, w = image.shape
    stride = int(window_size * (1 - overlap))
    
    output = None
    count = torch.zeros((batch, 1, d, h, w), device=image.device)
    
    for start_d in range(0, d, stride):
        for start_h in range(0, h, stride):
            for start_w in range(0, w, stride):
                end_d, end_h, end_w = min(start_d + window_size, d), min(start_h + window_size, h), min(start_w + window_size, w)
                s_d, s_h, s_w = max(0, end_d - window_size), max(0, end_h - window_size), max(0, end_w - window_size)
                patch = image[:, :, s_d:end_d, s_h:end_h, s_w:end_w]
                
                with torch.no_grad():
                    patch_output = model(patch)
                    if isinstance(patch_output, dict): 
                        patch_output = patch_output['main']
                    
                    if patch_output.shape[1] == 4:
                        # Full 4-class softmax for background + tumor regions
                        patch_output = torch.softmax(patch_output, dim=1)
                    else:
                        patch_output = torch.sigmoid(patch_output) # For 3 independent channels

                if output is None:
                    output = torch.zeros((batch, patch_output.shape[1], d, h, w), device=image.device)

                output[:, :, s_d:end_d, s_h:end_h, s_w:end_w] += patch_output
                count[:, :, s_d:end_d, s_h:end_h, s_w:end_w] += 1
    
    return output / (count + 1e-8)

# ==========================================================
# 2. UTILITY: Load BraTS (Ensure correct normalization)
# ==========================================================
def load_brats_case(input_path):
    # Check if input_path is a folder (old format) or file (nnU-Net format)
    if os.path.isdir(input_path):
        # Old format: folder with separate modality files
        modalities = ["t1", "t1ce", "t2", "flair"]
        files = os.listdir(input_path)
        images = []
        
        for mod in modalities:
            path = None
            for f in files:
                if mod in f.lower() and (f.endswith(".nii") or f.endswith(".nii.gz")):
                    path = os.path.join(input_path, f)
                    break
            
            if path is None: raise ValueError(f"File not found for modality: {mod}")
            
            data = nib.load(path).get_fdata().astype(np.float32)
            
            # Z-score normalization on brain region
            mask = data > 0
            if mask.any():
                data[mask] = (data[mask] - data[mask].mean()) / (data[mask].std() + 1e-8)
            images.append(data)

        # Return shape (C, H, W, D)
        return torch.from_numpy(np.stack(images, axis=0))
    else:
        # nnU-Net format: single 4D .nii.gz file
        data = nib.load(input_path).get_fdata().astype(np.float32)  # Shape: (D, H, W, 4)
        
        # Z-score normalization for each modality
        for i in range(4):
            mask = data[..., i] > 0
            if mask.any():
                data[..., i][mask] = (data[..., i][mask] - data[..., i][mask].mean()) / (data[..., i][mask].std() + 1e-8)
        
        # Return shape (C, H, W, D) to match old-format expectations
        return torch.from_numpy(np.transpose(data, (3, 0, 1, 2)))


def load_brats_seg(input_path):
    if os.path.isdir(input_path):
        # Old format: folder contains separate modality files plus seg file
        seg_path = None
        for f in os.listdir(input_path):
            if f.lower().endswith("_seg.nii") or f.lower().endswith("_seg.nii.gz"):
                seg_path = os.path.join(input_path, f)
                break
        if seg_path is None:
            raise FileNotFoundError(f"Segmentation file not found in folder {input_path}")

        seg = nib.load(seg_path).get_fdata().astype(np.int64)
        seg_mapped = np.zeros_like(seg, dtype=np.int64)
        seg_mapped[seg == 1] = 1
        seg_mapped[seg == 2] = 2
        seg_mapped[seg == 3] = 3
        seg_mapped[seg == 4] = 3
        return torch.from_numpy(seg_mapped)
    else:
        # nnU-Net format: single .nii.gz file
        seg = nib.load(input_path).get_fdata().astype(np.int64)
        # Ensure shape is (D, H, W) = (155, 240, 240)
        if seg.shape == (240, 240, 155):
            seg = np.transpose(seg, (2, 0, 1))
        seg_mapped = np.zeros_like(seg, dtype=np.int64)
        seg_mapped[seg == 1] = 1
        seg_mapped[seg == 2] = 2
        seg_mapped[seg == 3] = 3
        seg_mapped[seg == 4] = 3
        return torch.from_numpy(seg_mapped)


# ==========================================================
# 3. VISUALIZATION: Fix gray display (vmin/vmax)
# ==========================================================
def _prob_to_mask(pred_probs):
    probs = pred_probs.squeeze(0).cpu().numpy()
    if probs.shape[0] == 4:
        # Full softmax prediction: take argmax across classes
        return np.argmax(probs, axis=0).astype(np.int32)

    mask = np.zeros(probs.shape[1:], dtype=np.int32)
    mask[probs[1] > 0.5] = 2  # ED
    mask[probs[0] > 0.5] = 1  # NCR
    mask[probs[2] > 0.5] = 3  # ET
    return mask


def visualize_full_report(image_tensor, pred_probs, gt_seg=None, save_path="BraTS_Final_Report.png"):
    # 1. Convert prediction probabilities to mask
    pred_mask = _prob_to_mask(pred_probs)

    cmap = mcolors.ListedColormap(['black', 'red', 'green', 'blue'])
    norm = mcolors.BoundaryNorm([0, 1, 2, 3, 4], cmap.N)

    # Convert original image to (C, D, H, W)
    image_np = image_tensor.permute(0, 3, 1, 2).cpu().numpy()

    # Use prediction mask to determine best tumor slices
    tumor_axial = np.sum(pred_mask > 0, axis=(1, 2))
    idx_axial = np.argmax(tumor_axial) if np.any(tumor_axial) else image_np.shape[1] // 2
    tumor_coronal = np.sum(pred_mask > 0, axis=(0, 2))
    idx_coronal = np.argmax(tumor_coronal) if np.any(tumor_coronal) else image_np.shape[2] // 2
    tumor_sagittal = np.sum(pred_mask > 0, axis=(0, 1))
    idx_sagittal = np.argmax(tumor_sagittal) if np.any(tumor_sagittal) else image_np.shape[3] // 2

    # Prepare GT mask if given
    gt_mask = None
    if gt_seg is not None:
        gt_mask = gt_seg.squeeze(0).cpu().numpy().astype(np.int32)

    fig, axs = plt.subplots(3, 6, figsize=(26, 14), facecolor='black')
    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    views = [
        ('AXIAL',    lambda m: np.rot90(m[idx_axial, :, :])),
        ('CORONAL',  lambda m: np.rot90(m[:, idx_coronal, :])),
        ('SAGITTAL', lambda m: np.rot90(m[:, :, idx_sagittal]))
    ]

    titles = ['T1', 'T1ce', 'T2', 'FLAIR', 'PREDICTION', 'GT']

    for row, (view_name, get_slice) in enumerate(views):
        for col in range(4):
            axs[row, col].imshow(get_slice(image_np[col]), cmap='gray', vmin=-2, vmax=2, aspect='auto')
            axs[row, col].axis('off')
            if row == 0:
                axs[row, col].set_title(titles[col], color='white', fontsize=16)

        axs[row, 4].imshow(get_slice(pred_mask), cmap=cmap, norm=norm, interpolation='nearest', aspect='auto')
        axs[row, 4].axis('off')
        if row == 0:
            axs[row, 4].set_title(titles[4], color='cyan', fontsize=16, fontweight='bold')

        if gt_mask is not None:
            axs[row, 5].imshow(get_slice(gt_mask), cmap=cmap, norm=norm, interpolation='nearest', aspect='auto')
            axs[row, 5].axis('off')
            if row == 0:
                axs[row, 5].set_title(titles[5], color='lime', fontsize=16, fontweight='bold')

        axs[row, 0].text(-20, 120, view_name, color='yellow', rotation=90, va='center', fontsize=14, fontweight='bold')

    plt.suptitle("BraTS: Prediction vs GT", color='white', fontsize=24, y=0.98)
    plt.savefig(save_path, facecolor='black', bbox_inches='tight')
    print(f"✓ Full report saved at: {save_path}")

# ==========================================================
# 4. MAIN RUN
# ==========================================================
def load_test_cases(base_dir, test_list_path="test_patients_list.txt"):
    if not os.path.exists(test_list_path):
        raise FileNotFoundError(f"Test patient list not found: {test_list_path}")

    with open(test_list_path, "r") as f:
        patient_ids = [line.strip() for line in f if line.strip()]

    if not patient_ids:
        raise ValueError(f"Test patient list is empty: {test_list_path}")

    # Check if nnU-Net format
    dataset_json = os.path.join(base_dir, 'dataset.json')
    if os.path.exists(dataset_json):
        # nnU-Net format
        test_cases = []
        for pid in patient_ids:
            img_path = os.path.join(base_dir, 'imagesTr', f'BRATS_{pid}.nii.gz')
            label_path = os.path.join(base_dir, 'labelsTr', f'BRATS_{pid}.nii.gz')
            if os.path.exists(img_path) and os.path.exists(label_path):
                test_cases.append((img_path, label_path))
            else:
                print(f"Warning: Test case {pid} files not found")
        return test_cases
    else:
        # Old format
        case_folders = []
        for pid in patient_ids:
            folder = os.path.join(base_dir, pid)
            if not os.path.isdir(folder):
                raise FileNotFoundError(f"Test case folder does not exist: {folder}")
            case_folders.append(folder)
        return case_folders


def run_main():
    checkpoint_path = "./weight/BraTS/best_model_epoch=82_dice=0.8275.ckpt"  # Replace with your .ckpt file
    base_dir = os.getenv("BRATS_DATA_DIR", r"E:\\3D SEGMENTATION BraTs\\datasets\\BraTS")
    test_list_path = os.getenv("BRATS_TEST_LIST", "test_patients_list.txt")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Initialize Model
    model = LSNet3D_Seg()
    segmentor = Segmentor(model)

    # Load weights from Lightning Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    segmentor.load_state_dict(checkpoint['state_dict'])
    segmentor.to(device).eval()

    # 2. Get test cases from list
    test_cases = load_test_cases(base_dir, test_list_path)
    print(f"Found {len(test_cases)} test cases from {test_list_path}")

    # 3. Inference loop
    output_dir = os.getenv("BRATS_TEST_OUTPUT", "test_outputs")
    os.makedirs(output_dir, exist_ok=True)

    dice_et_list = []
    dice_tc_list = []
    dice_wt_list = []
    hd95_et_list = []
    hd95_tc_list = []
    hd95_wt_list = []

    for idx, case in enumerate(test_cases, 1):
        if isinstance(case, tuple):
            # nnU-Net format: (img_path, label_path)
            img_path, label_path = case
            case_name = os.path.basename(img_path).replace('.nii.gz', '').replace('BRATS_', '')
            print(f"\n--- Case {idx}/{len(test_cases)}: {case_name} ---")
            
            img_tensor = load_brats_case(img_path)
            seg_tensor = load_brats_seg(label_path).unsqueeze(0)  # (1, D, H, W)
        else:
            # Old format: folder path
            case_folder = case
            case_name = os.path.basename(case_folder)
            print(f"\n--- Case {idx}/{len(test_cases)}: {case_name} ---")
            
            img_tensor = load_brats_case(case_folder)
            seg_tensor = load_brats_seg(case_folder).unsqueeze(0)  # (1, D, H, W)

        input_batch = img_tensor.permute(0, 3, 1, 2).unsqueeze(0).to(device)

        with torch.no_grad():
            preds = sliding_window_inference(input_batch, segmentor, window_size=128, overlap=0.5)

        # Build 4-channel prediction (0=background,1=NCR,2=ED,3=ET) if needed
        if preds.shape[1] == 4:
            pred_probs_4 = preds
        else:
            out_probs = preds.squeeze(0).cpu()  # (3, D, H, W)
            back_prob = 1.0 - torch.max(out_probs, dim=0, keepdim=True)[0]
            pred_probs_4 = torch.cat([back_prob, out_probs], dim=0).unsqueeze(0)  # (1,4,D,H,W)

        seg_tensor = seg_tensor.to(device)

        # Compute metrics
        metrics = compute_brats_metrics(pred_probs_4, seg_tensor)
        hd95 = brats_hausdorff_distance(pred_probs_4, seg_tensor)

        print(f"dice_et={metrics['dice_et']:.4f}, dice_tc={metrics['dice_tc']:.4f}, dice_wt={metrics['dice_wt']:.4f}, dice_avg={metrics['dice_avg']:.4f}")
        if hd95 is not None:
            print(f"hd95_et={hd95['hd95_et']:.3f}, hd95_tc={hd95['hd95_tc']:.3f}, hd95_wt={hd95['hd95_wt']:.3f}")

        dice_et_list.append(metrics['dice_et'])
        dice_tc_list.append(metrics['dice_tc'])
        dice_wt_list.append(metrics['dice_wt'])
        if hd95 is not None:
            hd95_et_list.append(hd95['hd95_et'])
            hd95_tc_list.append(hd95['hd95_tc'])
            hd95_wt_list.append(hd95['hd95_wt'])

        save_path = os.path.join(output_dir, f"BraTS_Test_{case_name}_Final.png")
        visualize_full_report(img_tensor, preds, gt_seg=seg_tensor, save_path=save_path)

    # Summary
    print(f"\nAll test cases processed. Outputs saved to: {os.path.abspath(output_dir)}")
    print(f"Average Dice: ET={np.mean(dice_et_list):.4f}, TC={np.mean(dice_tc_list):.4f}, WT={np.mean(dice_wt_list):.4f}")
    if hd95_et_list:
        print(f"Average HD95: ET={np.mean(hd95_et_list):.3f}, TC={np.mean(hd95_tc_list):.3f}, WT={np.mean(hd95_wt_list):.3f}")

    print("\nAll test cases processed.")

if __name__ == "__main__":
    run_main()