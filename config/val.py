import sys
sys.path.append('.')

import pytorch_lightning as pl
from loss.loss import dice_focal_loss
from metrics.metrics import multi_class_dice, multi_class_iou, sensitivity_specificity
from module.model.decoder3d import LSNet3D_Seg
from datasets.datasets import val_dataset

class Segmentor(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        image, y_true = batch
        y_pred = self.model(image)
        loss = dice_focal_loss(y_pred, y_true, num_classes=4)
        dice = multi_class_dice(y_pred, y_true, num_classes=4)
        iou = multi_class_iou(y_pred, y_true, num_classes=4)
        sens, spec = sensitivity_specificity(y_pred, y_true, num_classes=4)
        # Log average metrics
        metrics = {
            "Test Dice": sum(dice)/len(dice),
            "Test IoU": sum(iou)/len(iou),
            "Test Sensitivity": sum(sens)/len(sens),
            "Test Specificity": sum(spec)/len(spec),
        }
        self.log_dict(metrics, prog_bar=True)
        return metrics

model = LSNet3D_Seg()  # CPU
model.eval()
# Dataset & Data Loader
CHECKPOINT_PATH = "./weight/BraTS/ckpt.ckpt"  # Adjust path
# Prediction
trainer = pl.Trainer()
segmentor = Segmentor.load_from_checkpoint(CHECKPOINT_PATH, model=model)
trainer.test(segmentor, val_dataset)