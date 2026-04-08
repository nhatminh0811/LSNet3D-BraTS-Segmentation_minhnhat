import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from loss.loss import DS_UNETR_PlusPlus_Loss
from metrics.metrics import compute_brats_metrics

class Segmentor(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        # Class weights: Reduce background (0.1), prioritize tumor (1.0) and strongly enforce ET (5.0)
        # According to the paper, using weights helps CE stabilize better with small tumor regions
        self.register_buffer('class_weights', torch.tensor([0.1, 1.0, 1.0, 5.0]))
        
        # Initialize loss function from Scientific Reports 2025 paper
        self.criterion = DS_UNETR_PlusPlus_Loss(num_classes=4)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, batch_idx):
        image, y_true = batch 
        outputs = self.model(image)
        
        # Helper function to calculate loss (CE + Soft Dice) for each output (Main & Aux)
        def calc_combined_loss(y_p, y_t):
            return self.criterion(y_p, y_t, weight=self.class_weights)

        if isinstance(outputs, dict):
            y_pred = outputs['main']
            
            # 1. Loss for main layer
            loss_main = calc_combined_loss(y_pred, y_true)
            
            # 2. Deep Supervision for auxiliary layers
            aux_losses = 0
            count_aux = 0
            for key in outputs:
                if key.startswith('aux'):
                    aux_output = outputs[key]
                    target_size = aux_output.shape[2:] 
                    
                    # Resize labels to the size of the corresponding aux layer
                    y_true_aux = F.interpolate(
                        y_true.unsqueeze(1).float(), 
                        size=target_size, 
                        mode='nearest'
                    ).squeeze(1).long()
                    
                    aux_losses += calc_combined_loss(aux_output, y_true_aux)
                    count_aux += 1
            
            # Combined Loss: Main layer + 0.3 * average of auxiliary layers
            loss = loss_main + (0.3 * aux_losses / max(count_aux, 1))
        else:
            y_pred = outputs
            loss = calc_combined_loss(y_pred, y_true)
        
        # Calculate BraTS metrics (ET, TC, WT)
        brats_metrics = compute_brats_metrics(y_pred, y_true)
        return loss, brats_metrics

    def training_step(self, batch, batch_idx):
        loss, brats_metrics = self._step(batch, batch_idx)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_dice_avg", brats_metrics['dice_avg'], on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, brats_metrics = self._step(batch, batch_idx)
        metrics = {
            "val_loss": loss,
            "val_dice_et": brats_metrics['dice_et'],
            "val_dice_tc": brats_metrics['dice_tc'],
            "val_dice_wt": brats_metrics['dice_wt'],
            "val_dice_avg": brats_metrics['dice_avg']
        }
        self.log_dict(metrics, prog_bar=True)
        return metrics

    def configure_optimizers(self):
        # AdamW and ReduceLROnPlateau are the optimal pair for 3D UNet
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="max", 
            factor=0.5, 
            patience=10, 
            min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "monitor": "val_dice_avg"
            },
        }