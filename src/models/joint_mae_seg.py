import torch
import torch.nn as nn
from lightning import LightningModule
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from monai.losses import DiceCELoss

from models.networks.unet import unet_b
from utils.masking import generate_random_mask
from torchmetrics import MeanMetric


class JointMaeSegModel(LightningModule):
    def __init__(
        self,
        config: dict,
        learning_rate: float = 1e-4,
        warmup_epochs: int = 5,
        epochs: int = 100,
        steps_per_epoch: int = 1,
        optimizer: str = "AdamW",
        should_compile: bool = False,
        compile_mode: str | None = None,
        rec_loss_masked_only: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = unet_b(mode="joint_mae_seg", input_channels=1, output_channels=1)

        if should_compile:
            self.model = torch.compile(self.model, mode=compile_mode)

        self.mae_loss = nn.MSELoss()
        self.seg_loss = DiceCELoss(to_onehot_y=True, softmax=True)
        self.masking_generator = MaskingGenerator(
            (
                self.config["mask_patch_size"],
                self.config["mask_patch_size"],
                self.config["mask_patch_size"],
            ),
            self.config["mask_ratio"],
        )
        self.lr = learning_rate
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.optimizer = optimizer
        self.rec_loss_masked_only = rec_loss_masked_only

        self.train_mae_loss = MeanMetric()
        self.train_seg_loss = MeanMetric()
        self.val_mae_loss = MeanMetric()
        self.val_seg_loss = MeanMetric()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, seg = batch["image"], batch["label"]

        # MAE part
        mask = self.masking_generator()
        mask = mask.to(self.device)
        masked_img = img * (1 - mask)
        
        mae_rec, seg_pred = self.forward(masked_img)

        if self.rec_loss_masked_only:
            mae_loss = self.mae_loss(mae_rec * mask, img * mask)
        else:
            mae_loss = self.mae_loss(mae_rec, img)

        # Segmentation part
        # The model will be trained on the full image for segmentation
        _, seg_pred_unmasked = self.forward(img)
        seg_loss = self.seg_loss(seg_pred_unmasked, seg)

        total_loss = mae_loss + seg_loss

        self.train_mae_loss.update(mae_loss)
        self.train_seg_loss.update(seg_loss)
        self.log("train/mae_loss", self.train_mae_loss, on_step=True, on_epoch=True)
        self.log("train/seg_loss", self.train_seg_loss, on_step=True, on_epoch=True)
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        img, seg = batch["image"], batch["label"]

        # MAE part
        mask = self.masking_generator()
        mask = mask.to(self.device)
        masked_img = img * (1 - mask)
        
        mae_rec, seg_pred = self.forward(masked_img)

        if self.rec_loss_masked_only:
            mae_loss = self.mae_loss(mae_rec * mask, img * mask)
        else:
            mae_loss = self.mae_loss(mae_rec, img)

        # Segmentation part
        _, seg_pred_unmasked = self.forward(img)
        seg_loss = self.seg_loss(seg_pred_unmasked, seg)

        total_loss = mae_loss + seg_loss

        self.val_mae_loss.update(mae_loss)
        self.val_seg_loss.update(seg_loss)
        self.log("val/mae_loss", self.val_mae_loss, on_step=True, on_epoch=True)
        self.log("val/seg_loss", self.val_seg_loss, on_step=True, on_epoch=True)
        self.log("val/total_loss", total_loss, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        if self.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=1e-5
            )
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-5
            )
        else:
            raise ValueError(f"Optimizer {self.optimizer} not supported")

        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.warmup_epochs * self.steps_per_epoch,
            T_mult=1,
            eta_min=1e-6,
        )
        return [optimizer], [scheduler]