import torch
import torch.nn as nn

from augmentations.mask import random_mask
from models.self_supervised import SelfSupervisedModel
from models import networks
from yucca.functional.utils.kwargs import filter_kwargs
import utils.visualisation as viz


class JointMAEAndSeg(SelfSupervisedModel):
    def __init__(self, *args, seg_loss_weight: float = 1.0, pretrained_encoder_path: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.seg_loss_weight = seg_loss_weight
        self._seg_loss_fn = self.dice_loss

        if pretrained_encoder_path:
            self.load_encoder_weights(pretrained_encoder_path)

    def load_encoder_weights(self, pretrained_encoder_path):
        print(f"Loading encoder weights from: {pretrained_encoder_path}")
        # The checkpoint is a lightning checkpoint, so we need to extract the state_dict.
        checkpoint = torch.load(pretrained_encoder_path, map_location=self.device)
        checkpoint_state_dict = checkpoint['state_dict']

        # The state_dict keys are prefixed with "model.", and we want to load the encoder part.
        # So we filter for keys starting with "model.encoder."
        # The keys in the checkpoint are like `model.encoder.layer. ...`
        # The keys in `self.model.encoder.state_dict()` are like `layer. ...`
        encoder_weights = {
            key.replace("model.encoder.", ""): value
            for key, value in checkpoint_state_dict.items()
            if key.startswith("model.encoder.")
        }

        # Now we load the weights into our model's encoder
        self.model.encoder.load_state_dict(encoder_weights)

        print("Successfully loaded pretrained encoder weights. Decoders are randomly initialized.")

    def dice_loss(self, y_hat, y):
        y_hat = torch.sigmoid(y_hat)
        # flatten
        y_hat = y_hat.view(-1)
        y = y.view(-1)
        intersection = (y_hat * y).sum()
        dice_score = (2.0 * intersection + 1e-6) / (y_hat.sum() + y.sum() + 1e-6)
        return 1.0 - dice_score

    def load_model(self):
        print(f"Loading Model: 3D {self.model_name} for joint MAE and Seg")
        model_func = getattr(networks, self.model_name)

        print("Found model: ", model_func)

        conv_op = torch.nn.Conv3d
        norm_op = torch.nn.InstanceNorm3d

        model_kwargs = {
            # Applies to all models
            "input_channels": self.input_channels,
            "num_classes": self.num_classes,
            # Applies to most CNN-based architectures
            "conv_op": conv_op,
            # Applies to most CNN-based architectures (exceptions: UXNet)
            "norm_op": norm_op,
            # Pretrainnig
            "mode": "joint_mae_seg",
            "patch_size": self.patch_size,
        }
        model_kwargs = filter_kwargs(model_func, model_kwargs)
        model = model_func(**model_kwargs)

        self.model = (
            torch.compile(model, mode=self.compile_mode)
            if self.should_compile
            else model
        )

    def _augment_and_forward(self, x):
        with torch.no_grad():
            x_masked, mask = random_mask(x, self.mask_ratio, self.mask_patch_size)

        y_hat_rec, y_hat_seg = self.model(x_masked)

        assert y_hat_rec is not None
        assert y_hat_seg is not None
        assert y_hat_rec.shape == x.shape, (
            f"Got shape: {y_hat_rec.shape}, expected: {x.shape}"
        )

        return y_hat_rec, y_hat_seg, mask

    def training_step(self, batch, batch_idx):
        x, y_seg = batch["image"], batch["label"]
        y_rec = x

        y_hat_rec, y_hat_seg, mask = self._augment_and_forward(x)

        rec_loss = self.rec_loss(
            y_hat_rec, y_rec, mask=mask if self.rec_loss_masked_only else None
        )
        seg_loss = self._seg_loss_fn(y_hat_seg, y_seg)

        total_loss = (
            1 - self.seg_loss_weight
        ) * rec_loss + self.seg_loss_weight * seg_loss

        if (
            batch_idx == 0
            and not self.disable_image_logging
            and not self.trainer.sanity_checking
        ):
            self._log_debug_images(
                x,
                y_rec,
                y_hat_rec,
                stage="train_recon",
                file_paths=batch["file_path"],
                idx=0,
            )
            self._log_debug_images(
                x,
                y_seg,
                torch.sigmoid(y_hat_seg),
                stage="train_seg",
                file_paths=batch["file_path"],
                idx=0,
            )

        self.log_dict(
            {
                "train/loss": total_loss,
                "train/rec_loss": rec_loss,
                "train/seg_loss": seg_loss,
            },
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y_seg = batch["image"], batch["label"]
        y_rec = x

        y_hat_rec, y_hat_seg, mask = self._augment_and_forward(x)

        rec_loss = self.rec_loss(
            y_hat_rec, y_rec, mask=mask if self.rec_loss_masked_only else None
        )
        seg_loss = self._seg_loss_fn(y_hat_seg, y_seg)

        total_loss = rec_loss + self.seg_loss_weight * seg_loss

        self.log_dict(
            {
                "val/loss": total_loss,
                "val/rec_loss": rec_loss,
                "val/seg_loss": seg_loss,
            }
        )
