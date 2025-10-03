from typing import List, Dict, Any
import torch
import torch.nn as nn
import pytorch_lightning as pl
from hydra.utils import instantiate


class MultiTaskModel(pl.LightningModule):
    def __init__(self,
                 backbone,
                 location_head,
                 matching_head,
                 task_weights):
        super().__init__()
        self.save_hyperparameters(ignore=["heads", "optimizer", "scheduler", "loss", "task_weights"])
        self.backbone = backbone
        self.location_head = location_head
        self.matching_head = matching_head
        self.task_weights = task_weights

        self.prev_info = None

    def forward(self, x, prev_info):
        features = self.backbone(x)
        location_out = self.location_head(features)
        match_out = self.matching_head(
            backbone_feats=features,
            prev_info=prev_info,  # {"coords": prev_xy, "frame_id": t0, "other_feat": prev_f}
            curr_info=location_out,  # {"coords": curr_xy, "frame_id": t1, "other_feat": curr_f}
        )
        return location_out, match_out

    def training_step(self, batch, batch_idx):
        # forward
        x, prev_info, curr_info = batch
        features = self.backbone(x)
        location_out = self.location_head(features)
        match_out = self.matching_head(
            backbone_feats=features,
            prev_info=prev_info,  # {"coords": prev_xy, "frame_id": t0, "other_feat": prev_f}
            curr_info=curr_info,  # {"coords": curr_xy, "frame_id": t1, "other_feat": curr_f}
        )

        # compute losses and metrics
        location_loss = self.location_head.compute_loss(location_out, curr_info)
        location_metrics = self.location_head.compute_metrics(location_out, curr_info)
        matching_loss = self.matching_head.compute_loss(
            match_out["logits_aug"],
            row_target=row_target,
            col_target=col_target
        )
        matching_metrics = self.matching_head.compute_metrics(
            match_out["logits_aug"],
            row_target=row_target,
            col_target=col_target
        )
        loss = location_loss * self.task_weights[0] + matching_loss * self.task_weights[1]
        metrics = {**location_metrics, **matching_metrics}

        # logging
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/location_loss", location_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/matching_loss", matching_loss, on_step=True, on_epoch=True, prog_bar=True)
        for k, v in metrics.items():
            self.log(f"train/{k}", v, on_step=False, on_epoch=True)

        return loss, metrics

    def validation_step(self, batch, batch_idx):
        loss, metrics = self._compute_losses(batch)
        self.log("val/total_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        # also log metrics
        for k, v in metrics.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        loss, metrics = self._compute_losses(batch)
        self.log("test/total_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for k, v in metrics.items():
            self.log(f"test/{k}", v, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        x, prev_info, curr_info = batch
        location_out, match_out = self(x, prev_info)
        matches, prev_unm, curr_unm = self.matching_head.assign_greedy(
            match_out["logits_aug"], match_out["row_mask"], match_out["col_mask"]
        )
        return location_out, matches, prev_unm, curr_unm

    def configure_optimizers(self):
        optimizer = instantiate(self.optimizer_cfg, params=self.parameters())
        if self.scheduler_cfg is None:
            return optimizer
        try:
            scheduler = instantiate(self.scheduler_cfg, optimizer=optimizer)
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "monitor": "val/total_loss", "interval": "epoch"}}
        except TypeError:
            scheduler = instantiate(self.scheduler_cfg, optimizer)
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "monitor": "val/total_loss", "interval": "epoch"}}
