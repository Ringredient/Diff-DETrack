import math
from typing import Dict, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierPosEnc(nn.Module):
    def __init__(self, num_freqs: int = 6):
        super().__init__()
        self.num_freqs = num_freqs
        freq = torch.tensor([2.0 ** k for k in range(num_freqs)], dtype=torch.float32) * math.pi
        self.register_buffer("freq", freq, persistent=False)

    def forward(self, xy):
        x = xy.unsqueeze(-1) * self.freq  # [N,2,F]
        s = torch.sin(x)                  # [N,2,F]
        c = torch.cos(x)                  # [N,2,F]
        feat = torch.cat([s, c], dim=1)   # [N,4,F]
        feat = feat.flatten(start_dim=1)  # [N, 4*F]
        return feat


class MatchingHead(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        num_transformer_layers: int = 2,
        coord_num_freqs: int = 6,
        backbone_feat_dim: int = 256,
        use_roi_norm: bool = True,
        scorer_hidden: int = 128,
        return_augmented_logits: bool = True,
    ):
        super().__init__()
        self.return_augmented_logits = return_augmented_logits

        self.fourier = FourierPosEnc(coord_num_freqs)
        coord_in = 4 * coord_num_freqs
        self.coord_proj = nn.Sequential(
            nn.Linear(coord_in, d_model // 2),
            nn.ReLU(inplace=True),
        )

        self.roi_proj = nn.Sequential(
            nn.Linear(backbone_feat_dim, d_model // 2),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_model // 2) if use_roi_norm else nn.Identity(),
        )

        fused_in = d_model
        self.fuse = nn.Sequential(
            nn.Linear(fused_in, d_model),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_model),
        )

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_transformer_layers)

        self.scorer = nn.Sequential(
            nn.Linear(d_model * 2 + d_model + 2, scorer_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(scorer_hidden, 1)
        )

        self.row_dustbin = nn.Sequential(nn.Linear(d_model, 1))
        self.col_dustbin = nn.Sequential(nn.Linear(d_model, 1))

        self.type_emb = nn.Embedding(2, d_model)  # 0: prev, 1: curr

    @staticmethod
    def _bilinear_roi(
        feats_3d: torch.Tensor,      # [B, C, T, H, W]
        coords_norm: torch.Tensor,   # [B, N, 2] in [0,1], (x_norm, y_norm)
        t_embed_id: int
    ) -> torch.Tensor:
        t_id = -1 if t_embed_id == 1 else -2
        fmap = feats_3d[:, :, t_id, :, :]  # [B, C, H, W]

        x = coords_norm[..., 0] * 2 - 1  # [B, N]
        y = coords_norm[..., 1] * 2 - 1  # [B, N]
        grid = torch.stack([x, y], dim=-1).unsqueeze(2)  # [B, N, 1, 2]

        sampled = F.grid_sample(
            fmap, grid, mode="bilinear", align_corners=True
        )  # [B, C, N, 1]

        sampled = sampled.squeeze(-1).transpose(1, 2).contiguous()  # [B, N, C]
        return sampled

    def _encode_particles(
        self,
        feats_3d: torch.Tensor,                # [B, C, T, H, W]
        coords_norm: torch.Tensor,             # [B, N, 2] in [0,1]
        t_embed_id: int                        # 0 prev / 1 curr
    ) -> torch.Tensor:
        device = feats_3d.device
        B, N, _ = coords_norm.shape

        fxy = self.fourier(coords_norm.view(-1, 2))          # [B*N, 4F]
        fxy = self.coord_proj(fxy)                           # [B*N, d_model//2]
        fxy = fxy.view(B, N, -1)                             # [B, N, d_model//2]

        roi = self._bilinear_roi(feats_3d, coords_norm, t_embed_id)  # [B, N, C]
        roi = self.roi_proj(roi)                             # [B, N, d_model//2]

        x = torch.cat([fxy, roi], dim=-1)      # [B, N, fused_in]
        x = self.fuse(x)                                     # [B, N, d_model]

        type_vec = self.type_emb(
            torch.tensor([t_embed_id], device=device)
        ).view(1, 1, -1).expand(B, N, -1)  # [B, N, d_model]
        x = x + type_vec

        return x

    def forward(
        self,
        backbone_feats: torch.Tensor,  # [B, C, T, H, W]
        prev_info: torch.Tensor,  # [B, Np, 2]
        curr_info: torch.Tensor,  # [B, Nc, 2]
    ) -> Dict[str, torch.Tensor]:
        device = backbone_feats.device
        B, _, T, _, _ = backbone_feats.shape
        Np, Nc = prev_info.size(1), curr_info.size(1)

        prev_tok = self._encode_particles(
            feats_3d=backbone_feats,
            coords_norm=prev_info,  # [B, Np, 2]
            t_embed_id=0
        )  # [B, Np, d_model]
        curr_tok = self._encode_particles(
            feats_3d=backbone_feats,
            coords_norm=curr_info,  # [B, Nc, 2]
            t_embed_id=1
        )  # [B, Nc, d_model]

        hs = self.decoder(curr_tok, prev_tok)        # [B, Nc, d]

        pi = prev_info[:, :, None, :]           # [B,Np,1,2]
        pj = curr_info[:, None, :, :]           # [B,1,Nc,2]
        dxy = pj - pi                           # [B,Np,Nc,2]

        prev_e = prev_tok[:, :, None, :].expand(-1, Np, Nc, -1)  # [B,Np,Nc,d]
        curr_e = hs[:, None, :, :].expand(-1, Np, Nc, -1)        # [B,Np,Nc,d]
        abs_diff = torch.abs(curr_e - prev_e)             # [B,Np,Nc,d]

        pair = torch.cat([prev_e, curr_e, abs_diff, dxy], dim=-1)  # [B,Np,Nc,2d+d+2]
        logits_pair = self.scorer(pair).squeeze(-1)                 # [B,Np,Nc]

        row_db = self.row_dustbin(prev_tok).squeeze(-1)    # [B,Np]
        col_db = self.col_dustbin(hs).squeeze(-1)          # [B,Nc]

        logits_aug = logits_pair.new_full((B, Np + 1, Nc + 1), -1e4)
        logits_aug[:, :Np, :Nc] = logits_pair
        logits_aug[:, :Np, Nc] = row_db
        logits_aug[:, Np, :Nc] = col_db

        out = {
            "logits_aug": logits_aug,  # [B, Np+1, Nc+1]
            "row_mask": torch.ones((B, Np), dtype=torch.bool, device=device),
            "col_mask": torch.ones((B, Nc), dtype=torch.bool, device=device),
        }
        return out

    def compute_loss(
        self,
        logits_aug: torch.Tensor,  # [B, Np+1, Nc+1]
        row_target: torch.Tensor,  # [B, Np]
        col_target: torch.Tensor,  # [B, Nc]
        row_mask: Optional[torch.Tensor] = None,  # [B, Np]
        col_mask: Optional[torch.Tensor] = None,  # [B, Nc]
    ) -> torch.Tensor:
        B, Np, Nc = logits_aug.shape
        Np, Nc = Np - 1, Nc - 1

        row_target_masked = row_target.clone()
        row_target_masked[~row_mask] = -1
        row_logits = logits_aug[:, :Np, :]  # [B, Np, Nc+1]
        row_loss = F.cross_entropy(
            row_logits.transpose(1, 2),  # [B, Nc+1, Np]
            row_target_masked,  # [B, Np]
            reduction="none",
            ignore_index=-1,
        ).mean(dim=1)

        col_target_masked = col_target.clone()
        col_target_masked[~col_mask] = -1
        col_logits = logits_aug[:, :, :Nc]
        col_loss = F.cross_entropy(
            col_logits,  # [B, Np+1, Nc]
            col_target_masked,  # [B, Nc]
            reduction="none",
            ignore_index=-1,
        ).mean(dim=1)

        loss = 0.5 * (row_loss + col_loss)
        return loss.mean()

    @torch.no_grad()
    def compute_metrics(
        self,
        logits_aug: torch.Tensor,  # [B, Np+1, Nc+1]
        row_target: torch.Tensor,  # [B, Np]
        col_target: torch.Tensor,  # [B, Nc]
        row_mask: Optional[torch.Tensor] = None,  # [B, Np]
        col_mask: Optional[torch.Tensor] = None,  # [B, Nc]
    ) -> Dict[str, float]:
        B, Np, Nc = logits_aug.shape
        Np, Nc = Np - 1, Nc - 1

        row_acc_list, col_acc_list = [], []
        TP_total = FP_total = FN_total = 0

        row_mask = row_mask if row_mask is not None else torch.ones_like(row_target, dtype=torch.bool)
        col_mask = col_mask if col_mask is not None else torch.ones_like(col_target, dtype=torch.bool)

        for b in range(B):
            logits_b = logits_aug[b]
            row_tgt = row_target[b]
            col_tgt = col_target[b]
            row_m = row_mask[b]
            col_m = col_mask[b]

            row_logits = logits_b[:Np, :]  # [Np, Nc+1]
            col_logits = logits_b[:, :Nc].transpose(0, 1)  # [Nc, Np+1]

            row_pred = row_logits.argmax(dim=-1)  # [Np]
            col_pred = col_logits.argmax(dim=-1)  # [Nc]

            # Accuracy
            row_acc = (row_pred[row_m] == row_tgt[row_m]).float().mean().item()
            col_acc = (col_pred[col_m] == col_tgt[col_m]).float().mean().item()
            row_acc_list.append(row_acc)
            col_acc_list.append(col_acc)

            # Precision / Recall / F1
            dustbin_c = Nc
            dustbin_r = Np

            valid_rows = (row_tgt != -1) & (row_tgt != dustbin_c) & row_m
            TP_r = ((row_pred == row_tgt) & valid_rows).sum().item()
            FP_r = ((row_pred != row_tgt) & (row_pred != dustbin_c)).sum().item()
            FN_r = ((row_pred != row_tgt) & valid_rows).sum().item()

            valid_cols = (col_tgt != -1) & (col_tgt != dustbin_r) & col_m
            TP_c = ((col_pred == col_tgt) & valid_cols).sum().item()
            FP_c = ((col_pred != col_tgt) & (col_pred != dustbin_r)).sum().item()
            FN_c = ((col_pred != col_tgt) & valid_cols).sum().item()

            TP_total += TP_r + TP_c
            FP_total += FP_r + FP_c
            FN_total += FN_r + FN_c

        row_acc = sum(row_acc_list) / B
        col_acc = sum(col_acc_list) / B
        precision = TP_total / (TP_total + FP_total + 1e-8)
        recall = TP_total / (TP_total + FN_total + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            "row_acc": row_acc,
            "col_acc": col_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    @torch.no_grad()
    def assign_greedy(
        self,
        logits_aug: torch.Tensor,      # [Np+1, Nc+1]
        row_mask: torch.Tensor,        # [Np]
        col_mask: torch.Tensor,        # [Nc]
        score_threshold: Optional[float] = None,
    ) -> Tuple[List[Tuple[int,int]], List[int], List[int]]:
        Np, Nc = logits_aug.shape
        Np, Nc = Np - 1, Nc - 1
        score = logits_aug.clone()

        valid_rows = torch.where(row_mask)[0].tolist()
        valid_cols = torch.where(col_mask)[0].tolist()

        candidates = []
        for r in valid_rows + [Np]:
            for c in valid_cols + [Nc]:
                if r == Np and c == Nc:
                    continue
                candidates.append((r, c, score[r, c].item()))
        candidates.sort(key=lambda x: x[2], reverse=True)

        taken_r, taken_c = set(), set()
        matches = []
        for r, c, s in candidates:
            if r < Np and r in taken_r:
                continue
            if c < Nc and c in taken_c:
                continue

            if r < Np and c < Nc:
                if (score_threshold is None) or (s >= score_threshold):
                    matches.append((r, c))
                    taken_r.add(r); taken_c.add(c)
            elif r < Np and c == Nc:
                taken_r.add(r)
            elif r == Np and c < Nc:
                taken_c.add(c)

        prev_unm = [i for i in valid_rows if i not in taken_r]
        curr_unm = [j for j in valid_cols if j not in taken_c]
        return matches, prev_unm, curr_unm

    @torch.no_grad()
    def assign_hungarian(self):
        # TODO implement hungarian assignment
        return