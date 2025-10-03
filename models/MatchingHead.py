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

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
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
        max_frames: int = 1024,
        frame_emb_dim: int = 32,
        other_feat_dim: int = 0,
        other_proj_dim: int = 32,
        backbone_feat_dim: int = 256,
        use_roi_norm: bool = True,
        scorer_hidden: int = 128,
        return_augmented_logits: bool = True,
    ):
        super().__init__()
        self.return_augmented_logits = return_augmented_logits

        # 1) Fourier -> Linear
        self.fourier = FourierPosEnc(coord_num_freqs)
        coord_in = 4 * coord_num_freqs
        self.coord_proj = nn.Sequential(
            nn.Linear(coord_in, d_model // 2),
            nn.ReLU(inplace=True),
        )

        # 2) frame id embedding
        self.frame_emb = nn.Embedding(max_frames, frame_emb_dim)

        # 3) other feature projection
        self.use_other = other_feat_dim > 0
        if self.use_other:
            self.other_proj = nn.Sequential(
                nn.Linear(other_feat_dim, other_proj_dim),
                nn.ReLU(inplace=True),
            )
        else:
            other_proj_dim = 0

        # 4) backbone ROI
        self.roi_proj = nn.Sequential(
            nn.Linear(backbone_feat_dim, d_model // 2),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_model // 2) if use_roi_norm else nn.Identity(),
        )

        # 5) fusion
        fused_in = (d_model // 2) + frame_emb_dim + other_proj_dim + (d_model // 2)
        self.fuse = nn.Sequential(
            nn.Linear(fused_in, d_model),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_model),
        )

        # 6) Transformer
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_transformer_layers)

        # 7) pairwise MLP
        self.scorer = nn.Sequential(
            nn.Linear(d_model * 2 + d_model + 2, scorer_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(scorer_hidden, 1)
        )

        # 8) dustbin
        self.row_dustbin = nn.Sequential(nn.Linear(d_model, 1))
        self.col_dustbin = nn.Sequential(nn.Linear(d_model, 1))

        # frame embedding (prev/curr)
        self.type_emb = nn.Embedding(2, d_model)  # 0: prev, 1: curr

    @staticmethod
    def _bilinear_roi_from_3d(
        feats_3d: torch.Tensor,
        b_idx: int,
        t_idx: int,
        coords_norm: torch.Tensor,
    ) -> torch.Tensor:
        C, T, H, W = feats_3d.shape[1], feats_3d.shape[2], feats_3d.shape[3], feats_3d.shape[4]
        assert 0 <= t_idx < T, "t_idx 越界"
        fmap = feats_3d[b_idx : b_idx + 1, :, t_idx]  # [1, C, H, W]

        x = coords_norm[:, 0] * 2 - 1
        y = coords_norm[:, 1] * 2 - 1
        grid = torch.stack([x, y], dim=-1).view(1, -1, 1, 2)

        sampled = F.grid_sample(
            fmap, grid, mode="bilinear", align_corners=True
        )  # [1, C, N, 1]
        sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1)
        return sampled

    def _encode_particles(
        self,
        feats_3d: torch.Tensor,
        b_idx: int,
        frame_id: int,
        coords_norm: torch.Tensor,
        other_feat: Optional[torch.Tensor],
        t_embed_id: int
    ) -> torch.Tensor:
        device = feats_3d.device
        N = coords_norm.size(0)

        fxy = self.fourier(coords_norm)
        fxy = self.coord_proj(fxy)

        frame_vec = self.frame_emb(torch.tensor([frame_id], device=device)).expand(N, -1)

        if self.use_other and other_feat is not None and other_feat.numel() > 0:
            o = self.other_proj(other_feat.to(device))
        else:
            o = torch.zeros(N, 0, device=device)

        roi = self._bilinear_roi_from_3d(feats_3d, b_idx, frame_id, coords_norm)
        roi = self.roi_proj(roi)

        x = torch.cat([fxy, frame_vec, o, roi], dim=-1)
        x = self.fuse(x)

        type_vec = self.type_emb(torch.tensor([t_embed_id], device=device)).expand(N, -1)
        x = x + type_vec
        return x

    def forward(
        self,
        backbone_feats: torch.Tensor,
        prev_info: Dict,
        curr_info: Dict,
        b_idx: int = 0,
    ) -> Dict[str, torch.Tensor]:
        device = backbone_feats.device

        prev_xy = prev_info["coords"].to(device)          # [Np,2]
        curr_xy = curr_info["coords"].to(device)          # [Nc,2]
        Np, Nc = prev_xy.size(0), curr_xy.size(0)

        prev_other = prev_info.get("other_feat", None)
        curr_other = curr_info.get("other_feat", None)
        prev_fid = int(prev_info["frame_id"])
        curr_fid = int(curr_info["frame_id"])

        prev_tok = self._encode_particles(backbone_feats, b_idx, prev_fid, prev_xy, prev_other, t_embed_id=0)  # [Np,d]
        curr_tok = self._encode_particles(backbone_feats, b_idx, curr_fid, curr_xy, curr_other, t_embed_id=1)  # [Nc,d]

        mem = prev_tok.unsqueeze(0)        # [1,Np,d]
        qry = curr_tok.unsqueeze(0)        # [1,Nc,d]
        hs = self.decoder(qry, mem)        # [1,Nc,d]
        hs = hs.squeeze(0)                 # [Nc,d]

        pi = prev_xy[:, None, :]           # [Np,1,2]
        pj = curr_xy[None, :, :]           # [1,Nc,2]
        dxy = pj - pi                      # [Np,Nc,2]

        prev_e = prev_tok[:, None, :].expand(Np, Nc, -1)  # [Np,Nc,d]
        curr_e = hs[None, :, :].expand(Np, Nc, -1)
        abs_diff = torch.abs(curr_e - prev_e)

        pair = torch.cat([prev_e, curr_e, abs_diff, dxy], dim=-1)
        logits_pair = self.scorer(pair).squeeze(-1)

        row_db = self.row_dustbin(prev_tok).squeeze(-1)
        col_db = self.col_dustbin(hs).squeeze(-1)

        logits_aug = logits_pair.new_full((Np + 1, Nc + 1), -1e4)
        logits_aug[:Np, :Nc] = logits_pair
        logits_aug[:Np, Nc] = row_db
        logits_aug[Np, :Nc] = col_db
        logits_aug[Np, Nc] = -1e4

        out = {
            "logits_aug": logits_aug,
            "row_mask": torch.ones(Np, dtype=torch.bool, device=device),
            "col_mask": torch.ones(Nc, dtype=torch.bool, device=device),
        }
        return out

    @torch.no_grad()
    def assign_greedy(
        self,
        logits_aug: torch.Tensor,
        row_mask: torch.Tensor,
        col_mask: torch.Tensor,
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

    def compute_loss(
        self,
        logits_aug: torch.Tensor,
        row_target: torch.Tensor,
        col_target: torch.Tensor,
    ) -> torch.Tensor:
        Np, Nc = logits_aug.shape
        Np, Nc = Np - 1, Nc - 1

        row_logits = logits_aug[:Np, :]
        row_loss = F.cross_entropy(row_logits, row_target, reduction="mean")

        col_logits = logits_aug[:, :Nc].transpose(0, 1)
        col_loss = F.cross_entropy(col_logits, col_target, reduction="mean")

        return 0.5 * (row_loss + col_loss)

    def compute_metrics(
            self,
            logits_aug,
            row_target,
            col_target
    ):
        Np1, Nc1 = logits_aug.shape
        Np, Nc = Np1 - 1, Nc1 - 1

        row_logits = logits_aug[:Np, :]
        col_logits = logits_aug[:, :Nc].transpose(0, 1)

        row_pred = row_logits.argmax(dim=-1)
        col_pred = col_logits.argmax(dim=-1)

        row_acc = (row_pred == row_target).float().mean().item()
        col_acc = (col_pred == col_target).float().mean().item()

        dustbin_c = Nc
        dustbin_r = Np

        valid_rows = row_target != dustbin_c
        TP_r = ((row_pred == row_target) & valid_rows).sum().item()
        FP_r = ((row_pred != row_target) & (row_pred != dustbin_c)).sum().item()
        FN_r = ((row_pred != row_target) & valid_rows).sum().item()

        valid_cols = col_target != dustbin_r
        TP_c = ((col_pred == col_target) & valid_cols).sum().item()
        FP_c = ((col_pred != col_target) & (col_pred != dustbin_r)).sum().item()
        FN_c = ((col_pred != col_target) & valid_cols).sum().item()

        TP = TP_r + TP_c
        FP = FP_r + FP_c
        FN = FN_r + FN_c

        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            "row_acc": row_acc,
            "col_acc": col_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
