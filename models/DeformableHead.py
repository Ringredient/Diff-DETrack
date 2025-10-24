import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment


class SinePositionalEncoding2D(nn.Module):
    def __init__(self, d_model, temperature=10000):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        assert d_model % 4 == 0, "d_model must be divisible by 4 for 2D sine"

    def forward(self, mask):
        if mask is None:
            raise ValueError("mask cannot be None")
        not_mask = ~mask  # valid positions True
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * math.pi
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * math.pi

        dim_t = torch.arange(self.d_model // 4, dtype=torch.float32, device=mask.device)  # d_model//4
        dim_t = self.temperature ** (2 * (dim_t // 2) / (self.d_model // 4))  # d_model//4
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3)  # (B,H,W,d_model)
        return pos


class SimpleFPN(nn.Module):
    def __init__(self, in_channels, out_channels, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.lateral = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1) for _ in scales
        ])
        self.smooth = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in scales
        ])

    def forward(self, x):
        feats = []
        for s_idx, s in enumerate(self.scales):
            if s == 1:
                xi = x
            else:
                xi = F.avg_pool2d(x, kernel_size=s, stride=s, ceil_mode=True)
            xi = self.lateral[s_idx](xi)
            xi = self.smooth[s_idx](xi)
            feats.append(xi)
        return feats


def sample_feature_grid(
    feat: torch.Tensor,
    ref_points: torch.Tensor,
    patch_size: int = 3
):
    B, C, H, W = feat.shape
    B2, nQ, nP, _ = ref_points.shape
    assert B == B2
    device = feat.device

    half = (patch_size - 1) / 2.0
    offsets = torch.linspace(-half, half, steps=patch_size, device=device)
    oy, ox = torch.meshgrid(offsets, offsets, indexing='ij')
    ox = ox[None, None, None, :, :].expand(B, nQ, nP, patch_size, patch_size)
    oy = oy[None, None, None, :, :].expand(B, nQ, nP, patch_size, patch_size)

    px = ref_points[..., 0].unsqueeze(-1).unsqueeze(-1)
    py = ref_points[..., 1].unsqueeze(-1).unsqueeze(-1)
    off_x_norm = ox * (2.0 / W)
    off_y_norm = oy * (2.0 / H)
    grid_x = px * 2.0 - 1.0 + off_x_norm
    grid_y = py * 2.0 - 1.0 + off_y_norm
    grid = torch.stack((grid_x, grid_y), dim=-1)

    grid_flat = grid.view(B * nQ * nP, patch_size, patch_size, 2)
    feat_rep = feat.unsqueeze(1).unsqueeze(1).expand(B, nQ, nP, C, H, W)
    feat_rep = feat_rep.contiguous().view(B * nQ * nP, C, H, W)
    sampled = F.grid_sample(feat_rep, grid_flat, mode='bilinear', padding_mode='border', align_corners=True)
    sampled = sampled.view(B, nQ, nP, C, patch_size, patch_size).permute(0,1,2,4,5,3)
    return sampled


def box_cxcywh_to_xyxy(x):
    # (cx,cy,w,h) -> (x1,y1,x2,y2)
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)


def generalized_box_iou(boxes1, boxes2):
    """
    IoU + enclosing box
    :param boxes1: [N1,4]
    :param boxes2: [N2,4]
    :return: [N1, N2] GIoU matrix
    """
    boxes1 = boxes1[:, None, :]
    boxes2 = boxes2[None, :, :]

    inter_x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    inter_y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
    inter_x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
    inter_y2 = torch.min(boxes1[..., 3], boxes2[..., 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union = area1 + area2 - inter
    iou = inter / (union + 1e-6)

    # enclosing box
    enclose_x1 = torch.min(boxes1[..., 0], boxes2[..., 0])
    enclose_y1 = torch.min(boxes1[..., 1], boxes2[..., 1])
    enclose_x2 = torch.max(boxes1[..., 2], boxes2[..., 2])
    enclose_y2 = torch.max(boxes1[..., 3], boxes2[..., 3])
    enclose = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)

    giou = iou - (enclose - union) / (enclose + 1e-6)
    return giou


class QueryCrossAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, n_points=4, patch_size=3):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points
        self.patch_size = patch_size
        assert d_model % n_heads == 0
        self.head_dim = d_model // n_heads

        # projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, d_model * 2)
        self.out_proj = nn.Linear(d_model, d_model)

        # small MLP to aggregate sampled patch -> feature vector (per point)
        self.point_mlp = nn.Sequential(
            nn.Linear(self.patch_size * self.patch_size * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(
        self, query: torch.Tensor,
        sampled_feats: torch.Tensor,
    ) -> torch.Tensor:
        B, nQ, d = query.shape
        B2, nQ2, nP, pH, pW, C = sampled_feats.shape
        assert B == B2 and nQ == nQ2 and C == d
        sampled_flat = sampled_feats.view(B, nQ, nP, -1)
        point_vec = self.point_mlp(sampled_flat)

        kv = point_vec.mean(dim=2)
        q = self.q_proj(query)
        k = kv
        v = kv
        attn = torch.einsum('bqd,bkd->bqk', q, k) / math.sqrt(d)
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('bqk,bkd->bqd', attn, v)
        out = self.out_proj(out)
        return query + out


class DeformableHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_queries: int = 200,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        n_points: int = 8,
        patch_size: int = 3
    ):
        super().__init__()
        self.d_model = d_model
        self.n_queries = n_queries
        self.n_points = n_points
        self.patch_size = patch_size

        # --- weights ---
        # losses
        self.noobj_coef = 0.1
        self.lambda_cls = 1.0
        self.lambda_box = 5.0
        self.lambda_giou = 2.0
        # metrics
        self.iou_thresh = 0.5,
        self.conf_thresh = 0.5
        # inference
        self.logit_threshold = 0.5

        # backbone: lightweight CNN to extract feature map
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True)
        )

        # positional encoding
        self.pos_enc = SinePositionalEncoding2D(d_model)

        # multi-scale FPN
        self.fpn = SimpleFPN(d_model, d_model, scales=[1, 2, 4])

        # query embeddings
        self.query_embed = nn.Embedding(n_queries, d_model)
        self.query_pos = nn.Embedding(n_queries, d_model)

        # decoder layers: each layer refines query boxes + features using sampling
        self.decoder_layers = nn.ModuleList([
            QueryCrossAttentionLayer(d_model, n_heads, n_points=n_points, patch_size=patch_size)
            for _ in range(n_layers)
        ])
        # box & class heads
        self.class_head = nn.Linear(d_model, 1)
        self.box_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4),
            nn.Sigmoid()
        )

        # small MLP to predict offsets (relative) for sampling reference points per query
        self.refpoint_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_points * 2),
            nn.Tanh()
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        if x.ndim == 3:
            x = x.unsqueeze(1)

        B, C, H, W = x.shape
        x = self.backbone(x)
        _, _, Hf, Wf = x.shape

        if mask is None:
            mask = torch.zeros((B, Hf, Wf), dtype=torch.bool, device=x.device)
        pos = self.pos_enc(mask)
        pos = pos.permute(0,3,1,2)
        x = x + pos

        feats = self.fpn(x)

        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        qpos = self.query_pos.weight.unsqueeze(0).expand_as(queries)
        queries = queries + qpos

        pred_boxes = torch.rand(B, self.n_queries, 4, device=x.device)
        for layer in self.decoder_layers:
            ref_offsets = self.refpoint_mlp(queries)
            ref_offsets = ref_offsets.view(B, self.n_queries, self.n_points, 2) * 0.5
            base = pred_boxes[..., :2].unsqueeze(2)  # (B,nQ,1,2)
            ref_points = (base + ref_offsets).clamp(0.0, 1.0)

            sampled_per_scale = []
            for feat_map in feats:
                sampled = sample_feature_grid(feat_map, ref_points, patch_size=self.patch_size)
                sampled_per_scale.append(sampled)
            sampled_all = torch.cat(sampled_per_scale, dim=2)

            if sampled_all.shape[2] != self.n_points:
                sampled_all = sampled_all[:, :, :self.n_points]

            queries = layer(queries, sampled_all)

            delta = self.box_head(queries)
            pred_boxes = (pred_boxes + 0.1 * (delta - 0.5).tanh())
            pred_boxes = pred_boxes.clamp(0.0, 1.0)

        logits = self.class_head(queries).squeeze(-1)
        return {'logits': logits, 'boxes': pred_boxes}

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        pred_logits = outputs["logits"]
        pred_boxes = outputs["boxes"]
        B, N_pred = pred_logits.shape



        total_cls = torch.tensor(0.0, device=pred_logits.device)
        total_l1 = torch.tensor(0.0, device=pred_logits.device)
        total_giou = torch.tensor(0.0, device=pred_logits.device)
        for b in range(B):
            mask_pred = (pred_boxes[b].abs().sum(dim=-1) > 0)
            valid_pred_logits = pred_logits[b][mask_pred]  # [N_pred]
            valid_pred_boxes = pred_boxes[b][mask_pred]  # [N_pred, 4]

            tgt_logits = targets[b]["logits"]  # [N_tgt]
            tgt_boxes = targets[b]["boxes"]  # [N_tgt, 4]
            N_tgt = tgt_logits.size(0)

            if N_tgt == 0:
                continue

            with torch.no_grad():
                cls_cost = F.binary_cross_entropy_with_logits(
                    valid_pred_logits.unsqueeze(1).expand(-1, N_tgt),
                    tgt_logits.unsqueeze(0).expand(N_pred, -1),
                    reduction='none'
                )

                # L1 cost
                l1_cost = torch.cdist(valid_pred_boxes, tgt_boxes, p=1)

                # GIoU cost
                giou = generalized_box_iou(
                    box_cxcywh_to_xyxy(valid_pred_boxes),
                    box_cxcywh_to_xyxy(tgt_boxes)
                )

                # cost matrix
                cost_matrix = (
                    self.lambda_cls * cls_cost
                    + self.lambda_box * l1_cost
                    + self.lambda_giou * (1 - giou)
                ).detach().cpu()

                row_idx, col_idx = linear_sum_assignment(cost_matrix)

            matched_pred_logits = valid_pred_logits[row_idx]
            matched_pred_boxes = valid_pred_boxes[row_idx]
            matched_tgt_logits = tgt_logits[col_idx]
            matched_tgt_boxes = tgt_boxes[col_idx]

            bce_loss = F.binary_cross_entropy_with_logits(
                matched_pred_logits,
                matched_tgt_logits.float(),
                reduction="none"
            )
            weight = torch.where(matched_tgt_logits > 0, 1.0, self.noobj_coef)
            cls_loss = (bce_loss * weight).mean()

            # L1 Loss
            L_box = F.l1_loss(matched_pred_boxes, matched_tgt_boxes, reduction="mean")

            # GIoU Loss
            giou = generalized_box_iou(
                box_cxcywh_to_xyxy(matched_pred_boxes),
                box_cxcywh_to_xyxy(matched_tgt_boxes)
            )
            L_giou = (1 - giou.diag()).mean()

            total_cls += cls_loss
            total_l1 += L_box
            total_giou += L_giou

        loss = (
            self.lambda_cls * total_cls
            + self.lambda_box * total_l1
            + self.lambda_giou * total_giou
        ) / B
        return loss

    @torch.no_grad()
    def compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> dict:
        pred_logits = torch.sigmoid(outputs["logits"])
        pred_boxes = outputs["boxes"]
        B, nQ = pred_logits.shape

        tp, fp, fn = 0, 0, 0
        iou_all = []
        ap_list = []

        for b in range(B):
            conf = pred_logits[b]
            boxes = box_cxcywh_to_xyxy(pred_boxes[b])

            tgt = targets[b]
            tgt_boxes = box_cxcywh_to_xyxy(tgt["boxes"])
            tgt_labels = tgt["logits"]

            valid_idx = conf > self.conf_thresh
            boxes = boxes[valid_idx]
            conf = conf[valid_idx]

            if len(tgt_boxes) == 0:
                fp += len(boxes)
                continue
            if len(boxes) == 0:
                fn += len(tgt_boxes)
                continue

            iou = box_iou(boxes, tgt_boxes)
            max_iou, max_idx = iou.max(dim=1)

            matched = torch.zeros(len(tgt_boxes), dtype=torch.bool, device=boxes.device)
            for i, iou_val in enumerate(max_iou):
                j = max_idx[i]
                if iou_val > self.iou_thresh and not matched[j]:
                    tp += 1
                    matched[j] = True
                    iou_all.append(iou_val.item())
                else:
                    fp += 1
            fn += (~matched).sum().item()

            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            ap = precision * recall  # simplified AP50 estimate
            ap_list.append(ap)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        mean_iou = torch.tensor(iou_all).mean().item() if len(iou_all) else 0.0
        mAP50 = torch.tensor(ap_list).mean().item() if len(ap_list) else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_iou": mean_iou,
            "mAP50": mAP50,
            "tp": tp,
            "fp": fp,
            "fn": fn
        }

    @torch.no_grad()
    def hungarian_match(
        self,
        pred_boxes: torch.Tensor,
        tgt_boxes: torch.Tensor,
        cost_type: str = 'l1',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if pred_boxes.numel() == 0 or tgt_boxes.numel() == 0:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

        # cost matrix
        if cost_type == 'l1':
            cost_matrix = torch.cdist(pred_boxes, tgt_boxes, p=1).cpu().numpy()
        elif cost_type == 'giou':
            giou = generalized_box_iou(pred_boxes, tgt_boxes).cpu().numpy()
            cost_matrix = 1 - giou
        else:
            raise ValueError(f"Unknown cost type: {cost_type}")

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return torch.as_tensor(row_ind, dtype=torch.long), torch.as_tensor(col_ind, dtype=torch.long)

    def inference(self, model_output):
        curr_info_list = []
        for batch in model_output:
            logits = batch["logits"]
            boxes = batch["boxes"]
            mask = logits > self.logit_threshold

            selected_boxes = boxes[mask]
            if selected_boxes.numel() == 0:
                curr_info_list.append(torch.zeros((0, 2), device=boxes.device))
            else:
                curr_info_list.append(selected_boxes[:, :2])

        curr_info = nn.utils.rnn.pad_sequence(curr_info_list, batch_first=True, padding_value=0.0)
        return curr_info
