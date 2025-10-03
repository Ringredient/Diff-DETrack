import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple


class SinePositionalEncoding2D(nn.Module):
    def __init__(self, d_model, temperature=10000):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        assert d_model % 4 == 0, "d_model must be divisible by 4 for 2D sine"

    def forward(self, mask: torch.Tensor):
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
        pos = torch.cat((pos_y, pos_x), dim=3)
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

    def forward(self, x: torch.Tensor):
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


def sample_feature_grid(feat: torch.Tensor, ref_points: torch.Tensor, patch_size: int = 3):

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


def generalized_box_iou(boxes1, boxes2):
    # boxes: [N,4] in xyxy
    inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - inter
    iou = inter / (union + 1e-6)

    # enclosing box
    enclose_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    enclose_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    enclose_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    enclose_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
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

    def forward(self, query: torch.Tensor, sampled_feats: torch.Tensor):
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
    def __init__(self, in_channels: int, d_model: int = 256, n_queries: int = 200,
                 n_heads: int = 8, n_layers: int = 6, n_points: int = 8, patch_size: int = 3):
        super().__init__()
        self.d_model = d_model
        self.n_queries = n_queries
        self.n_points = n_points
        self.patch_size = patch_size

        # temporal fusion
        self.temporal_fuse = nn.Conv3d(in_channels, in_channels, kernel_size=(3,1,1), padding=(1,0,0))
        self.input_proj = nn.Conv2d(in_channels, d_model, kernel_size=1)

        # positional encoding
        self.pos_enc = SinePositionalEncoding2D(d_model)

        # multi-scale FPN
        self.fpn = SimpleFPN(d_model, d_model, scales=[1, 2, 4])

        # query embeddings
        self.query_embed = nn.Embedding(n_queries, d_model)
        self.query_pos = nn.Embedding(n_queries, d_model)

        # decoder layers
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

        # coefficients for loss fn
        self.noobj_coef = 0.1
        self.lambda_box = 1
        self.lambda_giou = 1

    def forward(self, feat: torch.Tensor, mask: torch.Tensor = None):
        B, T, C_in, H, W = feat.shape
        x = self.temporal_fuse(feat)
        x = x.mean(dim=2)
        x = self.input_proj(x)

        if mask is None:
            mask = torch.zeros((B, H, W), dtype=torch.bool, device=x.device)
        pos = self.pos_enc(mask)
        pos = pos.permute(0,3,1,2)
        x = x + pos

        feats = self.fpn(x)

        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1).contiguous()
        qpos = self.query_pos.weight.unsqueeze(0).expand(B, -1, -1)
        queries = queries + qpos

        device = x.device
        pred_boxes = torch.rand(B, self.n_queries, 4, device=device)
        for layer in self.decoder_layers:
            ref_offsets = self.refpoint_mlp(queries)
            ref_offsets = ref_offsets.view(B, self.n_queries, self.n_points, 2) * 0.5
            base = pred_boxes[..., :2].unsqueeze(2)  # (B,nQ,1,2)
            ref_points = (base + ref_offsets).clamp(0.0, 1.0)

            sampled_per_scale = []
            for feat_map in feats:
                if feat_map.shape[1] != self.d_model:
                    proj = nn.Conv2d(feat_map.shape[1], self.d_model, kernel_size=1).to(device)
                    feat_map = proj(feat_map)
                sampled = sample_feature_grid(feat_map, ref_points, patch_size=self.patch_size)
                sampled_per_scale.append(sampled)

            sampled_all = torch.cat(sampled_per_scale, dim=2)
            if sampled_all.shape[2] != self.n_points:
                sampled_all = sampled_all[:, :, :self.n_points, ...]

            queries = layer(queries, sampled_all)

            delta = self.box_head(queries)
            pred_boxes = (pred_boxes + 0.1 * (delta - 0.5).tanh())
            pred_boxes = pred_boxes.clamp(0.0, 1.0)

        logits = self.class_head(queries).squeeze(-1)
        return {'pred_logits': logits, 'pred_boxes': pred_boxes}

    def compute_loss(self, outputs, targets):
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        B, nQ, num_classes = pred_logits.shape

        pred_logits = pred_logits.view(B * nQ, num_classes)
        pred_boxes = pred_boxes.view(B * nQ, 4)

        tgt_logits = torch.cat([t["logits"] for t in targets], dim=0)
        tgt_boxes = torch.cat([t["boxes"] for t in targets], dim=0)

        pos_weight = torch.ones([num_classes], device=pred_logits.device)
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_logits.squeeze(-1), tgt_logits.float(),
            pos_weight=pos_weight,
            reduction="none"
        )
        weight = torch.where(tgt_logits > 0, 1.0, self.noobj_coef)
        cls_loss = (bce_loss * weight).mean()

        # L1 loss
        L_box = F.l1_loss(pred_boxes, tgt_boxes, reduction="mean")

        # giou
        giou = generalized_box_iou(
            box_cxcywh_to_xyxy(pred_boxes),
            box_cxcywh_to_xyxy(tgt_boxes)
        )
        L_giou = (1 - giou).mean()

        loss = cls_loss + self.lambda_box * L_box + self.lambda_giou * L_giou
        return {"loss": loss}

    def compute_metrics(self):
        # TODO: metrics
        return
