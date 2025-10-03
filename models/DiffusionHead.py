import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleBoxDenoiser(nn.Module):
    def __init__(self, d_model=256, n_layers=4, n_heads=8):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.input_proj = nn.Linear(4, d_model)
        self.cond_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, 4)

    def forward(self, noisy_boxes, t_embed, image_token):
        # noisy_boxes: (B, Nq, 4)
        x = self.input_proj(noisy_boxes) + t_embed.unsqueeze(1)
        # condition: add image_token to each token
        cond = self.cond_proj(image_token).unsqueeze(0)
        x = x + cond
        x = x.permute(1,0,2)  # (Nq, B, d)
        x = self.encoder(x)
        x = x.permute(1,0,2)  # (B,Nq,d)
        out = self.output_proj(x)
        return out
