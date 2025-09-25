import torch
import torch.nn as nn
import torch.nn.functional as F

class ProfileEncoder(nn.Module):
    def __init__(self, latent_dim=37, profile_dim=128, bio_out_dim=64, biomech_dim=2048):
        super().__init__()
        self.latent_dim = latent_dim
        self.bio_out_dim = bio_out_dim
        self.biomech_dim = biomech_dim

        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )

        self.biomech_proj = nn.Sequential(
            nn.Linear(biomech_dim, self.bio_out_dim),
            nn.LayerNorm(self.bio_out_dim),
            nn.GELU()
        )

        self.attn_pool = nn.Linear(latent_dim, 1)

        in_dim = 128 + self.bio_out_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, profile_dim),
            nn.LayerNorm(profile_dim)
        )

    def forward(self, latents, biomech=None):
        if latents.dim() == 3:
            attn_logits = self.attn_pool(latents)
            attn_weights = torch.softmax(attn_logits, dim=1)
            x = (latents * attn_weights).sum(dim=1)
        else:
            x = latents 

        x_latent = self.latent_proj(x)


        if biomech is not None:
            x_bio = self.biomech_proj(biomech)
        else:
            x_bio = torch.zeros(x_latent.size(0), self.bio_out_dim,
                                device=x_latent.device, dtype=x_latent.dtype)

        x_cat = torch.cat([x_latent, x_bio], dim=-1)
        profile = self.head(x_cat)
        return profile

class ProfileDecoder(nn.Module):
    def __init__(self, profile_dim, codebook_dim, seq_len, num_actions, n_layers=2, n_heads=4):
        super().__init__()
        self.seq_len = seq_len
        self.codebook_dim = codebook_dim
        self.profile_proj = nn.Linear(profile_dim, codebook_dim)
        self.action_proj = nn.Linear(num_actions, codebook_dim)
        self.pos_emb = nn.Parameter(torch.randn(seq_len, codebook_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=codebook_dim, nhead=n_heads)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(codebook_dim, codebook_dim)

    def forward(self, profile, action_onehot):
        B = profile.size(0)
        profile_emb = self.profile_proj(profile)
        action_emb = self.action_proj(action_onehot)
        combined = profile_emb + action_emb  # (B, codebook_dim)
        combined = combined.unsqueeze(1)
        tgt = self.pos_emb.unsqueeze(0).repeat(B, 1, 1)
        tgt = tgt + combined
        tgt = tgt.transpose(0, 1)
        memory = combined.transpose(0, 1)
        out = self.transformer(tgt, memory)
        out = self.out_proj(out)
        out = out.transpose(0, 1)
        return out