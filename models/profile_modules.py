import torch
import torch.nn as nn
import torch.nn.functional as F

class ProfileEncoder(nn.Module):
    def __init__(self, latent_dim=37, profile_dim=128, bio_out_dim=64, biomech_dim=2048, metadata_dim=128, metadata_out_dim=128):
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

        self.metadata_proj = nn.Sequential(
            nn.Linear(metadata_dim, self.metadata_out_dim),
            nn.LayerNorm(self.metadata_out_dim),
            nn.GELU()
        )

        self.attn_pool = nn.Linear(latent_dim, 1)

        in_dim = 128 + self.bio_out_dim + self.metadata_out_dim
        self.head = nn.Sequential(
            nn.Linear(in_dim, profile_dim),
            nn.LayerNorm(profile_dim)
        )

    def forward(self, latents, biomech=None, metadata=None):
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

        if metadata is not None:
            x_metadata = self.metadata_proj(metadata)
        else:
            x_metadata = torch.zeros(x_latent.size(0), self.metadata_out_dim,
                                    device=x_latent.device, dtype=x_latent.dtype)

        x_cat = torch.cat([x_latent, x_bio, x_metadata], dim=-1)
        profile = self.head(x_cat)
        return profile

class ProfileActionToMotionTransformer(nn.Module):
    def __init__(self, profile_dim, codebook_dim, seq_len, action_emb_dim, n_layers=2, n_heads=4):
        super().__init__()
        self.seq_len = seq_len
        self.codebook_dim = codebook_dim
        # small projection with activation and dropout to reduce overfitting
        self.profile_proj = nn.Sequential(
            nn.Linear(profile_dim, codebook_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        # action_emb_dim is the dimensionality of a learned action embedding
        # (we expect an nn.Embedding produces these vectors). Project that
        # into the codebook latent space.
        self.action_proj = nn.Sequential(
            nn.Linear(action_emb_dim, codebook_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        # fusion MLP with residual connection + LayerNorm
        self.fusion = nn.Sequential(
            nn.Linear(codebook_dim * 2, codebook_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(codebook_dim, codebook_dim),
        )
        self.fusion_ln = nn.LayerNorm(codebook_dim)
        self.pos_emb = nn.Parameter(torch.randn(seq_len, codebook_dim))
        # increase dropout in transformer layers to help regularize
        decoder_layer = nn.TransformerDecoderLayer(d_model=codebook_dim, nhead=n_heads, dropout=0.2)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(codebook_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(codebook_dim, codebook_dim),
            nn.Dropout(0.2)
        )

    def forward(self, profile, action_emb):
        """
        Args:
            profile: (B, profile_dim)
            action_emb: (B, action_emb_dim) - learned action embeddings (not one-hot)
        """
        B = profile.size(0)
        profile_emb = self.profile_proj(profile)
        # project incoming action embedding into codebook space
        action_emb = self.action_proj(action_emb)
        # Fuse profile and action embeddings with residual + layernorm
        fused = self.fusion(torch.cat([profile_emb, action_emb], dim=-1))
        combined = self.fusion_ln(fused + profile_emb)
        combined = combined.unsqueeze(1)  # (B, 1, codebook_dim)
        
        # Create target sequence with positional embeddings
        tgt = self.pos_emb.unsqueeze(0).repeat(B, 1, 1)  # (B, seq_len, codebook_dim)
        tgt = tgt + combined  
        
        # Transformer expects (seq_len, B, dim)
        tgt = tgt.transpose(0, 1)
        memory = combined.transpose(0, 1)
        
        out = self.transformer(tgt, memory)  # (seq_len, B, codebook_dim)
        out = out.transpose(0, 1)  # (B, seq_len, codebook_dim)
        out = self.out_proj(out)  # Apply output projection to the final sequence
        return out