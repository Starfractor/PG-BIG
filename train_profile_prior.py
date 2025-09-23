import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
from dataset.dataset_183_retarget import retargeted183_data_loader
from dataset import dataset_183
from dataset import dataset_addb
from models import vqvae
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

loader = retargeted183_data_loader(data_dir='/home/mnt/datasets/183_retargeted', num_workers=12, pre_load=True)


import torch
from models import vqvae

# Load VQ-VAE Model
class Args:
    dataname = "mcs"
    nb_code = 512
    code_dim = 512
    output_emb_width = 512
    down_t = 2
    stride_t = 2
    width = 512
    depth = 3
    dilation_growth_rate = 3
    vq_act = "relu"
    vq_norm = None
    quantizer = "ema_reset"
    mu = 0.99
    nb_joints = 37


args = Args()

# Instantiate the model
model = vqvae.HumanVQVAE(
    args,
    nb_code=args.nb_code,
    code_dim=args.code_dim,
    output_emb_width=args.output_emb_width,
    down_t=args.down_t,
    stride_t=args.stride_t,
    width=args.width,
    depth=args.depth,
    dilation_growth_rate=args.dilation_growth_rate,
    activation=args.vq_act,
    norm=args.vq_norm
)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

ckpt = torch.load("/home/mnt/code/nicholas/output/183_training/70000.pth", map_location=device)

# Remove 'module.' prefix if present
state_dict = ckpt['net']
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace('module.', '') if k.startswith('module.') else k
    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict, strict=True)
model.eval()

encoder = model.vqvae.encoder

from tqdm import tqdm

# Use VQ-VAE encoder to encode motions
subject_latents = defaultdict(list)
with torch.no_grad():
    for i, batch in enumerate(tqdm(loader, desc="Encoding Motions")):
        try:
            motions, _, _, subject_names = batch
        except ValueError as e:
            print(f"Skipping batch due to error: {e}")
            continue
        except StopIteration:
            break
        motions = motions.to(device, dtype=torch.float32)
        latents = model.vqvae.encoder(model.vqvae.preprocess(motions))
        for latent, subject_name in zip(latents, subject_names):
            subject_latents[subject_name].append(latent.cpu())

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

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

# --- Perform Training Loop ---
import warnings
rng = np.random.RandomState(42)

# Subject selection
_all_subjects = sorted([s for s in subject_latents.keys() if len(subject_latents[s]) > 0])
n_test = 33
n_train_target = 150
if len(_all_subjects) < 1:
    raise RuntimeError("No subjects with motions found in subject_latents.")

if len(_all_subjects) <= n_test:
    raise RuntimeError(f"Not enough subjects ({len(_all_subjects)}) to reserve {n_test} for validation.")

test_subjects = list(rng.choice(_all_subjects, size=min(n_test, len(_all_subjects)), replace=False))
_remaining = [s for s in _all_subjects if s not in test_subjects]
train_candidates = [s for s in _remaining if len(subject_latents[s]) >= 2]

if len(train_candidates) < n_train_target:
    warnings.warn(f"Only {len(train_candidates)} subjects have >=2 motions; will use all of them for training.")
    train_subjects = train_candidates
else:
    rng.shuffle(train_candidates)
    train_subjects = train_candidates[:n_train_target]

subjects = list(train_subjects)
print(f"Using {len(subjects)} subjects for training, reserving {len(test_subjects)} subjects for validation.")

# Device and model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 16
profile_dim = 512
dataset = loader.dataset
biomech_dim = getattr(dataset, "default_biomech_dim", 2048)

num_epochs = 1000
batch_size = 512

encoder = ProfileEncoder(
    latent_dim=latent_dim,
    profile_dim=profile_dim,
    bio_out_dim=64,
    biomech_dim=biomech_dim
).to(device)
optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-5, weight_decay=1e-4)

# Scheduler: linear warmup then cosine
warmup_epochs = 100
main_epochs = max(1, num_epochs * 2 - warmup_epochs)
scheduler = SequentialLR(
    optimizer,
    schedulers=[
        LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
        CosineAnnealingLR(optimizer, T_max=main_epochs, eta_min=1e-6)
    ],
    milestones=[warmup_epochs],
)

loss_fn = nn.TripletMarginLoss()

def get_biomech_vector(subject):
    bm = dataset.get_biomech(subject)
    if isinstance(bm, dict):
        vec = bm.get('vector', None)
        if vec is None:
            vec = dataset._biomech_to_vector(
                bm,
                target_samples=dataset.default_biomech_target_samples,
                biomech_dim=dataset.default_biomech_dim
            )
    else:
        bm2 = dataset.get_biomech(subject)
        vec = bm2.get('vector', np.zeros(biomech_dim, dtype=np.float32))
    vec = np.asarray(vec, dtype=np.float32)
    if vec.shape[0] != biomech_dim:
        if vec.shape[0] < biomech_dim:
            vec = np.pad(vec, (0, biomech_dim - vec.shape[0]), mode='constant')
        else:
            vec = vec[:biomech_dim]
    return vec

def sample_triplet(subjects, subject_latents, biomech_dim, device):
    subject_pos = np.random.choice(subjects)
    latents_pos = subject_latents[subject_pos]
    if len(latents_pos) < 2:
        return None
    idx = np.random.choice(len(latents_pos), 2, replace=False)
    anchor = latents_pos[idx[0]].unsqueeze(0).to(device)
    positive = latents_pos[idx[1]].unsqueeze(0).to(device)
    negative_subjects = [s for s in subjects if s != subject_pos and len(subject_latents[s]) > 0]
    if not negative_subjects:
        return None
    subject_neg = np.random.choice(negative_subjects)
    latents_neg = subject_latents[subject_neg]
    negative = latents_neg[np.random.choice(len(latents_neg))].unsqueeze(0).to(device)
    a_bio = get_biomech_vector(subject_pos)
    p_bio = a_bio.copy()
    n_bio = get_biomech_vector(subject_neg)
    return anchor, positive, negative, a_bio, p_bio, n_bio

def create_batch(batch_size, subjects, subject_latents, biomech_dim, device):
    anchors, positives, negatives = [], [], []
    anchor_bios, positive_bios, negative_bios = [], [], []
    for _ in range(batch_size):
        result = sample_triplet(subjects, subject_latents, biomech_dim, device)
        if result is None:
            continue
        anchor, positive, negative, a_bio, p_bio, n_bio = result
        anchors.append(anchor)
        positives.append(positive)
        negatives.append(negative)
        anchor_bios.append(a_bio)
        positive_bios.append(p_bio)
        negative_bios.append(n_bio)
    return anchors, positives, negatives, anchor_bios, positive_bios, negative_bios

# Training loop
all_losses = []
for epoch in range(num_epochs):
    if len(subjects) == 0:
        raise RuntimeError("No subjects with >=2 motions found.")
    np.random.shuffle(subjects)

    anchors, positives, negatives, anchor_bios, positive_bios, negative_bios = create_batch(
        batch_size, subjects, subject_latents, biomech_dim, device
    )
    if len(anchors) == 0:
        continue

    anchor_batch = torch.cat(anchors, dim=0)
    positive_batch = torch.cat(positives, dim=0)
    negative_batch = torch.cat(negatives, dim=0)
    anchor_bio_batch = torch.tensor(np.stack(anchor_bios), device=device, dtype=torch.float32)
    positive_bio_batch = torch.tensor(np.stack(positive_bios), device=device, dtype=torch.float32)
    negative_bio_batch = torch.tensor(np.stack(negative_bios), device=device, dtype=torch.float32)

    anchor_emb = F.normalize(encoder(anchor_batch, anchor_bio_batch), dim=1)
    positive_emb = F.normalize(encoder(positive_batch, positive_bio_batch), dim=1)
    negative_emb = F.normalize(encoder(negative_batch, negative_bio_batch), dim=1)

    loss = loss_fn(anchor_emb, positive_emb, negative_emb)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    all_losses.append(loss.item())

    if len(all_losses) % 100 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Batch Loss: {loss.item():.6f}")

if len(all_losses) > 0:
    final_avg = float(np.mean(all_losses[-100:])) if len(all_losses) >= 100 else float(np.mean(all_losses))
    print(f"Final Loss: {all_losses[-1]:.6f}, Recent Avg: {final_avg:.6f}")
else:
    print("No training steps completed.")

# Save Profile Encoder
torch.save(encoder.state_dict(), "profile_encoder.pth")