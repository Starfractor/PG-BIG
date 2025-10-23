import argparse
import matplotlib.pyplot as plt
import numpy as np
import options.option_subject_prior as option_subject_prior
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
from collections import defaultdict
from dataset import dataset_183_retarget
from dataset import dataset_addbiomechanics
from models import profile_modules
from models import vqvae
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm

# --- Moved helper functions outside main for modularity ---

def get_biomech_vector(subject, dataset, biomech_dim):
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

def sample_triplet(subjects, subject_samples, biomech_dim, device, dataset):
    subject_pos = np.random.choice(subjects)
    latents_pos = [l for l, _ in subject_samples[subject_pos]]
    if len(latents_pos) < 2:
        return None
    idx = np.random.choice(len(latents_pos), 2, replace=False)
    anchor = latents_pos[idx[0]].unsqueeze(0).to(device)
    positive = latents_pos[idx[1]].unsqueeze(0).to(device)
    negative_subjects = [s for s in subjects if s != subject_pos and len(subject_samples[s]) > 0]
    if not negative_subjects:
        return None
    subject_neg = np.random.choice(negative_subjects)
    latents_neg = [l for l, _ in subject_samples[subject_neg]]
    negative = latents_neg[np.random.choice(len(latents_neg))].unsqueeze(0).to(device)
    a_bio = get_biomech_vector(subject_pos, dataset, biomech_dim)
    p_bio = a_bio.copy()
    n_bio = get_biomech_vector(subject_neg, dataset, biomech_dim)
    return anchor, positive, negative, a_bio, p_bio, n_bio

def create_batch(batch_size, subjects, subject_samples, biomech_dim, device, dataset):
    anchors, positives, negatives = [], [], []
    anchor_bios, positive_bios, negative_bios = [], [], []
    for _ in range(batch_size):
        result = sample_triplet(subjects, subject_samples, biomech_dim, device, dataset)
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

def compute_subject_profiles(subject_list, subject_samples, encoder, device, dataset, biomech_dim):
    prof_dict = {}
    for subject in subject_list:
        samples = subject_samples[subject]
        if len(samples) == 0:
            continue
        latents_only = [l for l, _ in samples]
        latents_tensor = torch.stack(latents_only).to(device)
        bio_vec = get_biomech_vector(subject, dataset, biomech_dim)
        biomech_tensor = torch.tensor(bio_vec, device=device, dtype=torch.float32).unsqueeze(0).repeat(latents_tensor.shape[0], 1)
        with torch.no_grad():
            try:
                profile_t = encoder(latents_tensor, biomech_tensor)
            except TypeError:
                profile_t = encoder(latents_tensor)
            profile_vectors = profile_t.detach().cpu()
        # Store list of (profile_vector, action_name)
        prof_dict[subject] = list(zip(profile_vectors, [name for _, name in samples]))
    return prof_dict

def get_action_index(action, action_to_idx):
    """Return integer index for action or None if unknown."""
    return action_to_idx.get(action, None)

def compute_loss_and_acc(batch_profiles, batch_action_emb, batch_latents, mapping_net, model, ce_loss, code_embed_dim, lengths=None, use_embedding_loss=False):
    """
    Compute loss and accuracy for the decoder with support for variable-length sequences.
    
    Args:
        batch_profiles: (B, profile_dim) - subject profiles
        batch_action_emb: (B, action_emb_dim) - action embeddings
        batch_latents: (B, code_dim, max_seq_len) - VQ-VAE latents to predict (padded)
        mapping_net: The decoder network
        model: VQ-VAE model containing quantizer
        ce_loss: CrossEntropyLoss
        code_embed_dim: Dimension of codebook embeddings
        lengths: (B,) tensor of actual sequence lengths (before padding)
    """
    # Predict latents: (B, max_seq_len, code_dim)
    pred_latents = mapping_net(batch_profiles, batch_action_emb)
    
    # Move batch_latents to same device
    batch_latents = batch_latents.to(pred_latents.device)
    
    # batch_latents is (B, code_dim, max_seq_len), need to transpose to (B, max_seq_len, code_dim)
    batch_latents = batch_latents.transpose(1, 2)
    
    B, max_seq_len, code_dim = batch_latents.shape
    
    # Flatten for quantization: (B*max_seq_len, code_dim)
    flat_targets = batch_latents.reshape(B * max_seq_len, code_dim)
    
    with torch.no_grad():
        # Quantize to get code indices
        code_indices = model.vqvae.quantizer.quantize(flat_targets)  # (B*max_seq_len,)
        batch_code_indices = code_indices.view(B, max_seq_len)
    
    # Get codebook and compute logits
    codebook = model.vqvae.quantizer.codebook.to(pred_latents.device)  # (num_codes, code_embed_dim)
    # Sanity check: predicted latent dimension must match codebook embedding dim
    if pred_latents.shape[-1] != codebook.shape[1]:
        raise RuntimeError(f"Dimension mismatch: pred_latents dim {pred_latents.shape[-1]} != codebook emb dim {codebook.shape[1]}")
    logits = F.linear(pred_latents, codebook)  # (B, max_seq_len, num_codes)
    
    # Create mask for variable-length sequences
    if lengths is not None:
        # Create mask: (B, max_seq_len)
        mask = torch.arange(max_seq_len, device=pred_latents.device).unsqueeze(0) < lengths.unsqueeze(1)

        if use_embedding_loss:
            # Build target embeddings from code indices and compute MSE per token
            # code_indices is (B*max_seq_len,)
            target_embs_flat = codebook[code_indices.long()].to(pred_latents.device)  # (B*max_seq_len, emb_dim)
            target_embs = target_embs_flat.view(B, max_seq_len, -1)  # (B, max_seq_len, emb_dim)
            # MSE per position (keep per-dim reduction=None to average dims later)
            loss_per_pos = F.mse_loss(pred_latents, target_embs, reduction='none')  # (B, max_seq_len, emb_dim)
            loss_per_token = loss_per_pos.mean(dim=-1)  # (B, max_seq_len)
            masked_loss = (loss_per_token * mask.float()).sum() / mask.float().sum()
            loss = masked_loss

            # Compute masked accuracy using discrete nearest-code evaluation
            with torch.no_grad():
                logits = F.linear(pred_latents, codebook)  # (B, max_seq_len, num_codes)
                pred_idx = logits.argmax(dim=-1)
                correct = (pred_idx == batch_code_indices).float() * mask.float()
                acc = correct.sum().item() / mask.float().sum().item()
        else:
            # Compute masked loss using cross-entropy over discrete code indices
            logits_transpose = logits.transpose(1, 2)  # (B, num_codes, max_seq_len)
            loss_per_token = F.cross_entropy(logits_transpose, batch_code_indices, reduction='none')  # (B, max_seq_len)
            masked_loss = (loss_per_token * mask.float()).sum() / mask.float().sum()
            loss = masked_loss

            # Compute masked accuracy
            with torch.no_grad():
                pred_idx = logits.argmax(dim=-1)  # (B, max_seq_len)
                correct = (pred_idx == batch_code_indices).float() * mask.float()
                acc = correct.sum().item() / mask.float().sum().item()
    else:
        # No masking (all sequences same length)
        if use_embedding_loss:
            target_embs_flat = codebook[code_indices.long()].to(pred_latents.device)  # (B*max_seq_len, emb_dim)
            target_embs = target_embs_flat.view(B, max_seq_len, -1)
            loss = F.mse_loss(pred_latents, target_embs)
            with torch.no_grad():
                logits = F.linear(pred_latents, codebook)
                pred_idx = logits.argmax(dim=-1)
                acc = (pred_idx == batch_code_indices).float().mean().item()
        else:
            loss = ce_loss(logits.transpose(1, 2), batch_code_indices)
            with torch.no_grad():
                pred_idx = logits.argmax(dim=-1)
                acc = (pred_idx == batch_code_indices).float().mean().item()
    
    return loss, acc

def get_train_batch(batch_idx, train_profiles, train_action_indices, train_latents, device):
    """Get a training batch with padding for variable-length sequences."""
    batch_profiles = torch.stack([train_profiles[j] for j in batch_idx])
    batch_action_idx = torch.tensor([train_action_indices[j] for j in batch_idx], dtype=torch.long, device=device)
    
    # Get actual lengths before padding
    lengths = torch.tensor([train_latents[j].shape[1] for j in batch_idx], dtype=torch.long, device=device)
    max_len = lengths.max().item()
    
    # Pad latents to max length in batch
    padded_latents = []
    for j in batch_idx:
        lat = train_latents[j]  # (code_dim, seq_len)
        if lat.shape[1] < max_len:
            # Pad on the sequence dimension (dim=1)
            pad_len = max_len - lat.shape[1]
            padded = F.pad(lat, (0, pad_len), mode='constant', value=0)
        else:
            padded = lat
        padded_latents.append(padded)
    
    batch_latents = torch.stack(padded_latents).to(device)
    return batch_profiles, batch_action_idx, batch_latents, lengths

def evaluate_validation(val_profiles, val_action_indices, val_latents, batch_size, mapping_net, action_embedding, model, ce_loss, code_embed_dim, use_embedding_loss=False):
    """Evaluate using integer action indices and an nn.Embedding module with variable-length support.

    val_action_indices: list of int indices aligned with val_profiles/val_latents
    action_embedding: nn.Embedding instance on the correct device
    """
    if len(val_profiles) == 0:
        return None, None
    mapping_net.eval()
    with torch.no_grad():
        val_loss_sum = 0.0
        val_acc_sum = 0.0
        total_tokens = 0
        
        for i in range(0, len(val_profiles), batch_size):
            end_idx = min(i + batch_size, len(val_profiles))
            batch_profiles = torch.stack(val_profiles[i:end_idx])
            batch_idx = torch.tensor(val_action_indices[i:end_idx], dtype=torch.long, device=batch_profiles.device)
            batch_actions = action_embedding(batch_idx)
            
            # Pad validation latents
            lengths = torch.tensor([val_latents[j].shape[1] for j in range(i, end_idx)], 
                                   dtype=torch.long, device=batch_profiles.device)
            max_len = lengths.max().item()
            
            padded_latents = []
            for j in range(i, end_idx):
                lat = val_latents[j]
                if lat.shape[1] < max_len:
                    pad_len = max_len - lat.shape[1]
                    padded = F.pad(lat, (0, pad_len), mode='constant', value=0)
                else:
                    padded = lat
                padded_latents.append(padded)
            
            batch_latents = torch.stack(padded_latents).to(batch_profiles.device)
            
            loss, acc = compute_loss_and_acc(batch_profiles, batch_actions, batch_latents, 
                                            mapping_net, model, ce_loss, code_embed_dim, lengths=lengths, use_embedding_loss=use_embedding_loss)
            
            # Weight by number of actual tokens
            num_tokens = lengths.sum().item()
            val_loss_sum += loss.item() * num_tokens
            val_acc_sum += acc * num_tokens
            total_tokens += num_tokens
            
        val_loss = val_loss_sum / max(1, total_tokens)
        val_acc = val_acc_sum / max(1, total_tokens)
    # Additional per-action breakdown for diagnostics
    # Build mapping from action_idx -> list of sample indices
    action_to_indices = {}
    for i, aidx in enumerate(val_action_indices):
        action_to_indices.setdefault(aidx, []).append(i)

    per_action_stats = {}
    for aidx, idx_list in action_to_indices.items():
        # build a small batch for this action (may do multiple batches if large)
        # We'll compute average loss/acc across all tokens for this action
        action_loss_sum = 0.0
        action_acc_sum = 0.0
        action_tokens = 0
        for start in range(0, len(idx_list), batch_size):
            sub_idx = idx_list[start:start+batch_size]
            batch_profiles = torch.stack([val_profiles[j] for j in sub_idx])
            batch_idx = torch.tensor([val_action_indices[j] for j in sub_idx], dtype=torch.long, device=batch_profiles.device)
            batch_actions = action_embedding(batch_idx)

            lengths = torch.tensor([val_latents[j].shape[1] for j in sub_idx], dtype=torch.long, device=batch_profiles.device)
            max_len = lengths.max().item()
            padded_latents = []
            for j in sub_idx:
                lat = val_latents[j]
                if lat.shape[1] < max_len:
                    pad_len = max_len - lat.shape[1]
                    padded = F.pad(lat, (0, pad_len), mode='constant', value=0)
                else:
                    padded = lat
                padded_latents.append(padded)
            batch_latents = torch.stack(padded_latents).to(batch_profiles.device)

            loss, acc = compute_loss_and_acc(batch_profiles, batch_actions, batch_latents,
                                            mapping_net, model, ce_loss, code_embed_dim, lengths=lengths)
            num_tokens = lengths.sum().item()
            action_loss_sum += loss.item() * num_tokens
            action_acc_sum += acc * num_tokens
            action_tokens += num_tokens

        if action_tokens > 0:
            per_action_stats[aidx] = {
                'loss': action_loss_sum / action_tokens,
                'acc': action_acc_sum / action_tokens,
                'count': len(idx_list)
            }

    # Print top 10 worst actions by loss for quick diagnostics
    if len(per_action_stats) > 0:
        sorted_actions = sorted(per_action_stats.items(), key=lambda x: x[1]['loss'], reverse=True)
        print("\nTop validation actions by loss (action_idx -> loss, acc, count):")
        for aidx, stats in sorted_actions[:10]:
            act_name = aidx if isinstance(aidx, int) else str(aidx)
            print(f"  {act_name} -> loss={stats['loss']:.4f}, acc={stats['acc']:.4f}, count={stats['count']}")

    mapping_net.train()
    return val_loss, val_acc

def main():
    args = option_subject_prior.get_args_parser().parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VQ-VAE model using parameters from args
    # Try to use weights_only=True when supported to reduce pickle surface; fall back if not available.
    try:
        vqvae_ckpt = torch.load(args.vqvae_checkpoint, map_location=device, weights_only=True)
    except TypeError:
        vqvae_ckpt = torch.load(args.vqvae_checkpoint, map_location=device)

    # If the VQ-VAE checkpoint contains saved args, use any missing values to
    # populate the current args. We don't overwrite CLI-provided values; we only
    # set attributes that are not present on args.
    if isinstance(vqvae_ckpt, dict) and 'args' in vqvae_ckpt:
        ckpt_args = vqvae_ckpt['args']
        for k, v in ckpt_args.items():
            if not hasattr(args, k) or getattr(args, k) is None:
                setattr(args, k, v)
        print(f"Merged {len(ckpt_args)} args from VQ-VAE checkpoint into current args (missing fields only).")

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
    # Helper to extract a state_dict from common checkpoint wrappers and strip DataParallel 'module.' prefixes
    def _extract_state_dict(ckpt):
        if isinstance(ckpt, dict):
            # common keys that wrap state dicts
            for k in ('net', 'state_dict', 'model'):
                if k in ckpt:
                    sd = ckpt[k]
                    break
            else:
                sd = ckpt
        else:
            sd = ckpt

        # If keys are prefixed with 'module.', strip that (checkpoint saved from DataParallel)
        if isinstance(sd, dict):
            keys = list(sd.keys())
            if any(k.startswith('module.') for k in keys):
                new_sd = {}
                for k, v in sd.items():
                    new_k = k[len('module.'):] if k.startswith('module.') else k
                    new_sd[new_k] = v
                sd = new_sd
        return sd

    sd = _extract_state_dict(vqvae_ckpt)
    # Load leniently and report missing/unexpected keys to help debug; strict loading may fail if architecture differs.
    load_res = model.load_state_dict(sd, strict=False)
    if hasattr(load_res, 'missing_keys') or hasattr(load_res, 'unexpected_keys'):
        missing = getattr(load_res, 'missing_keys', [])
        unexpected = getattr(load_res, 'unexpected_keys', [])
        if missing:
            print(f"Warning: missing keys when loading VQ-VAE checkpoint: {missing}")
        if unexpected:
            print(f"Warning: unexpected keys in VQ-VAE checkpoint: {unexpected}")
    model.to(device)
    model.eval()

    # Load pretrained profile encoder
    try:
        encoder_ckpt = torch.load(args.encoder_checkpoint, map_location=device, weights_only=True)
    except TypeError:
        encoder_ckpt = torch.load(args.encoder_checkpoint, map_location=device)

    # If the encoder checkpoint contains saved args, populate missing fields in args.
    if isinstance(encoder_ckpt, dict) and 'args' in encoder_ckpt:
        enc_ck_args = encoder_ckpt['args']
        for k, v in enc_ck_args.items():
            if not hasattr(args, k) or getattr(args, k) is None:
                setattr(args, k, v)
        print(f"Merged {len(enc_ck_args)} args from encoder checkpoint into current args (missing fields only).")

    encoder = profile_modules.ProfileEncoder(
        latent_dim=args.latent_dim,  # Use latent_dim (16) - the encoder was trained with this
        profile_dim=args.profile_dim,
        bio_out_dim=args.bio_out_dim,
        biomech_dim=args.biomech_dim
    ).to(device)
    # Encoder checkpoint may also be wrapped; extract and strip 'module.' if necessary
    enc_sd = _extract_state_dict(encoder_ckpt)
    enc_res = encoder.load_state_dict(enc_sd, strict=False)
    if hasattr(enc_res, 'missing_keys') or hasattr(enc_res, 'unexpected_keys'):
        missing = getattr(enc_res, 'missing_keys', [])
        unexpected = getattr(enc_res, 'unexpected_keys', [])
        if missing:
            print(f"Warning: missing keys when loading encoder checkpoint: {missing}")
        if unexpected:
            print(f"Warning: unexpected keys in encoder checkpoint: {unexpected}")
    encoder.eval()

    # Prepare dataset loader
    loader = dataset_183_retarget.retargeted183_data_loader(
        data_dir=args.dataset_path,
        num_workers=4,
        pre_load=True
    )
    dataset = loader.dataset

    # Encode motions using pretrained encoder
    subject_samples = defaultdict(list)
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Encoding Motions")):
            try:
                motions, _, names, subject_names = batch
            except ValueError as e:
                print(f"Skipping batch due to error: {e}")
                continue
            except StopIteration:
                break
            motions = motions.to(device, dtype=torch.float32)
            latents = model.vqvae.encoder(model.vqvae.preprocess(motions))
            # Store as-is without transpose, matching the notebook's approach
            for latent, subject_name, action_name in zip(latents, subject_names, names):
                subject_samples[subject_name].append((latent.cpu(), action_name))
    
    # Print diagnostics about the stored latent shapes
    if len(subject_samples) > 0:
        sample_subject = next(iter(subject_samples.keys()))
        sample_latent = subject_samples[sample_subject][0][0]
        print(f"Stored latent shape per sample: {sample_latent.shape}")
        print(f"VQ-VAE codebook shape: {model.vqvae.quantizer.codebook.shape}")

    rng = np.random.RandomState(42)
    _all_subjects = sorted([s for s in subject_samples.keys() if len(subject_samples[s]) > 0])
    n_test = min(33, len(_all_subjects))
    n_train_target = min(150, len(_all_subjects) - n_test)
    if len(_all_subjects) < 1:
        raise RuntimeError("No subjects with motions found in subject_samples.")
    if len(_all_subjects) <= n_test:
        raise RuntimeError(f"Not enough subjects ({len(_all_subjects)}) to reserve {n_test} for validation.")

    test_subjects = list(rng.choice(_all_subjects, size=n_test, replace=False))
    _remaining = [s for s in _all_subjects if s not in test_subjects]
    train_candidates = [s for s in _remaining if len(subject_samples[s]) >= 2]

    if len(train_candidates) < n_train_target:
        warnings.warn(f"Only {len(train_candidates)} subjects have >=2 motions; will use all of them for training.")
        train_subjects = train_candidates
    else:
        rng.shuffle(train_candidates)
        train_subjects = train_candidates[:n_train_target]

    subjects = list(train_subjects)
    print(f"Using {len(subjects)} subjects for training, reserving {len(test_subjects)} subjects for validation.")

    biomech_dim = args.biomech_dim
    profile_dim = args.profile_dim
    latent_dim = args.latent_dim
    seq_len = args.seq_len
    batch_size = args.batch_size
    num_epochs_decoder = args.decoder_epochs

    # Compute subject profiles
    train_prof_dict = compute_subject_profiles(train_subjects, subject_samples, encoder, device, dataset, biomech_dim)
    test_prof_dict = compute_subject_profiles(test_subjects, subject_samples, encoder, device, dataset, biomech_dim)

    action_labels = set()
    train_profiles = []
    train_actions = []
    train_latents = []

    for subject in train_subjects:
        if subject not in train_prof_dict:
            continue
        samples = subject_samples[subject]
        prof_samples = train_prof_dict[subject]
        for (latent, action_name), (profile_vec, prof_action_name) in zip(samples, prof_samples):
            assert action_name == prof_action_name
            action_labels.add(action_name)
            train_profiles.append(profile_vec.to(device))
            train_actions.append(action_name)
            train_latents.append(latent)

    action_list = sorted(list(action_labels))
    action_to_idx = {a: i for i, a in enumerate(action_list)}
    num_actions = len(action_list)

    # Build integer action indices for each training sample (for nn.Embedding)
    train_action_indices = [get_action_index(a, action_to_idx) for a in train_actions]
    if any(idx is None for idx in train_action_indices):
        raise RuntimeError("Found training action not in action_to_idx mapping")
    # Keep indices as plain Python list; we'll convert to tensors per-batch

    val_profiles = []
    val_action_indices = []
    val_latents = []
    for subject in test_subjects:
        if subject not in test_prof_dict:
            continue
        samples = subject_samples[subject]
        prof_samples = test_prof_dict[subject]
        for (latent, action_name), (profile_vec, prof_action_name) in zip(samples, prof_samples):
            assert action_name == prof_action_name
            idx = get_action_index(action_name, action_to_idx)
            if idx is None:
                continue
            val_profiles.append(profile_vec.to(device))
            val_action_indices.append(idx)
            val_latents.append(latent)

    if len(val_profiles) == 0:
        print("Warning: No validation samples (all actions unseen). Validation will be skipped.")

    # Diagnostic: report action overlap between train and val
    val_action_set = set([action_list[idx] for idx in val_action_indices]) if len(val_action_indices) > 0 else set()
    train_action_set = set(action_list)
    overlap = train_action_set & val_action_set
    print(f"Train actions: {len(train_action_set)}, Val actions: {len(val_action_set)}, Overlap: {len(overlap)}")

    # Diagnostic: sequence length distributions
    import collections
    train_len_counts = collections.Counter([lat.shape[1] for lat in train_latents])
    val_len_counts = collections.Counter([lat.shape[1] for lat in val_latents]) if len(val_latents) > 0 else collections.Counter()
    print("Train sequence length distribution (len:count) sample: ")
    for l, c in list(train_len_counts.items())[:10]:
        print(f"  {l}: {c}")
    if len(val_len_counts) > 0:
        print("Val sequence length distribution (len:count) sample: ")
        for l, c in list(val_len_counts.items())[:10]:
            print(f"  {l}: {c}")

    # Determine actual dimensions from the stored latents
    sample_latent_shape = train_latents[0].shape
    print(f"Sample training latent shape: {sample_latent_shape}")
    
    # The latents are (code_dim, seq_len) from VQ-VAE encoder
    actual_code_dim = sample_latent_shape[0]
    
    # Find max and median sequence lengths for architecture design
    all_seq_lens = [lat.shape[1] for lat in train_latents]
    min_seq_len = min(all_seq_lens)
    max_seq_len = max(all_seq_lens)
    median_seq_len = int(np.median(all_seq_lens))
    unique_lens = set(all_seq_lens)
    
    num_codes = model.vqvae.quantizer.codebook.shape[0]
    code_embed_dim = model.vqvae.quantizer.codebook.shape[1]
    
    print(f"VQ-VAE codebook: {num_codes} codes, embedding dim: {code_embed_dim}")
    print(f"Latent dimensions: code_dim={actual_code_dim}")
    print(f"\nSequence length statistics across {len(train_latents)} training samples:")
    print(f"  Min length: {min_seq_len}")
    print(f"  Max length: {max_seq_len}")
    print(f"  Median length: {median_seq_len}")
    print(f"  Unique lengths: {len(unique_lens)}")
    
    if len(unique_lens) > 1:
        print(f"\n  Found {len(unique_lens)} different sequence lengths:")
        len_counts = {l: all_seq_lens.count(l) for l in sorted(unique_lens)[:10]}  # Show top 10
        for length, count in sorted(len_counts.items()):
            print(f"    Length {length}: {count} samples ({100*count/len(all_seq_lens):.1f}%)")
        if len(unique_lens) > 10:
            print(f"    ... and {len(unique_lens) - 10} more unique lengths")
        
        print(f"\n  Using VARIABLE-LENGTH architecture with max_seq_len={max_seq_len}")
        print(f"  All sequences will be padded to max length within each batch.")
        print(f"  Loss and accuracy will use masking to ignore padding tokens.\n")
    
    # Use max_seq_len for decoder architecture (will pad shorter sequences)
    actual_seq_len = max_seq_len

    # Choose action embedding dim (small, trainable vector per action)
    action_emb_dim = min(64, max(8, num_actions // 2))
    
    # Use actual_seq_len (max) from data (not args.seq_len)
    print(f"Initializing decoder with max_seq_len={actual_seq_len}, code_dim={actual_code_dim}")
    mapping_net = profile_modules.ProfileActionToMotionTransformer(
        profile_dim, actual_code_dim, actual_seq_len, action_emb_dim
    ).to(device)

    # Learned action embeddings
    action_embedding = nn.Embedding(num_actions, action_emb_dim).to(device)
    # include action_embedding parameters in optimizer so they're trained with weight decay
    dec_optimizer = torch.optim.AdamW(
        list(mapping_net.parameters()) + list(action_embedding.parameters()),
        lr=1e-3, weight_decay=1e-4
    )
    # Scheduler to reduce LR on plateau (validation loss)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(dec_optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    ce_loss = nn.CrossEntropyLoss()
    num_samples = len(train_profiles)
    indices = np.arange(num_samples)
    all_losses = []
    val_losses = []
    best_val = float('inf')

    for epoch in range(num_epochs_decoder):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        epoch_acc = 0.0
        total = 0

        for i in range(0, num_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_profiles, batch_action_idx, batch_latents, lengths = get_train_batch(
                batch_idx, train_profiles, train_action_indices, train_latents, device
            )

            # Convert integer indices to learned embeddings
            batch_action_emb = action_embedding(batch_action_idx)

            use_embedding_loss = getattr(args, 'use_embedding_loss', False)
            loss, acc = compute_loss_and_acc(
                batch_profiles, batch_action_emb, batch_latents,
                mapping_net, model, ce_loss, actual_code_dim, lengths=lengths, use_embedding_loss=use_embedding_loss
            )
            dec_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(mapping_net.parameters(), max_norm=5.0)
            dec_optimizer.step()

            # Weight epoch metrics by the number of actual tokens (not just samples)
            # compute_loss_and_acc returns a per-token average when lengths is provided,
            # so multiply by the number of tokens to accumulate correctly (matching validation).
            num_tokens = lengths.sum().item()
            epoch_loss += loss.item() * num_tokens
            epoch_acc += acc * num_tokens
            total += num_tokens

        epoch_loss /= max(1, total)
        epoch_acc /= max(1, total)
        all_losses.append(epoch_loss)
        # Run validation every epoch so scheduler and early stopping can act
        if len(val_profiles) > 0:
            use_embedding_loss = getattr(args, 'use_embedding_loss', False)
            val_loss, val_acc = evaluate_validation(
                val_profiles, val_action_indices, val_latents, batch_size,
                mapping_net, action_embedding, model, ce_loss, actual_code_dim, use_embedding_loss=use_embedding_loss
            )
        else:
            val_loss, val_acc = None, None

        # Scheduler steps using validation loss; save best checkpoint when improved.
        if val_loss is not None:
            scheduler.step(val_loss)
            val_losses.append(val_loss)
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                # save best checkpoint (weights + embedding + mapping)
                best_path = os.path.join(args.out_dir, 'profile_decoder_best.pth')
                torch.save({
                    'net': mapping_net.state_dict(),
                    'action_embedding': action_embedding.state_dict(),
                    'action_list': action_list,
                    'action_to_idx': action_to_idx,
                    'decoder_args': {
                        'profile_dim': profile_dim,
                        'code_dim': actual_code_dim,
                        'seq_len': actual_seq_len,
                        'action_emb_dim': action_emb_dim,
                        'num_actions': num_actions
                    }
                }, best_path)

        msg = f"Decoder Epoch {epoch+1}/{num_epochs_decoder} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | PPL: {np.exp(epoch_loss):.2f}"
        if val_loss is not None:
            msg += f" || Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val PPL: {np.exp(val_loss):.2f}"
        print(msg)

    if getattr(args, 'use_embedding_loss', False):
        print("Note: Decoder was trained using embedding-MSE loss (use_embedding_loss=True). Validation metrics still compute discrete accuracy via nearest-code.")

    print("Decoder training complete.")
    
    # Save the ACTUAL dimensions used during training (not args values)
    decoder_args = {
        'profile_dim': profile_dim,
        'code_dim': actual_code_dim,        # Save actual from data
        'seq_len': actual_seq_len,          # Save actual from data
        'action_emb_dim': action_emb_dim,
        'num_actions': num_actions,
        'biomech_dim': biomech_dim,
        'latent_dim': latent_dim,
        'downsample_factor': getattr(args, 'stride_t', 2) ** getattr(args, 'depth', 3),
        'window_size': getattr(args, 'window_size', None)
    }
    
    # Save into the output directory provided by the CLI
    decoder_save_path = os.path.join(args.out_dir, "profile_decoder.pth")
    # Save a single checkpoint that contains the decoder weights, action embeddings, and architecture
    checkpoint = {
        'net': mapping_net.state_dict(),
        'action_embedding': action_embedding.state_dict(),
        'action_list': action_list,
        'action_to_idx': action_to_idx,
        'decoder_args': decoder_args,       # Add explicit decoder architecture params
        'use_embedding_loss': getattr(args, 'use_embedding_loss', False),
        'args': vars(args)
    }
    torch.save(checkpoint, decoder_save_path)
    print(f"Saved decoder to {decoder_save_path}")
    print(f"Decoder architecture: code_dim={actual_code_dim}, seq_len={actual_seq_len}, action_emb_dim={action_emb_dim}")
    print(f"  Motion frame length: ~{actual_seq_len * decoder_args['downsample_factor']} frames")

    # Keep a human-readable mapping as well
    mapping_save = os.path.join(args.out_dir, "action_mapping.json")
    with open(mapping_save, "w") as f:
        json.dump({
            'action_list': action_list, 
            'action_to_idx': action_to_idx,
            'decoder_args': decoder_args
        }, f, indent=2)
    mapping_net.eval()

if __name__ == "__main__":
    main()