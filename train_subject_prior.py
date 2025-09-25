from collections import defaultdict
from dataset import dataset_183_retarget
from dataset import dataset_addbiomechanics
from models import vqvae
from models import profile_encdec
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./dataset/183_retargeted', help='Path to the dataset')
    parser.add_argument('--checkpoint_path', type=str, default='./output/VQVAE/300000.pth', help='Path to VQ-VAE checkpoint')
    args = parser.parse_args()

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    model_args = ckpt.get('args', None)

    if model_args is None:
        # Provide default values if args are missing from checkpoint
        class Args:
            nb_code = 512
            code_dim = 512
            output_emb_width = 512
            down_t = 2
            stride_t = 2
            width = 512
            depth = 3
            dilation_growth_rate = 3
            vq_act = "relu"
            dataname = "183_athletes"
            quantizer = "ema_reset"
            vq_norm = None
            nb_joints=37
            mu = 0.99
        model_args = Args()
    

    # Load data and model
    model = vqvae.HumanVQVAE(
        model_args,
        nb_code=model_args.nb_code,
        code_dim=model_args.code_dim,
        output_emb_width=model_args.output_emb_width,
        down_t=model_args.down_t,
        stride_t=model_args.stride_t,
        width=model_args.width,
        depth=model_args.depth,
        dilation_growth_rate=model_args.dilation_growth_rate,
        activation=model_args.vq_act,
        norm=model_args.vq_norm
    )

    model.to(device)
    # Load dataset
    loader = dataset_183_retarget.retargeted183_data_loader(data_dir=args.dataset_path, num_workers=4, pre_load=False)

    # Remove 'module.' prefix if present
    state_dict = ckpt['net']
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    encoder = model.vqvae.encoder

    # Use VQ-VAE encoder to encode motions
    # Store per-subject list of (latent, action_name) to keep actions aligned
    subject_samples = defaultdict(list)
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Encoding Motions")):
            try:
                motions, _, names, subject_names = batch  # names: action names
            except ValueError as e:
                print(f"Skipping batch due to error: {e}")
                continue
            except StopIteration:
                break
            motions = motions.to(device, dtype=torch.float32)
            latents = model.vqvae.encoder(model.vqvae.preprocess(motions))
            for latent, subject_name, action_name in zip(latents, subject_names, names):
                subject_samples[subject_name].append((latent.cpu(), action_name))

    # --- Perform Training Loop for Profile Encoder (triplet on latents) ---

    rng = np.random.RandomState(42)

    # Subject selection
    _all_subjects = sorted([s for s in subject_samples.keys() if len(subject_samples[s]) > 0])
    n_test = 33
    n_train_target = 150
    if len(_all_subjects) < 1:
        raise RuntimeError("No subjects with motions found in subject_samples.")

    if len(_all_subjects) <= n_test:
        raise RuntimeError(f"Not enough subjects ({len(_all_subjects)}) to reserve {n_test} for validation.")

    test_subjects = list(rng.choice(_all_subjects, size=min(n_test, len(_all_subjects)), replace=False))
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

    # Device and model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 16
    profile_dim = 512
    bio_out_dim = 64
    dataset = loader.dataset
    biomech_dim = getattr(dataset, "default_biomech_dim", 2048)

    num_epochs = 2500
    batch_size = 512

    encoder = profile_encdec.ProfileEncoder(
        latent_dim=latent_dim,
        profile_dim=profile_dim,
        bio_out_dim=bio_out_dim,
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

    def sample_triplet(subjects, subject_samples, biomech_dim, device):
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
        a_bio = get_biomech_vector(subject_pos)
        p_bio = a_bio.copy()
        n_bio = get_biomech_vector(subject_neg)
        return anchor, positive, negative, a_bio, p_bio, n_bio

    def create_batch(batch_size, subjects, subject_samples, biomech_dim, device):
        anchors, positives, negatives = [], [], []
        anchor_bios, positive_bios, negative_bios = [], [], []
        for _ in range(batch_size):
            result = sample_triplet(subjects, subject_samples, biomech_dim, device)
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

    # Training loop (Profile Encoder)
    all_losses = []
    for epoch in range(num_epochs):
        if len(subjects) == 0:
            raise RuntimeError("No subjects with >=2 motions found.")
        np.random.shuffle(subjects)

        anchors, positives, negatives, anchor_bios, positive_bios, negative_bios = create_batch(
            batch_size, subjects, subject_samples, biomech_dim, device
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
    encoder_args = {
        'latent_dim': latent_dim,
        'profile_dim': profile_dim,
        'bio_out_dim': bio_out_dim,
        'biomech_dim': biomech_dim
    }
    torch.save({'encoder_state_dict': encoder.state_dict(),'args': encoder_args}, "profile_encoder.pth")
    encoder.eval()

    # --- Create subject tokens/profiles for train and test subjects ---
    def compute_subject_profiles(subject_list):
        prof_dict = {}
        rep_profiles = {}
        for subject in subject_list:
            samples = subject_samples[subject]
            if len(samples) == 0:
                continue
            latents_only = [l for l, _ in samples]
            latents_tensor = torch.stack(latents_only).to(device)
            bio_vec = get_biomech_vector(subject)
            biomech_tensor = torch.tensor(bio_vec, device=device, dtype=torch.float32).unsqueeze(0).repeat(latents_tensor.shape[0], 1)
            with torch.no_grad():
                try:
                    profile_t = encoder(latents_tensor, biomech_tensor)
                except TypeError:
                    profile_t = encoder(latents_tensor)
                profile_vectors = profile_t.detach().cpu().numpy()
            prof_dict[subject] = profile_vectors
            rep_profiles[subject] = profile_vectors.mean(axis=0)
        return prof_dict, rep_profiles

    train_prof_dict, train_rep_profiles = compute_subject_profiles(train_subjects)
    test_prof_dict, test_rep_profiles = compute_subject_profiles(test_subjects)

    # Prepare training and validation data for the decoder
    # action vocab from train only
    action_labels = set()
    train_profiles = []
    train_actions = []
    train_latents = []

    for subject in train_subjects:
        rep_profile = train_rep_profiles.get(subject, None)
        if rep_profile is None:
            continue
        samples = subject_samples[subject]
        for real_latent, name in samples:
            action_labels.add(name)
            train_profiles.append(torch.tensor(rep_profile, dtype=torch.float32, device=device))
            train_actions.append(name)
            train_latents.append(real_latent)

    action_list = sorted(list(action_labels))
    if "drop_jump" in action_list:
        action_list.remove("drop_jump")
        action_list.append("drop_jump")
    action_to_idx = {a: i for i, a in enumerate(action_list)}
    num_actions = len(action_list)

    def get_action_onehot(action):
        if action not in action_to_idx:
            return None
        idx = action_to_idx[action]
        onehot = torch.zeros(num_actions, device=device)
        onehot[idx] = 1.0
        return onehot

    train_actions_onehot = torch.stack([get_action_onehot(a) for a in train_actions])

    # Validation data (skip unseen actions)
    val_profiles = []
    val_actions_onehot = []
    val_latents = []
    for subject in test_subjects:
        rep_profile = test_rep_profiles.get(subject, None)
        if rep_profile is None:
            continue
        for real_latent, name in subject_samples[subject]:
            onehot = get_action_onehot(name)
            if onehot is None:
                continue
            val_profiles.append(torch.tensor(rep_profile, dtype=torch.float32, device=device))
            val_actions_onehot.append(onehot)
            val_latents.append(real_latent)

    if len(val_profiles) == 0:
        print("Warning: No validation samples (all actions unseen). Validation will be skipped.")

    # Decoder training: use logits over codes (K), not embedding dim
    seq_len = 512
    num_actions = len(action_list)
    num_codes = model.vqvae.quantizer.codebook.shape[0]   # K
    code_embed_dim = model.vqvae.quantizer.codebook.shape[1]  # De

    mapping_net = profile_encdec.ProfileDecoder(profile_dim, latent_dim, seq_len, num_actions).to(device)
    dec_optimizer = torch.optim.AdamW(mapping_net.parameters(), lr=1e-3, weight_decay=1e-4)
    projection = nn.Linear(mapping_net.codebook_dim, code_embed_dim).to(device)
    ce_loss = nn.CrossEntropyLoss()
    batch_size = 124
    num_epochs = 50000
    num_samples = len(train_profiles)
    indices = np.arange(num_samples)
    all_losses = []

    # Utility: forward pass and CE/accuracy over B,T tokens
    def compute_loss_and_acc(batch_profiles, batch_actions_onehot, batch_latents):
        pred_latents = mapping_net(batch_profiles, batch_actions_onehot)   # (B,T,Dp)
        pred_latents_proj = projection(pred_latents)                      # (B,T,De)

        # Targets: project and quantize to code indices
        target_latents_proj = projection(batch_latents)                   # (B,T,De)
        flat_targets = target_latents_proj.reshape(-1, code_embed_dim)
        with torch.no_grad():
            code_indices = model.vqvae.quantizer.quantize(flat_targets)   # (B*T,)
        batch_code_indices = code_indices.view(batch_latents.shape[0], batch_latents.shape[1]).to(device)  # (B,T)

        # Logits over codes via similarity to codebook embeddings
        codebook = model.vqvae.quantizer.codebook.to(device)              # (K, De)
        logits = F.linear(pred_latents_proj, codebook)                    # (B,T,K)

        loss = ce_loss(logits.transpose(1, 2), batch_code_indices)        # CE over K
        with torch.no_grad():
            pred_idx = logits.argmax(dim=-1)                              # (B,T)
            acc = (pred_idx == batch_code_indices).float().mean().item()
        return loss, acc

    # Build tensors for training arrays on demand inside loop to save mem
    def get_train_batch(batch_idx):
        batch_profiles = torch.stack([train_profiles[j] for j in batch_idx])
        batch_actions = torch.stack([train_actions_onehot[j] for j in batch_idx])
        batch_latents = torch.stack([train_latents[j] for j in batch_idx]).to(device)
        return batch_profiles, batch_actions, batch_latents

    def evaluate_validation():
        if len(val_profiles) == 0:
            return None, None
        mapping_net.eval()
        with torch.no_grad():
            val_loss_sum = 0.0
            val_acc_sum = 0.0
            total = 0
            for i in range(0, len(val_profiles), batch_size):
                batch_profiles = torch.stack(val_profiles[i:i+batch_size])
                batch_actions = torch.stack(val_actions_onehot[i:i+batch_size])
                batch_latents = torch.stack(val_latents[i:i+batch_size]).to(device)
                loss, acc = compute_loss_and_acc(batch_profiles, batch_actions, batch_latents)
                bs = batch_profiles.size(0)
                val_loss_sum += loss.item() * bs
                val_acc_sum += acc * bs
                total += bs
            val_loss = val_loss_sum / max(1, total)
            val_acc = val_acc_sum / max(1, total)
        mapping_net.train()
        return val_loss, val_acc

    for epoch in range(num_epochs):
        np.random.shuffle(indices)
        epoch_loss = 0.0
        epoch_acc = 0.0
        total = 0

        for i in range(0, num_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_profiles, batch_actions, batch_latents = get_train_batch(batch_idx)

            loss, acc = compute_loss_and_acc(batch_profiles, batch_actions, batch_latents)
            dec_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(mapping_net.parameters(), max_norm=5.0)
            dec_optimizer.step()

            bs = batch_profiles.size(0)
            epoch_loss += loss.item() * bs
            epoch_acc += acc * bs
            total += bs

        epoch_loss /= max(1, total)
        epoch_acc /= max(1, total)
        all_losses.append(epoch_loss)

        if (epoch + 1) % 100 == 0 or epoch == 0:
            val_loss, val_acc = evaluate_validation()
            msg = f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | PPL: {np.exp(epoch_loss):.2f}"
            if val_loss is not None:
                msg += f" || Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val PPL: {np.exp(val_loss):.2f}"
            print(msg)

    print("Training complete.")
    decoder_args = {
        'profile_dim': profile_dim,
        'latent_dim': latent_dim,
        'seq_len': seq_len,
        'num_actions': num_actions
    }
    torch.save({'mapping_net_state_dict': mapping_net.state_dict(), 'args': decoder_args}, "profile_decoder.pth")
    mapping_net.eval()

if __name__ == "__main__":
    main()