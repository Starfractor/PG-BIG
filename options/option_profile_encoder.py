import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(
        description='Profile Encoder Training Options',
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data and experiment
    parser.add_argument('--dataset-path', type=str, default='/home/mnt/datasets/183_retargeted', help='Path to the dataset')
    parser.add_argument('--checkpoint-path', type=str, default='./outputs/VQVAE/300000.pth', help='Path to VQ-VAE checkpoint')
    parser.add_argument('--exp-name', type=str, default='profile_encoder_meta', help='Experiment name for saving results')
    parser.add_argument('--out-dir', type=str, default='outputs/', help='Output directory for experiment results')

    # Training
    parser.add_argument('--total-epochs', type=int, default=2500, help='Number of epochs for encoder training')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for encoder training')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for encoder')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--warmup-epochs', type=int, default=100, help='Number of warmup epochs for LR scheduler')

    # Model architecture (VQ-VAE)
    parser.add_argument('--nb-code', type=int, default=512, help='Number of codes in VQ-VAE codebook')
    parser.add_argument('--code-dim', type=int, default=512, help='Code dimension')
    parser.add_argument('--output-emb-width', type=int, default=512, help='Output embedding width')
    parser.add_argument('--down-t', type=int, default=2, help='Downsampling factor in time')
    parser.add_argument('--stride-t', type=int, default=2, help='Stride in time')
    parser.add_argument('--width', type=int, default=512, help='Model width')
    parser.add_argument('--depth', type=int, default=3, help='Model depth')
    parser.add_argument('--dilation-growth-rate', type=int, default=3, help='Dilation growth rate')
    parser.add_argument('--vq-act', type=str, default='relu', help='Activation function for VQ-VAE')
    parser.add_argument('--vq-norm', type=str, default=None, help='Normalization for VQ-VAE')
    parser.add_argument('--nb-joints', type=int, default=37, help='Number of joints')
    parser.add_argument('--mu', type=float, default=0.99, help='Moving average decay for VQ-VAE')
    parser.add_argument('--quantizer', type=str, default='ema_reset', help='Quantizer type')
    parser.add_argument('--dataname', type=str, default='183_athletes', help='Dataset name')

    # Profile encoder architecture
    parser.add_argument('--latent-dim', type=int, default=16, help='Latent dimension size')
    parser.add_argument('--profile-dim', type=int, default=512, help='Profile embedding dimension')
    parser.add_argument('--bio-out-dim', type=int, default=64, help='Biomechanics output dimension')
    parser.add_argument('--biomech-dim', type=int, default=2048, help='Biomechanics vector dimension')

    # Misc
    parser.add_argument('--seed', type=int, default=123, help='Random seed')

    return parser.parse_args()