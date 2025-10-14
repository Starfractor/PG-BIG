import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(
        description='Subject Prior Training Options',
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data and experiment
    parser.add_argument('--dataset-path', type=str, default='./dataset/183_retargeted', help='Path to the dataset')
    parser.add_argument('--vqvae-checkpoint', type=str, required=True, help='Path to VQ-VAE checkpoint')
    parser.add_argument('--encoder-checkpoint', type=str, required=True, help='Path to encoder checkpoint')
    parser.add_argument('--exp-name', type=str, default='subject_prior', help='Experiment name for saving results')
    parser.add_argument('--out-dir', type=str, default='./outputs/subject_prior', help='Directory to save experiment outputs')

    # Training
    parser.add_argument('--decoder-epochs', type=int, default=5000, help='Number of epochs for decoder training')
    parser.add_argument('--batch-size', type=int, default=124, help='Batch size for training')

    # VQ-VAE architecture
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

    # Decoder architecture
    parser.add_argument('--seq-len', type=int, default=512, help='Sequence length')
    parser.add_argument('--num-actions', type=int, default=20, help='Number of actions')

    # Misc
    parser.add_argument('--seed', type=int, default=123, help='Random seed')

    return parser