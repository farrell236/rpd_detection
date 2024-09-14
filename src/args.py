import argparse
import json


def load_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def parse_args():
    parser = argparse.ArgumentParser(description="Train RPD Detection network with Keras.")

    # Configuration file parameter
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the configuration JSON file.')

    # Parse known arguments first to get the config file, if provided
    args, unknown = parser.parse_known_args()
    defaults = load_config(args.config) if args.config else {}

    # Model parameters
    parser.add_argument('--task', type=str, default=defaults.get('task'), choices=['faf', 'cfp', 'combined'],
                        help='Specifies the input type to use in the model.',
                        required='task' not in defaults)
    parser.add_argument('--imagenet', action='store_true', default=defaults.get('imagenet'),
                        help='Enable ImageNet weights and preprocessing for model initialization.')
    parser.add_argument('--load_model', type=str, default=defaults.get('load_model'),
                        help='Path to load a pre-trained model from.')

    # Data parameters
    parser.add_argument('--csv_root', type=str, default=defaults.get('csv_root'),
                        help='Root directory path where the CSV files containing dataset information are located.',
                        required='csv_root' not in defaults)
    parser.add_argument('--faf_root', type=str, default=defaults.get('faf_root'),
                        help='Directory path for storing or accessing FAF data.',
                        required='faf_root' not in defaults)
    parser.add_argument('--cfp_root', type=str, default=defaults.get('cfp_root'),
                        help='Directory path for storing or accessing CFP data.',
                        required='cfp_root' not in defaults)
    parser.add_argument('--img_x', type=int, default=defaults.get('img_x'),
                        help='Width of the input image',
                        required='img_x' not in defaults)
    parser.add_argument('--img_y', type=int, default=defaults.get('img_y'),
                        help='Height of the input image',
                        required='img_y' not in defaults)
    parser.add_argument('--img_ch', type=int, default=defaults.get('img_ch'),
                        help='Number of channels in the input image',
                        required='img_ch' not in defaults)
    parser.add_argument('--batch_size', type=int, default=defaults.get('batch_size'),
                        help='Number of samples per batch to load.',
                        required='batch_size' not in defaults)

    # Attention parameters
    parser.add_argument('--num_heads', type=int, default=defaults.get('num_heads'),
                        help='Number of attention heads',
                        required='num_heads' not in defaults)
    parser.add_argument('--key_dim', type=int, default=defaults.get('key_dim'),
                        help='Key dimension of transformer layers',
                        required='key_dim' not in defaults)
    parser.add_argument('--dropout_rate', type=float, default=defaults.get('dropout_rate'),
                        help='Dropout rate',
                        required='dropout_rate' not in defaults)

    # Training parameters
    parser.add_argument('--epochs', type=int, default=defaults.get('epochs'),
                        help='Total number of training epochs.',
                        required='epochs' not in defaults)
    parser.add_argument('--init_lr', type=float, default=defaults.get('init_lr'),
                        help='Initial learning rate for the optimizer.',
                        required='init_lr' not in defaults)
    parser.add_argument('--optimizer', type=str, default=defaults.get('optimizer'),
                        help='Type of optimizer to use (e.g., "adam", "sgd").',
                        required='optimizer' not in defaults)
    parser.add_argument('--seed', type=int, default=defaults.get('seed', 0),
                        help='Set a seed for reproducible randomness.')
    parser.add_argument('--gpu', type=str, default=defaults.get('gpu', ''),
                        help='GPU device ID to use for training, e.g., "0", "1".')

    # Logging and checkpoints
    parser.add_argument('--checkpoint_dir', type=str, default=defaults.get('checkpoint_dir'),
                        help='Directory where trained model checkpoints will be saved.',
                        required='checkpoint_dir' not in defaults)
    parser.add_argument('--log_dir', type=str, default=defaults.get('log_dir', 'logs'),
                        help='Directory where TensorBoard logs will be saved.')
    parser.add_argument('--results_dir', type=str, default=defaults.get('results_dir', 'results'),
                        help='Directory where TensorBoard logs will be saved.')

    # Weights & Biases Logging
    parser.add_argument('--use_wandb', action='store_true', default=defaults.get('use_wandb'),
                        help='Enable Weights & Biases logging.')
    parser.add_argument('--project', type=str, default=defaults.get('project'),
                        help='Weights & Biases project name.')
    parser.add_argument('--entity', type=str, default=defaults.get('entity'),
                        help='Weights & Biases user or team name.')
    parser.add_argument('--group', type=str, default=defaults.get('group'),
                        help='Specify a group name for this run.')

    return parser.parse_args()
