import os
import json

import numpy as np
import tensorflow as tf

from tqdm import tqdm

from src.args import parse_args
from src.dataloader import load_data
from src.utils import (plot_roc_curves,
                       compute_classification_metrics,
                       bootstrap_confidence_interval,
                       combine_metrics_and_intervals)


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    args.config_name = os.path.basename(args.config.replace('.json', ''))

    # Setup TensorFlow for GPU usage
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Consistent GPU indexing
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # Select GPU based on argument
    os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/CUDA/11.3.0/'

    # Load the pre-trained model
    model_path = f'{args.checkpoint_dir}/{args.config_name}.tf'
    model = tf.keras.models.load_model(model_path)

    # Load test dataset
    test_dataset = load_data(args.task, 'test', args=args)

    # Collect predictions and ground truths
    y_pred, y_true = [], []
    for idx, (image, label) in tqdm(enumerate(test_dataset), total=len(test_dataset)):
        y_pred.append(tf.nn.sigmoid(model.predict(image, verbose=False)).numpy())
        y_true.append(label)
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)

    # Plot and save ROC curves
    plot_roc_curves(y_true, y_pred, savefig=f'{args.results_dir}/{args.config_name}.pdf')

    # Calculate and print metrics and intervals
    results = combine_metrics_and_intervals(
        compute_classification_metrics(y_true, np.squeeze(y_pred)),
        bootstrap_confidence_interval(y_true, np.squeeze(y_pred))
    )
    print(results)

    # Save results to JSON
    results_path = f'{args.results_dir}/{args.config_name}.json'
    with open(results_path, 'w') as file:
        json.dump(results, file, indent=4)
