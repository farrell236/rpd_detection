import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'   # see issue #152
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/CUDA/11.3.0/'

import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm

from utils import plot_roc_curves, compute_classification_metrics


faf_root = '/data/houbb/data/AREDS/AREDS2/jpeg'
cfp_root = '/data/houbb/data/AREDS/AREDS2/rpd_pp_2048'

test_df = pd.read_csv('data/faf_cfp_valid.csv')
test_df['filepath_faf'] = test_df['filepath_faf'].apply(lambda x: os.path.join(faf_root, x.replace('dcm', 'jpeg')))
test_df['filepath_cfp'] = test_df['filepath_cfp'].apply(lambda x: os.path.join(cfp_root, x.replace('dcm', 'jpeg')))


def parse_function(filename, label):
    # Read entire contents of image
    image_string = tf.io.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.io.decode_jpeg(image_string, channels=3)

    # Resize image with padding to 1024x1024
    image = tf.image.resize(image, [512, 512], method='nearest')

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image, label


test_dataset = tf.data.Dataset.from_tensor_slices((test_df['filepath_faf'], test_df['rpd_labels']))
test_dataset = test_dataset.map(parse_function, num_parallel_calls=8)
test_dataset = test_dataset.batch(4)


model = tf.keras.models.load_model('checkpoints/faf_only_model.tf')

y_pred = []
y_true = []
for idx, (image, label) in tqdm(enumerate(test_dataset), total=len(test_dataset)):
    y_true.append(label)
    y_pred.append(tf.nn.sigmoid(model.predict(image, verbose=False)).numpy())
y_pred = np.concatenate(y_pred)
y_true = np.concatenate(y_true)

plot_roc_curves(y_true, y_pred)
print(compute_classification_metrics(y_true, np.squeeze(y_pred)))
