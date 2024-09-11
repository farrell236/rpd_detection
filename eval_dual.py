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


def parse_function(filepath_faf, filepath_cfp, rpd_labels):
    # Read FAF Image
    faf_image = tf.io.read_file(filepath_faf)
    faf_image = tf.io.decode_jpeg(faf_image, channels=3)
    faf_image = faf_image[:768, :768]  # delete bottom border
    faf_image = tf.image.resize(faf_image, [512, 512], method='nearest')
    faf_image = tf.image.convert_image_dtype(faf_image, tf.float32)

    # Read CFP Image
    cfp_image = tf.io.read_file(filepath_cfp)
    cfp_image = tf.io.decode_jpeg(cfp_image, channels=3)
    cfp_image = tf.image.resize(cfp_image, [512, 512], method='nearest')
    cfp_image = tf.image.convert_image_dtype(cfp_image, tf.float32)

    return {'faf_input': faf_image, 'cfp_input': cfp_image}, rpd_labels


test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_df['filepath_faf'], test_df['filepath_cfp'], test_df['rpd_labels']))
test_dataset = test_dataset.map(parse_function, num_parallel_calls=8)
test_dataset = test_dataset.batch(4)


model = tf.keras.models.load_model('checkpoints/faf_cfp_model.tf')

y_pred = []
y_true = []
for idx, (image, label) in tqdm(enumerate(test_dataset), total=len(test_dataset)):
    y_true.append(label)
    y_pred.append(tf.nn.sigmoid(model.predict(image, verbose=False)).numpy())
y_pred = np.concatenate(y_pred)
y_true = np.concatenate(y_true)

plot_roc_curves(y_true, y_pred)
print(compute_classification_metrics(y_true, np.squeeze(y_pred)))
