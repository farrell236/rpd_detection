import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'   # see issue #152
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/CUDA/11.3.0/'
os.environ['WANDB_API_KEY'] = 'REDACTED'

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

from rpd_model import RPD_Model_2

import wandb
from wandb.integration.keras import WandbMetricsLogger

wandb.login()
wandb.init(project='DeepSeeNet', entity="farrell236", group="RPD", name=f'faf_cfp_model')


faf_root = '/data/houbb/data/AREDS/AREDS2/jpeg'
cfp_root = '/data/houbb/data/AREDS/AREDS2/rpd_pp_2048'

train_df = pd.read_csv('data/faf_cfp_train.csv')
train_df['filepath_faf'] = train_df['filepath_faf'].apply(lambda x: os.path.join(faf_root, x.replace('dcm', 'jpeg')))
train_df['filepath_cfp'] = train_df['filepath_cfp'].apply(lambda x: os.path.join(cfp_root, x.replace('dcm', 'jpeg')))

valid_df = pd.read_csv('data/faf_cfp_valid.csv')
valid_df['filepath_faf'] = valid_df['filepath_faf'].apply(lambda x: os.path.join(faf_root, x.replace('dcm', 'jpeg')))
valid_df['filepath_cfp'] = valid_df['filepath_cfp'].apply(lambda x: os.path.join(cfp_root, x.replace('dcm', 'jpeg')))


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


def augmentation_fn(image):
    # Random left-right flip the image
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # Random rotation
    degree = tf.random.normal([]) * 360
    image = tfa.image.rotate(image, degree * np.pi / 180., interpolation='nearest')

    # Random brightness, saturation and contrast shifting
    # image = tf.image.random_brightness(image, 0.2)
    # image = tf.image.random_hue(image, 0.08)
    # image = tf.image.random_saturation(image, 0.6, 1.6)
    # image = tf.image.random_contrast(image, 0.7, 1.3)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image


def load_image_train(filepath_faf, filepath_cfp, rpd_labels):
    images, label = parse_function(filepath_faf, filepath_cfp, rpd_labels)
    images['cfp_input'] = augmentation_fn(images['cfp_input'])
    return images, label


train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_df['filepath_faf'], train_df['filepath_cfp'], train_df['rpd_labels']))
train_dataset = train_dataset.shuffle(len(train_dataset))
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=8)
train_dataset = train_dataset.batch(16)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

valid_dataset = tf.data.Dataset.from_tensor_slices(
    (valid_df['filepath_faf'], valid_df['filepath_cfp'], valid_df['rpd_labels']))
valid_dataset = valid_dataset.map(parse_function, num_parallel_calls=8)
valid_dataset = valid_dataset.batch(4)


model = RPD_Model_2(input_shape=(512, 512, 3), weights=None)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=f'checkpoints/faf_cfp_model.tf',
    monitor='val_accuracy', verbose=1, save_best_only=True
)

model.fit(
    train_dataset,
    validation_data=valid_dataset,
    # class_weight=class_weights,
    steps_per_epoch=len(train_dataset),
    validation_steps=len(valid_dataset),
    epochs=200,
    callbacks=[checkpoint, WandbMetricsLogger()],
)
