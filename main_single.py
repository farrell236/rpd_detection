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

from rpd_model import RPD_Model_1

import wandb
from wandb.integration.keras import WandbMetricsLogger

wandb.login()
# wandb.init(project='DeepSeeNet', entity="farrell236", group="RPD", name=f'faf_only_model')
wandb.init(project='DeepSeeNet', entity="farrell236", group="RPD", name=f'cfp_only_model')


# faf_root = '/data/houbb/data/AREDS/AREDS2/jpeg'
cfp_root = '/data/houbb/data/AREDS/AREDS2/rpd_pp_2048'

train_df = pd.read_csv('data/faf_cfp_train.csv')
# train_df['filepath_faf'] = train_df['filepath_faf'].apply(lambda x: os.path.join(faf_root, x.replace('dcm', 'jpeg')))
train_df['filepath_cfp'] = train_df['filepath_cfp'].apply(lambda x: os.path.join(cfp_root, x.replace('dcm', 'jpeg')))

valid_df = pd.read_csv('data/faf_cfp_valid.csv')
# valid_df['filepath_faf'] = valid_df['filepath_faf'].apply(lambda x: os.path.join(faf_root, x.replace('dcm', 'jpeg')))
valid_df['filepath_cfp'] = valid_df['filepath_cfp'].apply(lambda x: os.path.join(cfp_root, x.replace('dcm', 'jpeg')))


def parse_function(filename, label):
    # Read entire contents of image
    image_string = tf.io.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.io.decode_jpeg(image_string, channels=3)
    # image = image[:768, :768]  # delete bottom border (FAF only)

    # Resize image with padding to 1024x1024
    image = tf.image.resize(image, [512, 512], method='nearest')

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image, label


def augmentation_fn(image, label):
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

    return image, label


def load_image_train(image, label):
    image, label = parse_function(image, label)
    # image, label = augmentation_fn(image, label)  # (FAF only)
    return image, label


# train_dataset = tf.data.Dataset.from_tensor_slices((train_df['filepath_faf'], train_df['rpd_labels']))
train_dataset = tf.data.Dataset.from_tensor_slices((train_df['filepath_cfp'], train_df['rpd_labels']))
train_dataset = train_dataset.shuffle(len(train_dataset))
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=8)
train_dataset = train_dataset.batch(16)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

# valid_dataset = tf.data.Dataset.from_tensor_slices((valid_df['filepath_faf'], valid_df['rpd_labels']))
valid_dataset = tf.data.Dataset.from_tensor_slices((valid_df['filepath_cfp'], valid_df['rpd_labels']))
valid_dataset = valid_dataset.map(parse_function, num_parallel_calls=8)
valid_dataset = valid_dataset.batch(4)


model = RPD_Model_1(input_shape=(512, 512, 3), weights=None)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    # filepath=f'checkpoints/faf_only_model.tf',
    filepath=f'checkpoints/cfp_only_model.tf',
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
