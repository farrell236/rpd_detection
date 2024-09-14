import os.path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa


def parse_image(file_path, args):
    """
    Reads and preprocesses an image based on the provided arguments and parameters.

    Args:
        file_path (str): The path to the image file.
        args (Namespace): The namespace containing runtime arguments.
        params (dict): A dictionary of parameters used for preprocessing the image.

    Returns:
        image (tf.Tensor): The preprocessed image tensor.
    """
    image = tf.io.read_file(file_path)
    image = tf.io.decode_jpeg(image, channels=args.img_ch)
    image = tf.image.resize(image, [args.img_x, args.img_y], method='nearest')
    if args.imagenet:
        image = tf.keras.applications.inception_v3.preprocess_input(tf.cast(image, tf.float32))
    else:
        image = tf.image.convert_image_dtype(image, tf.float32)
    return image


def augment_image(image):
    """
    Applies data augmentation to an image.

    Args:
        image (tf.Tensor): Image tensor to augment.

    Returns:
        image (tf.Tensor): Augmented image tensor.
    """
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


def load_data(task, mode, args):
    """
    Load and process data for training, validation, or testing, applying augmentation only to 'cfp' in the dual-input scenario.
    This function handles different tasks and modes specified by the arguments and uses additional parameters for data handling.

    Args:
        task (str): Specifies which input type ('faf', 'cfp', 'combined') is to be processed.
        mode (str): Mode of operation ('train', 'valid', 'test') determining how data is processed.
        args (Namespace): A namespace containing runtime arguments that may influence data loading and processing.
        params (dict): A dictionary of parameters providing detailed configuration for data processing.

    Returns:
        dataset (tf.data.Dataset): The processed dataset appropriate for the specified task and mode.
    """

    # Load the CSV file containing paths to images and labels
    df = pd.read_csv(os.path.join(args.csv_root, f'faf_cfp_{mode}.csv'))

    # Convert DICOM file paths to JPEG for both 'faf' and 'cfp' datasets
    df['filepath_faf'] = df['filepath_faf'].apply(lambda x: os.path.join(args.faf_root, x.replace('dcm', 'jpeg')))
    df['filepath_cfp'] = df['filepath_cfp'].apply(lambda x: os.path.join(args.cfp_root, x.replace('dcm', 'jpeg')))

    if task in ['faf', 'cfp']:
        # Create a dataset from the appropriate columns for single input types
        dataset = tf.data.Dataset.from_tensor_slices((df[f'filepath_{task}'], df['rpd_labels']))

        # Shuffle the dataset early in the pipeline if in training mode using a fixed seed for reproducibility
        if mode == 'train':
            dataset = dataset.shuffle(len(dataset), seed=args.seed)

        # Map the dataset to apply preprocessing and image parsing functions
        dataset = dataset.map(lambda x, y: (parse_image(x, args), y))

        # Apply augmentation specifically to 'cfp' dataset in training mode
        if task == 'cfp' and mode == 'train':
            dataset = dataset.map(lambda x, y: (augment_image(x), y))
    else:  # 'combined' handling for dual input types
        # Create datasets for both input types
        dataset_faf = tf.data.Dataset.from_tensor_slices((df['filepath_faf'], df['rpd_labels']))
        dataset_cfp = tf.data.Dataset.from_tensor_slices((df['filepath_cfp'], df['rpd_labels']))

        # Shuffle both datasets with the same seed to keep them synchronized
        if mode == 'train':
            dataset_faf = dataset_faf.shuffle(len(dataset_faf), seed=args.seed)
            dataset_cfp = dataset_cfp.shuffle(len(dataset_cfp), seed=args.seed)

        # Parse images and labels
        dataset_faf = dataset_faf.map(lambda x, y: (parse_image(x, args), y))
        dataset_cfp = dataset_cfp.map(lambda x, y: (parse_image(x, args), y))

        # Apply augmentation to 'cfp' images in training mode
        if mode == 'train':
            dataset_cfp = dataset_cfp.map(lambda x, y: (augment_image(x), y))

        # Zip the datasets to create pairs of 'faf' and 'cfp' inputs with the same labels
        dataset = tf.data.Dataset.zip((dataset_faf, dataset_cfp))
        dataset = dataset.map(lambda x, y: ({'faf_input': x[0], 'cfp_input': y[0]}, x[1]))

    # Batch the dataset and prefetch to improve performance
    dataset = dataset.batch(args.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
