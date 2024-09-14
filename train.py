import os
import datetime
import tensorflow as tf

from src.args import parse_args
from src.dataloader import load_data
from src.model import RPD_Model_1, RPD_Model_2


def create_optimizer(name, learning_rate):
    """
    Create an optimizer based on the provided name and learning rate.

    Args:
        name (str): Name of the optimizer, e.g., 'Adam', 'SGD', 'RMSprop', 'Nadam', 'Adagrad'.
        learning_rate (float): The learning rate to use with the optimizer.

    Returns:
        A TensorFlow optimizer instance.
    """
    optimizers = {
        'Adam': tf.keras.optimizers.Adam(learning_rate=learning_rate),
        'SGD': tf.keras.optimizers.SGD(learning_rate=learning_rate),
        'RMSprop': tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
        'Nadam': tf.keras.optimizers.Nadam(learning_rate=learning_rate),
        'Adagrad': tf.keras.optimizers.Adagrad(learning_rate=learning_rate),
    }
    if name not in optimizers:
        raise ValueError(f"Unsupported optimizer: {name}. Supported optimizers are: {list(optimizers.keys())}")
    return optimizers[name]


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    args.config_name = os.path.basename(args.config.replace('.json', ''))

    # Setup TensorFlow to dynamically allocate GPU memory and specify GPU order
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # Ensures consistent GPU indexing in case of multiple GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # Sets which GPU to use based on the command-line argument
    os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/CUDA/11.3.0/'

    # Determine input shape and weights based on command-line arguments
    input_shapes = (args.img_x, args.img_y, args.img_ch)  # Defines the input shape for the model
    weights = 'imagenet' if args.imagenet else None       # Determines whether to use ImageNet pre-trained weights

    # Initialize the appropriate model based on the task
    if args.task == 'combined':
        # Use a more complex model suitable for combined inputs
        model = RPD_Model_2(n_classes=1, input_shape=input_shapes, weights=weights,
                            num_heads=12, key_dim=args.key_dim, dropout=args.dropout_rate,
                            name='RPD_Model_combined')
    else:
        # Use a simpler model for single input tasks
        model = RPD_Model_1(n_classes=1, input_shape=input_shapes, weights=weights,
                            num_heads=args.num_heads, key_dim=args.key_dim, dropout=args.dropout_rate,
                            name=f'RPD_Model_{args.task}')

    # Load weights from a pre-trained model if specified
    if args.load_model:
        model.load_weights(args.load_model)

    # Compile the model with the specified optimizer, loss function, and metrics
    model.compile(
        optimizer=create_optimizer(args.optimizer, args.init_lr),  # Creates an optimizer with the specified learning rate
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), # Specifies the loss type for binary classification
        metrics=['accuracy']                                       # The metric to be evaluated by the model during training and testing
    )

    # Set up the model checkpoint callback to save the best model during training
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'{args.checkpoint_dir}/{args.config_name}.tf',
        monitor='val_accuracy', verbose=1, save_best_only=True
    )

    # Initialize the TensorBoard callback
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=f'{args.log_dir}/{args.config_name}/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}',
        histogram_freq=1)

    # List of callbacks for the training, starting with model checkpointing
    callbacks = [checkpoint, tensorboard]

    # Optional: Set up Weights & Biases logging if enabled
    if args.use_wandb:
        import wandb
        from wandb.integration.keras import WandbMetricsLogger

        # Ensure the W&B API key is set
        if not os.getenv('WANDB_API_KEY'):
            raise ValueError("WANDB_API_KEY environment variable not set.")

        wandb.login()
        # Initialize a W&B run with the specified project, entity, group, and config
        wandb.init(project=args.project_name, entity=args.entity, group=args.group,
                   config=vars(args), name=args.config_name)
        # Add W&B logging to the list of callbacks
        callbacks.append(WandbMetricsLogger())

    # Load the training and validation datasets
    train_dataset = load_data(args.task, 'train', args=args)
    valid_dataset = load_data(args.task, 'valid', args=args)

    # Start training the model
    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        steps_per_epoch=len(train_dataset),       # Ensures that one epoch uses all training data
        validation_steps=len(valid_dataset),      # Ensures that one epoch uses all validation data
        epochs=args.epochs,                       # Total number of epochs to train
        callbacks=callbacks                       # List of callbacks to be used during training
    )
