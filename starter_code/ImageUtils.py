import numpy as np

"""This script implements the functions for data augmentation
and preprocessing.
"""


def parse_record(record, training):
    """Parse a record to an image and perform data preprocessing.
    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [32, 32, 3].
    """
    ### YOUR CODE HERE
    if training:
        record = np.reshape(record, (32, 32, 3))
    ### END CODE HERE
    image = preprocess_image(record, training)  # If any.
    return image


def preprocess_image(image, training):
    """Preprocess a single image of shape [height, width, depth].
    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [32, 32, 3]. The processed image.
    """
    ### YOUR CODE HERE
    image = image.astype(np.float32) / 255.0
    ### END CODE HERE
    return image


# Other functions
### YOUR CODE HERE

### END CODE HERE
