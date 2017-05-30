import tensorflow as tf



def conv2d(data, weights, stride):
    """
    Returns a TensforFlow 2D convolutional layer.

    Args:
        data: The input tensor to the convolutional layer.
        weights: The convolutional weights for this layer.
        stride: The x and y stride for the convolution.

    Returns:
        The TensorFlow convolutional layer.
    """
    return tf.nn.conv2d(data, weights, strides=[1, stride, stride, 1], padding='SAME')

def pool(data, stride=2):
    """
    Returns a TensforFlow pooling layer.
    Args:
        data: The input tensor to the pooling layer.
        stride: The x and y stride for the convolution.

    Returns:
        The TensorFlow pooling layer.
    """
    return tf.nn.max_pool(data, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding='SAME')


def weight_variable(shape):
    """Returns a TensforFlow weight variable.

    Args:
        shape: The shape of the weight variable.

    Returns:
        A TensorFlow weight variable of the given size.
    """
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01))


def bias_variable(shape):
    """Returns a TensforFlow 2D bias variable.

    Args:
        shape: The shape of the bias variable.

    Returns:
        A TensorFlow bias variable of the specified shape.
    """
    return tf.Variable(tf.constant(0.01, shape=shape))
    