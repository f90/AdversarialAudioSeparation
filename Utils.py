import tensorflow as tf
import numpy as np

def getTrainableVariables(tag=""):
    return [v for v in tf.trainable_variables() if tag in v.name]

def getNumParams(tensors):
    return np.sum([np.prod(t.get_shape().as_list()) for t in tensors])

def crop(x1, x2):
    x1_shape = x1.get_shape().as_list()
    x2_shape = x2.get_shape().as_list()
    if x1_shape != x2_shape:
        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1 = tf.slice(x1, offsets, size)
    return x1

def crop_and_concat(x1,x2):
    '''
    Copy-and-crop operation for two feature maps of different size.
    Crops the first input x1 equally along its borders so that its shape is equal to 
    the shape of the second input x2, then concatenates them along the feature channel axis.
    :param x1: First input that is cropped and combined with the second input
    :param x2: Second input
    :return: Combined feature map
    '''
    x1 = crop(x1,x2)
    return tf.concat([x1, x2], axis=3)

def pad_freqs(tensor, target_shape):
    '''
    Pads the frequency axis of a 4D tensor of shape [batch_size, freqs, timeframes, channels] or 2D tensor [freqs, timeframes] with zeros
    so that it reaches the target shape. If the number of frequencies to pad is uneven, the rows are appended at the end. 
    :param tensor: Input tensor to pad with zeros along the frequency axis
    :param target_shape: Shape of tensor after zero-padding, list of length 2 or 4 (depending on input rank)
    :return: Input tensor padded with zeros along frequency axis so that its shape is target_shape
    '''
    target_freqs = (target_shape[1] if len(target_shape) == 4 else target_shape[0])
    if isinstance(tensor, tf.Tensor):
        input_shape = tensor.get_shape().as_list()
    else:
        input_shape = tensor.shape

    if len(input_shape) == 2:
        input_freqs = input_shape[0]
    else:
        input_freqs = input_shape[1]

    diff = target_freqs - input_freqs
    if diff % 2 == 0:
        pad = [(diff/2, diff/2)]
    else:
        pad = [(diff//2, diff//2 + 1)] # Add extra frequency bin at the end

    if len(target_shape) == 2:
        pad = pad + [(0,0)]
    else:
        pad = [(0,0)] + pad + [(0,0), (0,0)]

    if isinstance(tensor, tf.Tensor):
        return tf.pad(tensor, pad, mode='constant', constant_values=0.0)
    else:
        return np.pad(tensor, pad, mode='constant', constant_values=0.0)

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)