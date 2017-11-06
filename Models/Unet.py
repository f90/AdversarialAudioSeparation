import tensorflow as tf

import Utils
import numpy as np

class Unet:
    '''
    U-Net separator network for singing voice separation.
    Takes in the mixture magnitude spectrogram and return estimates of the accompaniment and voice magnitude spectrograms.
    Uses valid convolutions, so it predicts for the centre part of the input - only certain input and output shapes are therefore possible (see getUnetPadding)
    '''

    def __init__(self, num_layers):
        '''
        Initialize U-net
        :param num_layers: Number of down- and upscaling layers in the network 
        '''
        assert(num_layers > 0)
        self.num_layers = num_layers

    def getUnetPadding(self, shape):
        '''
        Calculates the required amounts of padding along each axis of the input and output, so that the Unet works and has the given shape as output shape
        :param shape: Desired output shape 
        :return: Padding along each axis (total): (Input frequency, input time)
        '''

        # Check if desired shape is possible as output shape - go from output shape towards lowest-res feature map
        rem = shape[1:3] # Cut off batch size number and channel
        for i in range(self.num_layers):
            rem += 2 # Conv
            if np.sum(rem % 2) > 0:
                print("Warning: U-Net cannot be constructed with desired architecture and output shape - Padding output accordingly")
                rem = np.asarray(rem, dtype=np.float32)
            rem = (rem / 2) #Transposed-up-conv
        # Round resulting feature map dimensions up to nearest EVEN integer (even because up-convolution by factor two is needed)
        x = np.asarray(np.ceil(rem),dtype=np.int64)
        x += (x % 2)

        # Compute input and output shapes based on lowest-res feature map
        output_shape = x
        input_shape = x
        for i in range(self.num_layers):
            output_shape = output_shape * 2 - 2
            input_shape = (input_shape + 2) * 2
        #output_shape -= 2 # Final conv has filter size 1 => no loss
        input_shape += 2 # First conv

        input_shape = np.concatenate([[shape[0]], input_shape, [1]])
        output_shape = np.concatenate([[shape[0]], output_shape, [1]])

        return input_shape, output_shape

    def get_output(self, input, reuse=True):
        '''
        Creates symbolic computation graph of the U-Net for a given input batch
        :param input: Input batch of mixtures, 4D tensor [batch_size, freqs, time_frames, 1]
        :param reuse: Whether to create new parameter variables or reuse existing ones
        :return: U-Net output: Log-normalized accompaniment and voice magnitudes as two 4D tensors
        '''
        NUM_INITIAL_FILTERS = 16
        with tf.variable_scope("separator", reuse=reuse):
            enc_outputs = list()

            current_layer = tf.layers.conv2d(input, NUM_INITIAL_FILTERS, 3, activation=tf.nn.relu, padding='valid') # - 2
            enc_outputs.append(current_layer)
            # Down-convolution: Repeat pool-conv
            for i in range(self.num_layers):
                assert(current_layer.get_shape().as_list()[1] % 2 == 0 and current_layer.get_shape().as_list()[2] % 2 == 0)
                current_layer = tf.layers.max_pooling2d(current_layer, pool_size=2, strides=2, padding='valid', data_format='channels_last') # MAXPOOL # :2
                current_layer = tf.layers.conv2d(current_layer, NUM_INITIAL_FILTERS * (2 ** (i+1)), 3, activation=tf.nn.relu, padding='valid') # CONV # -2
                if i < self.num_layers - 1:
                    enc_outputs.append(current_layer)

            # Upconvolution
            for i in range(self.num_layers):
                assert (current_layer.get_shape().as_list()[1] % 2 == 0 and current_layer.get_shape().as_list()[2] % 2 == 0)
                # Repeat: Up-convolution (transposed conv with stride), copy-and-crop feature map from down-ward path, convolution to combine both feature maps
                current_layer = tf.layers.conv2d_transpose(current_layer, NUM_INITIAL_FILTERS*(2**(self.num_layers-i-1)), 2, strides=2, activation=tf.nn.relu, padding='valid') # *2
                current_layer = Utils.crop_and_concat(enc_outputs[-i-1], current_layer) #tf.concat([enc_outputs[-i - 1], current_layer], axis=3)
                current_layer = tf.layers.conv2d(current_layer, NUM_INITIAL_FILTERS * (2 ** (self.num_layers - i - 1)), 3, activation=tf.nn.relu, padding='valid') # - 2

            # Output layer
            current_layer = Utils.crop_and_concat(input, current_layer) # Passing the input to the final feature map helps especially when we output sources directly

            acc_norm = tf.layers.conv2d(current_layer, 1, 1, activation=tf.nn.relu, padding='valid')  # 0
            voice_norm = tf.layers.conv2d(current_layer, 1, 1, activation=tf.nn.relu, padding='valid')  # 0

            return acc_norm, voice_norm