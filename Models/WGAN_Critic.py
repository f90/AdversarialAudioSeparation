import tensorflow as tf

from Utils import LeakyReLU

def create_critic(model_config, real_input, fake_input, scope, network_func):
    '''
    Function creating the Wasserstein critic loss taking real source samples and fake (generator/separator) source samples, given a network architecture.
    :param model_config: Experiment configuration
    :param real_input: Batch of real source samples [n_batch, freqs, t, 1]
    :param fake_input: Batch of fake source samples [n_batch, freqs, t, 1] from the separator network
    :param scope: Tensorflow scope that the network should be created with
    :param network_func: Function yielding the network output for a given input (contains graph of network architecture)
    :return: List: Discriminator loss, loss for real samples, loss for fake samples, gradient penalty term, Wasserstein distance
    '''
    # Discriminators
    disc_real = network_func(real_input, scope, reuse=False)
    disc_fake = network_func(fake_input, scope)

    # WGAN grad penalty
    eps = tf.random_uniform([model_config["batch_size"], 1, 1, 1], minval=0., maxval=1.)
    real_shuffle = tf.random_shuffle(real_input)
    interp = real_shuffle + (eps * (fake_input - real_shuffle))
    disc_interp = network_func(interp, scope)
    grad = tf.gradients(disc_interp, [interp])[0]
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
    grad_pen = tf.reduce_mean(tf.maximum(grad_norm - 1., 0.) ** 2)

    wasserstein_dist = tf.reduce_mean(disc_real) - tf.reduce_mean(disc_fake)
    disc_loss = -wasserstein_dist + model_config["lam"] * grad_pen

    tf.summary.scalar(scope + "_disc_loss", disc_loss, collections=[scope])
    tf.summary.scalar(scope + "_disc_real", tf.reduce_mean(disc_real), collections=[scope])
    tf.summary.scalar(scope + "_disc_fake", tf.reduce_mean(disc_fake), collections=[scope])
    tf.summary.scalar(scope + "_wasserstein_dist", wasserstein_dist, collections=[scope])
    tf.summary.scalar(scope + "_grad_pen", grad_pen, collections=[scope])

    tf.summary.image(scope + "_real_input", real_input, collections=[scope])
    tf.summary.image(scope + "_fake_input", fake_input, collections=[scope])

    return disc_loss, disc_real, disc_fake, grad_pen, wasserstein_dist

def dcgan(input, name, reuse=True):
    '''
    Adapted DCGAN discriminator architecture.
    :param input: 4D tensor [batch_size, freqs, time_frames, 1]
    :param name: Tensorflow scope to create variables under
    :param reuse: Whether to create new parameter variables or reuse existing ones
    :return: DCGAN output in the form of unnormalised logits (4D tensor)
    '''
    with tf.variable_scope(name, reuse=reuse):
        filters = 32

        # Convolve with stride 2 until one of the dimensions is 4 or less
        conv = tf.layers.conv2d(input, filters, [4,4], strides=[2,2], padding='same', activation=LeakyReLU, use_bias=True)
        while conv.get_shape().as_list()[1] > 4 and conv.get_shape().as_list()[2] > 4:
            filters *= 2
            conv = tf.layers.conv2d(conv, filters, [4,4], strides=[2,2], padding='same', activation=LeakyReLU, use_bias=True)

        # Convolve with stride 2 only along frequency axis until frequency axis is length 4 or less
        while conv.get_shape().as_list()[1] > 4:
            conv = tf.layers.conv2d(conv, filters, [4,2], strides=[2,1], padding='same', activation=LeakyReLU, use_bias=True)

        conv = tf.contrib.layers.flatten(conv)
        hidden = tf.layers.dense(conv, 32, activation=LeakyReLU, use_bias=True)
        logits = tf.layers.dense(hidden, 1, activation=None, use_bias=False)
        return logits