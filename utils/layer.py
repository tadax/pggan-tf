import numpy as np
import tensorflow as tf

def lrelu(x):
    alpha = 0.2
    return tf.maximum(alpha * x, x)

def conv_layer(x, filter_shape, stride, padding=None):
    if padding is not None:
        x = tf.pad(x, [[0, 0], padding, padding, [0, 0]], 'CONSTANT')
    filter_ = tf.get_variable(
        name='weight', 
        shape=filter_shape,
        dtype=tf.float32, 
        initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0),
        trainable=True)
    return tf.nn.conv2d(
        input=x,
        filter=filter_,
        strides=[1, stride, stride, 1],
        padding='VALID')

def equalized_conv_layer(x, filter_shape, stride, padding=None):
    in_ch = filter_shape[2]
    ksize = filter_shape[0]
    c = np.sqrt(2.0 / (in_ch * ksize ** 2))
    return conv_layer(x * c, filter_shape, stride, padding)

def full_connection_layer(x, out_ch):
    in_ch = x.get_shape().as_list()[-1]
    W = tf.get_variable(
        name='weight',
        shape=[in_ch, out_ch],
        dtype=tf.float32,
        initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0),
        trainable=True)
    b = tf.get_variable(
        name='bias',
        shape=[out_ch],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=True)
    return tf.add(tf.matmul(x, W), b)

def equalized_full_connection_layer(x, out_ch):
    in_ch = x.shape.as_list()[-1]
    c = np.sqrt(2.0 / in_ch) 
    return full_connection_layer(x * c, out_ch)

def feature_vector_normalization(x):
    alpha = 1.0 / tf.sqrt(tf.reduce_mean(tf.square(x), axis=3, keepdims=True) + 1e-8)
    return x * alpha

def flatten_layer(x):
    input_shape = x.get_shape().as_list()
    dim = input_shape[1] * input_shape[2] * input_shape[3]
    transposed = tf.transpose(x, (0, 3, 1, 2))
    return tf.reshape(transposed, [-1, dim])

def avg_pooling_layer(x, size, stride):
    return tf.nn.avg_pool(value=x, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='VALID')

def minibatch_std(x):
    ch_in = tf.shape(x)[0]
    group_size = tf.minimum(4, ch_in)
    y = tf.reshape(x, [group_size, -1, x.shape[1], x.shape[2], x.shape[3]]) # [GMHWC] N = G * M
    mean = tf.reduce_mean(y, axis=0, keepdims=True)
    dev = tf.reduce_mean(tf.square(y - mean), axis=0) # [MHWC]
    std = tf.reduce_mean(tf.sqrt(dev + 1e-8), axis=[1, 2, 3], keepdims=True) # [M111]
    features = tf.tile(std, [group_size, x.shape[1], x.shape[2], 1]) # [NHW1]
    return tf.concat([x, features], 3)

def upsample(x):
    sh = x.shape
    x = tf.image.resize_images(x, [sh[1] * 2, sh[2] * 2], method=tf.image.ResizeMethod.BILINEAR)
    return x
