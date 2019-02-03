import tensorflow as tf
from utils.layer import *

class PGGAN:
    def __init__(self):
        self.channels = [512, 512, 512, 512, 512, 256, 128, 64, 32, 16]
        
    def generator(self, x, alpha, stage):
        c = self.channels
        with tf.variable_scope('generator'):
            for i in range(1, stage+1, 1):
                _reuse = False if stage == i else True
                if i == 1:
                    with tf.variable_scope('stage1', reuse=_reuse):
                        with tf.variable_scope('conv1'):
                            x = equalized_conv_layer(x, [4, 4, c[i-1], c[i]], 1, padding=[3, 3])
                            x = lrelu(x)
                        with tf.variable_scope('conv2'):
                            x = equalized_conv_layer(x, [3, 3, c[i], c[i]], 1, padding=[1, 1])
                            x = feature_vector_normalization(x)
                            x = lrelu(x)
                        with tf.variable_scope('toRGB'):
                            rgb = equalized_conv_layer(x, [1, 1, c[i], 3], 1)
                else:
                    with tf.variable_scope('stage{}'.format(i), reuse=_reuse):
                        x = upsample(x)
                        with tf.variable_scope('shortcut'):
                            shortcut = equalized_conv_layer(x, [1, 1, c[i-1], 3], 1)
                        with tf.variable_scope('conv1'):
                            x = equalized_conv_layer(x, [3, 3, c[i-1], c[i]], 1, padding=[1, 1])
                            x = feature_vector_normalization(x)
                            x = lrelu(x)
                        with tf.variable_scope('conv2'):
                            x = equalized_conv_layer(x, [3, 3, c[i], c[i]], 1, padding=[1, 1])
                            x = feature_vector_normalization(x)
                            x = lrelu(x)
                        with tf.variable_scope('toRGB'):
                            rgb = equalized_conv_layer(x, [1, 1, c[i], 3], 1)
                            if stage == i:
                                rgb = rgb * alpha + shortcut * (1 - alpha)
        return rgb


    def discriminator(self, x, alpha, stage, reuse):
        c = self.channels
        with tf.variable_scope('discriminator'):
            for i in range(stage, 1, -1):
                _reuse = (False ^ reuse) if stage == i else True
                with tf.variable_scope('stage{}'.format(i), reuse=_reuse):
                    if i == stage:
                        with tf.variable_scope('shortcut'):
                            shortcut = avg_pooling_layer(x, 2, 2)
                            shortcut = equalized_conv_layer(shortcut, [1, 1, 3, c[i-1]], 1)
                            shortcut = lrelu(shortcut)
                        with tf.variable_scope('fromRGB'):
                            x = equalized_conv_layer(x, [1, 1, 3, c[i]], 1)
                            x = lrelu(x)
                        with tf.variable_scope('conv1'):
                            x = equalized_conv_layer(x, [3, 3, c[i], c[i]], 1, padding=[1, 1])
                            x = lrelu(x)
                        with tf.variable_scope('conv2'):
                            x = equalized_conv_layer(x, [3, 3, c[i], c[i-1]], 1, padding=[1, 1])
                            x = lrelu(x)
                        x = avg_pooling_layer(x, 2, 2)
                        x = x * alpha + shortcut * (1 - alpha)
                    else:
                        with tf.variable_scope('conv1'):
                            x = equalized_conv_layer(x, [3, 3, c[i], c[i]], 1, padding=[1, 1])
                            x = lrelu(x)
                        with tf.variable_scope('conv2'):
                            x = equalized_conv_layer(x, [3, 3, c[i], c[i-1]], 1, padding=[1, 1])
                            x = lrelu(x)
                        x = avg_pooling_layer(x, 2, 2)
            
            _reuse = (False ^ reuse) if stage == 1 else True
            with tf.variable_scope('stage1', reuse=_reuse):
                if stage == 1: 
                    with tf.variable_scope('fromRGB'):
                        x = equalized_conv_layer(x, [1, 1, 3, c[1]], 1)
                        x = lrelu(x)
                x = minibatch_std(x)
                with tf.variable_scope('conv1'):
                    x = equalized_conv_layer(x, [3, 3, c[1]+1, c[1]], 1, padding=[1, 1])
                    x = lrelu(x)
                with tf.variable_scope('conv2'):
                    x = equalized_conv_layer(x, [4, 4, c[1], c[0]], 1)
                    x = lrelu(x)
            
            with tf.variable_scope('fc{}'.format(stage), reuse=reuse):
                x = flatten_layer(x)
                x = equalized_full_connection_layer(x, 1)

        return x
