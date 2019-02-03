import tensorflow as tf

def calc_losses(d_reals, d_fakes, xhats, d_xhats):
    g_losses = []
    d_losses = []
    for d_real, d_fake, xhat, d_xhat in zip(d_reals, d_fakes, xhats, d_xhats):
        g_loss = -tf.reduce_mean(d_fake)
        d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)

        drift_loss = tf.reduce_mean(d_real ** 2 * 1e-3)
        d_loss += drift_loss

        scale = 10.0
        grad = tf.gradients(d_xhat, [xhat])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.0) * scale)
        d_loss += gradient_penalty

        g_losses.append(g_loss)
        d_losses.append(d_loss)

    return g_losses, d_losses
