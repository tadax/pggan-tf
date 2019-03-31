import os
import numpy as np
import tensorflow as tf
import cv2
import logging
import argparse

from pggan import PGGAN
from utils.data import Data
from utils.loss import calc_losses

def main(args):
    logger = logging.getLogger()
    hdlr = logging.FileHandler(args.log_path)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(threadName)-10s] %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    databox = Data(args.input_dir)
    dataset_size = databox.size
    logger.info('Dataset size: {}'.format(dataset_size))

    pggan = PGGAN()

    resolutions = [2**(i+2) for i in range(9)] 
    z = tf.placeholder(tf.float32, [None, 1, 1, 512])
    reals = [tf.placeholder(tf.float32, [None, r, r, 3]) for r in resolutions]
    alpha = tf.placeholder(tf.float32, [])

    fakes = [pggan.generator(z, alpha, stage=i+1) for i in range(9)]
    d_reals = [pggan.discriminator(x, alpha, stage=i+1, reuse=False) for i, x in enumerate(reals)]
    d_fakes = [pggan.discriminator(x, alpha, stage=i+1, reuse=True) for i, x in enumerate(fakes)]

    xhats = []
    d_xhats = []
    for i, (real, fake) in enumerate(zip(reals, fakes)):
        epsilon = tf.random_uniform(shape=[tf.shape(real)[0], 1, 1, 1], minval=0.0, maxval=1.0)
        inter = real * epsilon + fake * (1 - epsilon)
        d_xhat = pggan.discriminator(inter, alpha, stage=i+1, reuse=True)
        xhats.append(inter)
        d_xhats.append(d_xhat)

    g_losses, d_losses = calc_losses(d_reals, d_fakes, xhats, d_xhats)

    g_var_list = []
    d_var_list = []
    for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        if ('generator' in v.name):
            g_var_list.append(v)
        elif ('discriminator' in v.name):
            d_var_list.append(v)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    opt = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.0, beta2=0.99, epsilon=1e-8)
    g_train_op = [opt.minimize(loss, global_step=global_step, var_list=g_var_list) for loss in g_losses]
    d_train_op = [opt.minimize(loss, global_step=global_step, var_list=d_var_list) for loss in d_losses]

    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    if args.resume:
        saver = tf.train.Saver()
        saver.restore(sess, args.resume)
        logger.info('Resuming training')
    if args.finetuning:
        sess.run(global_step.assign(0))
        logger.info('Fine-tuning')

    stage_steps = [
        int(epoch * dataset_size / batch_size) 
        for epoch, batch_size 
        in zip(args.epochs, args.batch_sizes)
    ]

    current_stage = None
    while True:
        step = int(sess.run(global_step) / 2)
        if step >= sum(stage_steps):
            logger.info('Done!')
            break

        for i in range(len(stage_steps)):
            if step < sum(stage_steps[:i+1]):
                stage = i
                break

        image_size = resolutions[i] 
        if current_stage != stage:
            if current_stage is not None:
                databox.terminate()
            databox.start(image_size)
            current_stage = stage

        progress = step + 1 - sum(stage_steps[:i])
        logger.info('step: {}/{} - {}x{} (stage {})'.format(
            progress, stage_steps[i], image_size, image_size, stage+1))

        current_stage_step = stage_steps[stage]
        current_stage_progress = step - sum(stage_steps[:stage])
        delta = 4 / current_stage_step # 25 %
        if stage == 0:
            alp = 1.0
        else:
            alp = min(current_stage_progress * delta, 1.0)
        
        x_batch = databox.get(args.batch_sizes[stage])
        z_batch = np.random.normal(size=[args.batch_sizes[stage], 1, 1, 512])
        _, d_loss = sess.run([d_train_op[stage], d_losses[stage]], 
                             feed_dict={reals[stage]: x_batch, z: z_batch, alpha: alp})

        z_batch = np.random.normal(size=[args.batch_sizes[stage], 1, 1, 512])
        _, g_loss = sess.run([g_train_op[stage], g_losses[stage]], feed_dict={z: z_batch, alpha: alp})
    
        if progress % 1000 == 0:
            saver = tf.train.Saver()
            saver.save(sess, os.path.join(args.weights_dir, 'latest'), write_meta_graph=False)

            z_batch = np.random.normal(size=[args.batch_sizes[stage], 1, 1, 512])
            out = fakes[stage].eval(feed_dict={z: z_batch, alpha: 1.0}, session=sess)
            out = np.array((out[0] + 1) * 127.5, dtype=np.uint8)
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            outdir = os.path.join(args.output_dir, 'stage{}'.format(stage+1))
            os.makedirs(outdir, exist_ok=True)
            dst = os.path.join(outdir, '{}.png'.format('{0:09d}'.format(progress)))
            cv2.imwrite(dst, out)

        if int(sess.run(global_step) / 2) == sum(stage_steps[:stage+1]):
            saver = tf.train.Saver()
            saver.save(sess, os.path.join(args.weights_dir, 'stage{}'.format(stage+1)), write_meta_graph=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', nargs='+', required=True)
    parser.add_argument('--weights_dir', default='weights/')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--log_path', default='weights/out.log')
    parser.add_argument('--output_dir', default='weights/outputs/')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[64, 64, 64, 32, 16, 8, 4, 2, 1])
    parser.add_argument('--epochs', type=int, nargs='+', default=[0, 0, 60, 60, 60, 60, 90, 120, 150])
    parser.add_argument('--finetuning', action='store_true', default=False)
    parser.add_argument('--gpu', type=str, default='0')
    os.environ['CUDA_VISIBLE_DEVICES'] = parser.parse_args().gpu
    main(parser.parse_args())
