import os
import numpy as np
import tensorflow as tf
import cv2
import time
import argparse

from pggan import PGGAN

def main(args):
    pggan = PGGAN()
    z = tf.placeholder(tf.float32, [None, 1, 1, 512])
    alpha = tf.constant(1.0)
    fakes = [pggan.generator(z, alpha, stage=i+1) for i in range(9)]
    fake = fakes[args.stage-1]

    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    saver = tf.train.Saver()
    saver.restore(sess, args.model_path)

    batch_size = [256, 128, 64, 32, 16, 8, 4, 2, 1][args.stage-1]

    while True:
        z_batch = np.random.normal(size=[batch_size, 1, 1, 512])
        out = fake.eval(feed_dict={z: z_batch}, session=sess)[0]
        out = np.tanh(out)
        out = np.array((out + 1) * 127.5, dtype=np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        dst = os.path.join(args.output_dir, '{}.jpg'.format(int(time.time() * 1000)))
        cv2.imwrite(dst, out)
        cv2.imshow('', out)
        cv2.waitKey(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--stage', type=int, required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--gpu', type=str, default='0')
    os.environ['CUDA_VISIBLE_DEVICES'] = parser.parse_args().gpu
    main(parser.parse_args())
