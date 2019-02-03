import cv2
import numpy as np
import tensorflow as tf

def conv_layer(inp, filters, kernel_size, strides, padding, name):
    output = tf.layers.conv2d(
        inp, filters=filters, kernel_size=kernel_size, 
        strides=strides, padding=padding, name=name)
    return output

def prelu_layer(inp, name):
    with tf.variable_scope(name):
        alpha = tf.get_variable('alpha', int(inp.shape[-1]))
    output = tf.nn.relu(inp) + tf.multiply(alpha, -tf.nn.relu(-inp))
    return output

def max_pooling_layer(inp, pool_size, strides, padding):
    output = tf.layers.max_pooling2d(
        inp, pool_size=pool_size, strides=strides, padding=padding)
    return output

def softmax_layer(inp, axis):
    max_axis = tf.reduce_max(inp, axis, keep_dims=True)
    inp_exp = tf.exp(inp - max_axis)
    normalize = tf.reduce_sum(inp_exp, axis, keep_dims=True)
    softmax = tf.div(inp_exp, normalize)
    return softmax

def full_connection_layer(inp, units, name):
    if inp.shape.ndims == 4:
        dim = np.prod(inp.shape.as_list()[1:])
        inp = tf.reshape(inp, [-1, dim])
    output = tf.layers.dense(inp, units=units, name=name)
    return output

class MTCNN(object):
    def __init__(self, model_path, use_gpu=False):
        self.model_path = model_path
        self.min_size = 20
        self.threshold = [0.6, 0.7, 0.7]
        self.pnet_inp = tf.placeholder(tf.float32, (None, None, None, 3))
        self.rnet_inp = tf.placeholder(tf.float32, (None, 24, 24, 3))
        self.onet_inp = tf.placeholder(tf.float32, (None, 48, 48, 3))
        if use_gpu:
            config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        else:
            config = tf.ConfigProto(device_count={'GPU': 0})
        self.sess = tf.Session(config=config)
        self.build_network()
        
    def build_network(self):
        with tf.variable_scope('mtcnn'):
            self.pnet_out = self.pnet(self.pnet_inp)
            self.rnet_out = self.rnet(self.rnet_inp)
            self.onet_out = self.onet(self.onet_inp)

        to_restore_variables = []
        for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            if v.name.startswith('mtcnn'):
                to_restore_variables.append(v)
        init_op = tf.variables_initializer(to_restore_variables)
        self.sess.run(init_op)
        saver = tf.train.Saver(to_restore_variables)
        saver.restore(self.sess, self.model_path)

    def pnet(self, inp):
        with tf.variable_scope('pnet'):
            conv1 = conv_layer(inp, 10, 3, 1, 'VALID', 'conv1')
            prelu1 = prelu_layer(conv1, 'prelu1')
            pool1 = max_pooling_layer(prelu1, 2, 2, 'SAME')
            conv2 = conv_layer(pool1, 16, 3, 1, 'VALID', 'conv2')
            prelu2 = prelu_layer(conv2, 'prelu2')
            conv3 = conv_layer(prelu2, 32, 3, 1, 'VALID', 'conv3')
            prelu3 = prelu_layer(conv3, 'prelu3')
            conv4_1 = conv_layer(prelu3, 2, 1, 1, 'SAME', 'conv4_1')
            conv4_2 = conv_layer(prelu3, 4, 1, 1, 'SAME', 'conv4_2')
            prob = softmax_layer(conv4_1, 3)
            return [conv4_2, prob]

    def rnet(self, inp):
        with tf.variable_scope('rnet'):
            conv1 = conv_layer(inp, 28, 3, 1, 'VALID', 'conv1')
            prelu1 = prelu_layer(conv1, 'prelu1')
            pool1 = max_pooling_layer(prelu1, 3, 2, 'SAME')
            conv2 = conv_layer(pool1, 48, 3, 1, 'VALID', 'conv2')
            prelu2 = prelu_layer(conv2, 'prelu2')
            pool2 = max_pooling_layer(prelu2, 3, 2, 'VALID')
            conv3 = conv_layer(pool2, 64, 2, 1, 'VALID', 'conv3')
            prelu3 = prelu_layer(conv3, 'prelu3')
            fc4 = full_connection_layer(prelu3, 128, 'fc4')
            prelu4 = prelu_layer(fc4, 'prelu4')
            fc5_1 = full_connection_layer(prelu4, 2, 'fc5_1')
            fc5_2 = full_connection_layer(prelu4, 4, 'fc5_2')
            prob = softmax_layer(fc5_1, 1)
            return [fc5_2, prob]

    def onet(self, inp):
        with tf.variable_scope('onet'):
            conv1 = conv_layer(inp, 32, 3, 1, 'VALID', 'conv1')
            prelu1 = prelu_layer(conv1, 'prelu1')
            pool1 = max_pooling_layer(prelu1, 3, 2, 'SAME')
            conv2 = conv_layer(pool1, 64, 3, 1, 'VALID', 'conv2')
            prelu2 = prelu_layer(conv2, 'prelu2')
            pool2 = max_pooling_layer(prelu2, 3, 2, 'VALID')
            conv3 = conv_layer(pool2, 64, 3, 1, 'VALID', 'conv3')
            prelu3 = prelu_layer(conv3, 'prelu3')
            pool3 = max_pooling_layer(prelu3, 2, 2, 'SAME')
            conv4 = conv_layer(pool3, 128, 2, 1, 'VALID', 'conv4')
            prelu4 = prelu_layer(conv4, 'prelu4')
            fc5 = full_connection_layer(prelu4, 256, 'fc5')
            prelu5 = prelu_layer(fc5, 'prelu5')
            fc6_1 = full_connection_layer(prelu5, 2, 'fc6_1')
            fc6_2 = full_connection_layer(prelu5, 4, 'fc6_2')
            fc6_3 = full_connection_layer(prelu5, 10, 'fc6_3')
            prob = softmax_layer(fc6_1, 1)
            return [fc6_2, fc6_3, prob]

    def detect(self, img, factor=0.709):
        factor_count = 0
        total_boxes = np.empty((0, 9))
        points = np.empty(0)
        h, w = img.shape[:2]
        ratio = 12 / self.min_size
        length = min(h, w) * ratio
        scales = []
        while length >= 12:
            scales += [ratio * np.power(factor, factor_count)]
            length *= factor
            factor_count += 1

        # First Stage
        for scale in scales:
            hs = int(np.ceil(h * scale))
            ws = int(np.ceil(w * scale))
            resized = cv2.resize(img, (ws, hs), interpolation=cv2.INTER_AREA)
            normalized = (resized - 127.5) * 0.0078125
            inp = np.transpose(np.asarray([normalized]), (0, 2, 1, 3))
            pnet_out = self.sess.run(self.pnet_out, feed_dict={self.pnet_inp: inp})
            boxes = self.generate_boundingbox(pnet_out, scale, self.threshold[0])

            pick = self.nms(boxes.copy(), 0.5, 'Union')
            if boxes.size > 0 and pick.size > 0:
                boxes = boxes[pick, :]
                total_boxes = np.append(total_boxes, boxes, axis=0)

        num_box = total_boxes.shape[0]
        if num_box == 0:
            return total_boxes, points

        pick = self.nms(total_boxes.copy(), 0.7, 'Union')
        total_boxes = total_boxes[pick, :]
        regw = total_boxes[:, 2] - total_boxes[:, 0]
        regh = total_boxes[:, 3] - total_boxes[:, 1]
        qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
        qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
        qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
        qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
        total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:, 4]]))
        total_boxes = self.rerec(total_boxes.copy())
        total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = self.pad(total_boxes.copy(), w, h)

        num_box = total_boxes.shape[0]
        if num_box == 0:
            return total_boxes, points

        # Second Stage
        tmpimg = np.zeros((24, 24, 3, num_box))
        for k in range(0, num_box):
            tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
            tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k], :] = img[y[k] - 1:ey[k], x[k] - 1:ex[k], :]
            if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                tmpimg[:, :, :, k] = cv2.resize(tmp, (24, 24), interpolation=cv2.INTER_AREA)
            else:
                return np.empty() # ???

        tmpimg = (tmpimg - 127.5) * 0.0078125
        tmpimg = np.transpose(tmpimg, (3, 1, 0, 2))
        rnet_out = self.sess.run(self.rnet_out, feed_dict={self.rnet_inp: tmpimg})
        out0 = np.transpose(rnet_out[0])
        out1 = np.transpose(rnet_out[1])
        score = out1[1, :]
        ipass = np.where(score > self.threshold[1])
        total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])
        mv = out0[:, ipass[0]]
        if total_boxes.shape[0] > 0:
            pick = self.nms(total_boxes, 0.7, 'Union')
            total_boxes = total_boxes[pick, :]
            total_boxes = self.bbreg(total_boxes.copy(), np.transpose(mv[:, pick]))
            total_boxes = self.rerec(total_boxes.copy())

        num_box = total_boxes.shape[0]
        if num_box == 0:
            return total_boxes, points

        # Third Stage
        total_boxes = np.fix(total_boxes).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = self.pad(total_boxes.copy(), w, h)
        tmpimg = np.zeros((48, 48, 3, num_box))
        for k in range(0, num_box):
            tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
            tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k], :] = img[y[k] - 1:ey[k], x[k] - 1:ex[k], :]
            if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                tmpimg[:, :, :, k] = cv2.resize(tmp, (48, 48), interpolation=cv2.INTER_AREA)
            else:
                return np.empty() # ???
        tmpimg = (tmpimg - 127.5) * 0.0078125
        tmpimg = np.transpose(tmpimg, (3, 1, 0, 2))
        onet_out = self.sess.run(self.onet_out, feed_dict={self.onet_inp: tmpimg})
        out0 = np.transpose(onet_out[0])
        out1 = np.transpose(onet_out[1])
        out2 = np.transpose(onet_out[2])
        score = out2[1, :]
        points = out1
        ipass = np.where(score > self.threshold[2])
        points = points[:, ipass[0]]
        total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])
        mv = out0[:, ipass[0]]

        w = total_boxes[:, 2] - total_boxes[:, 0] + 1
        h = total_boxes[:, 3] - total_boxes[:, 1] + 1
        points[0:5, :] = np.tile(w, (5, 1)) * points[0:5, :] + np.tile(total_boxes[:, 0], (5, 1)) - 1
        points[5:10, :] = np.tile(h, (5, 1)) * points[5:10, :] + np.tile(total_boxes[:, 1], (5, 1)) - 1
        if total_boxes.shape[0] > 0:
            total_boxes = self.bbreg(total_boxes.copy(), np.transpose(mv))
            pick = self.nms(total_boxes.copy(), 0.7, 'Min')
            total_boxes = total_boxes[pick, :]
            points = points[:, pick]

        return total_boxes, points

    def generate_boundingbox(self, outs, scale, threshold):
        stride = 2
        cell_size = 12

        out0 = np.transpose(outs[0], (0, 2, 1, 3))
        out1 = np.transpose(outs[1], (0, 2, 1, 3))
        reg = out0[0, :, :, :]
        imap = np.transpose(out1[0, :, :, 1])
        dx1 = np.transpose(reg[:, :, 0])
        dy1 = np.transpose(reg[:, :, 1])
        dx2 = np.transpose(reg[:, :, 2])
        dy2 = np.transpose(reg[:, :, 3])
        y, x = np.where(imap >= threshold)
        if y.shape[0] == 1:
            dx1 = np.flipud(dx1)
            dy1 = np.flipud(dy1)
            dx2 = np.flipud(dx2)
            dy2 = np.flipud(dy2)
        score = imap[(y, x)]
        reg = np.transpose(np.vstack([dx1[(y, x)], dy1[(y, x)], dx2[(y, x)], dy2[(y, x)]]))
        if reg.size == 0:
            reg = np.empty((0, 3))
        bb = np.transpose(np.vstack([y, x]))
        q1 = np.fix((stride * bb + 1) / scale)
        q2 = np.fix((stride * bb + cell_size) / scale)
        boundingbox = np.hstack([q1, q2, np.expand_dims(score, 1), reg])
        return boundingbox

    def nms(self, boxes, threshold, method):
        if boxes.size == 0:
            return np.empty((0, 3))

        #x1, y1, x2, y2, s = boxes.T
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        s = boxes[:, 4]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        I = np.argsort(s)
        pick = np.zeros_like(s, dtype=np.int16)
        counter = 0
        while I.size > 0:
            i = I[-1]
            pick[counter] = i
            counter += 1
            
            idx = I[0:-1]
            xx1 = np.maximum(x1[i], x1[idx])
            yy1 = np.maximum(y1[i], y1[idx])
            xx2 = np.minimum(x2[i], x2[idx])
            yy2 = np.minimum(y2[i], y2[idx])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            if method is 'Min':
                o = inter / np.minimum(area[i], area[idx])
            else:
                o = inter / (area[i] + area[idx] - inter)
            I = I[np.where(o <= threshold)]

        pick = pick[0:counter]
        return pick

    def pad(self, total_boxes, w, h):
        tmpw = (total_boxes[:, 2] - total_boxes[:, 0] + 1).astype(np.int32)
        tmph = (total_boxes[:, 3] - total_boxes[:, 1] + 1).astype(np.int32)
        num_box = total_boxes.shape[0]

        dx = np.ones((num_box), dtype=np.int32)
        dy = np.ones((num_box), dtype=np.int32)
        edx = tmpw.copy()
        edy = tmph.copy()
        
        x = total_boxes[:, 0].copy().astype(np.int32)
        y = total_boxes[:, 1].copy().astype(np.int32)
        ex = total_boxes[:, 2].copy().astype(np.int32)
        ey = total_boxes[:, 3].copy().astype(np.int32)

        tmp = np.where(ex > w)
        edx.flat[tmp] = np.expand_dims(-ex[tmp] + w + tmpw[tmp], 1)
        ex[tmp] = w

        tmp = np.where(ey > h)
        edy.flat[tmp] = np.expand_dims(-ey[tmp] + h + tmph[tmp], 1)
        ey[tmp] = h

        tmp = np.where(x < 1)
        dx.flat[tmp] = np.expand_dims(2 - x[tmp], 1)
        x[tmp] = 1

        tmp = np.where(y < 1)
        dy.flat[tmp] = np.expand_dims(2 - y[tmp], 1)
        y[tmp] = 1

        return dy, edy, dx, edx, y, ey, x, ex ,tmpw, tmph

    def rerec(self, bbox):
        h = bbox[:, 3] - bbox[:, 1]
        w = bbox[:, 2] - bbox[:, 0]
        l = np.maximum(w, h)
        bbox[:, 0] = bbox[:, 0] + w * 0.5 - l * 0.5
        bbox[:, 1] = bbox[:, 1] + h * 0.5 - l * 0.5
        bbox[:, 2:4] = bbox[:, 0:2] + np.transpose(np.tile(l, (2, 1)))
        return bbox

    def bbreg(self, boundingbox, reg):
        if reg.shape[1] == 1:
            reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))

        w = boundingbox[:, 2] - boundingbox[:, 0] + 1
        h = boundingbox[:, 3] - boundingbox[:, 1] + 1
        b1 = boundingbox[:, 0] + reg[:, 0] * w
        b2 = boundingbox[:, 1] + reg[:, 1] * h
        b3 = boundingbox[:, 2] + reg[:, 2] * w
        b4 = boundingbox[:, 3] + reg[:, 3] * h
        boundingbox[:, 0:4] = np.transpose(np.vstack([b1, b2, b3, b4]))
        return boundingbox
