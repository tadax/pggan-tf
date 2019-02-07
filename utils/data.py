import multiprocessing
import os
import glob
import time
import cv2
import numpy as np 

from utils.augment import augment

class Data:
    def __init__(self, input_dir):
        self.queue_max_size = 1000
        self.input_dir = input_dir
        self.size = self.get_size()

    def start(self, image_size):
        self.image_size = image_size
        self.q = multiprocessing.Queue()
        self.p = multiprocessing.Process(target=self.put, args=(self.q, self.queue_max_size))
        self.p.start()

    def terminate(self):
        self.p.terminate()
        self.p.join()

    def get_paths(self):
        paths = []
        for folder in args.input_dir:
            paths += glob.glob(os.path.join(folder, '*'))
        return paths

    def get_size(self):
        return len(self.get_paths())

    def put(self, q, queue_max_size):
        paths = []
        while True:
            if len(paths) == 0:
                paths = self.get_paths()
            if q.qsize() >= queue_max_size:
                time.sleep(0.1)
                continue
            ix = np.random.randint(0, len(paths))
            path = paths.pop(ix)
            img = np.load(path)
            img = augment(img, self.image_size)
            q.put(img)

    def get(self, batch_size):
        x_batch = []
        for _ in range(batch_size):
            while True:
                if self.q.qsize() == 0:
                    time.sleep(1)
                    continue
                img = self.q.get()
                x_batch.append(img)
                break
        return np.asarray(x_batch)
