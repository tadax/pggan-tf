import numpy as np
import cv2

def random_flip(img):
    if np.random.uniform() > 0.5:
        return cv2.flip(img, 1)
    else:
        return img

def normalize(img):
    img = img / 127.5 - 1
    return img

def augment(img, image_size):
    img = cv2.resize(img, (image_size, image_size),
                     interpolation=cv2.INTER_CUBIC)
    img = random_flip(img)
    img = normalize(img)
    return img
