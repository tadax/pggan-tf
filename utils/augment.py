import numpy as np
import cv2

def random_crop(img, image_size, ratio=0.05):
    margin = int(image_size * ratio)
    img = cv2.resize(img, (image_size + margin, image_size + margin), 
                     interpolation=cv2.INTER_LINEAR)
    x1, x2 = np.random.randint(0, margin + 1, 2)
    img = img[x1:x1+image_size, x2:x2+image_size, :]
    return img

def random_flip(img):
    if np.random.uniform() > 0.5:
        return cv2.flip(img, 1)
    else:
        return img

def normalize(img):
    img = img / 127.5 - 1
    return img

def augment(img, image_size):
    img = random_crop(img, image_size)
    img = random_flip(img)
    img = normalize(img)
    return img
