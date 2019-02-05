import requests
import cv2
import numpy as np
import os
import argparse

def download(url):
    res = requests.get(url, timeout=10)
    arr = np.asarray(bytearray(res.content), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    return img

def main(args):
    with open(args.source) as f:
        lines = f.read().splitlines()
    for line in lines:
        index, url = line.split('\t')
        filename = '{0:05d}'.format(int(index)) + '.png'
        dst = os.path.join(args.output_dir, filename)
        if os.path.exists(dst):
            continue
        img = download(url)
        cv2.imwrite(dst, img)
        print(dst)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='urls.txt')
    parser.add_argument('--output_dir', required=True)
    main(parser.parse_args())
