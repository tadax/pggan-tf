import requests
import cv2
import numpy as np
import os
import argparse

def download(url):
    try:
        res = requests.get(url, timeout=10)
    except KeyboardInterrupt:
        exit()
    except:
        return None

    if res.status_code != 200:
        return None

    try:
        arr = np.asarray(bytearray(res.content), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
    except:
        return None

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
        if img is None:
            print('Failed: {}'.format(url))
        else:
            cv2.imwrite(dst, img)
            print(dst)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='urls.txt')
    parser.add_argument('--output_dir', required=True)
    main(parser.parse_args())
