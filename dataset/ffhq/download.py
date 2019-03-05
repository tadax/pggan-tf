import requests
import cv2
import numpy as np
import os
from queue import Queue
import threading
import argparse

def download(q, output_dir):
    while not q.empty():
        index, url = q.get().split('\t')

        filename = '{0:05d}'.format(int(index)) + '.png'
        dst = os.path.join(output_dir, filename)
        if os.path.exists(dst):
            continue
        
        try:
            res = requests.get(url, proxies=None, timeout=10)
        except KeyboardInterrupt:
            exit()
        except:
            continue

        if res.status_code != 200:
            continue

        try:
            arr = np.asarray(bytearray(res.content), dtype=np.uint8)
            img = cv2.imdecode(arr, -1)
        except:
            continue

        if img is None:
            print('Failed: {}'.format(url))
        else:
            cv2.imwrite(dst, img)
            print(dst)
    
def main(args):
    with open(args.source) as f:
        lines = f.read().splitlines()

    q = Queue()
    for line in lines:
        q.put(line)

    for _ in range(args.threads):
        th = threading.Thread(target=download, args=(q, args.output_dir,))
        th.start()
    for th in threading.enumerate():
        if th != threading.main_thread():
            th.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='urls.txt')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--threads', type=int, default=10)
    main(parser.parse_args())
