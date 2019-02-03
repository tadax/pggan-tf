import numpy as np
import cv2
import os
import glob
import argparse

def main(args):
    paths = glob.glob(os.path.join(args.input_dir, '*'))
    for i, path in enumerate(paths):
        filename = os.path.splitext(os.path.basename(path))[0] + '.npy'
        dst = os.path.join(args.output_dir, filename)
        if os.path.exists(dst):
            continue
        npimg = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        np.save(dst, img)
        print('{}/{} - {}'.format(i+1, len(paths), dst))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    main(parser.parse_args())
