import cv2
import os
import glob
import argparse

from mp import MirrorPadding

def main(args):
    mp = MirrorPadding('shape_predictor_68_face_landmarks.dat')

    paths = glob.glob(os.path.join(args.input_dir, '*'))
    for i, path in enumerate(paths):
        dst = os.path.join(args.output_dir, os.path.basename(path))
        if os.path.exists(dst):
            continue
        print('{}/{} - {}'.format(i+1, len(paths), path))
        img = cv2.imread(path)
        detected = mp.align(img)
        if detected is not None:
            cv2.imwrite(dst, detected)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--gpu', type=str)
    if parser.parse_args().gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = parser.parse_args().gpu
    main(parser.parse_args())
