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
        img = cv2.imread(path)
        detected = mp.align(img)
        if detected is None:
            print('Not detected: {}'.format(path))
            continue
        if min(detected.shape[:2]) < args.image_size:
            print('Too small: {}'.format(path))
            continue

        scaled = cv2.resize(detected, (args.image_size, args.image_size),
                            interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(dst, scaled)
        print('{}/{} - {}'.format(i+1, len(paths), path))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--gpu', type=str)
    if parser.parse_args().gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = parser.parse_args().gpu
    main(parser.parse_args())
