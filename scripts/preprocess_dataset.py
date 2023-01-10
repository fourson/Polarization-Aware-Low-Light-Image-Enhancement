import os
import random
import fnmatch

import numpy as np
import cv2


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def ensure_dirs(base_dir, dir_names):
    for dir_name in dir_names:
        ensure_dir(os.path.join(base_dir, dir_name))


def save_npy(f, save_dir, f_name):
    out_path = os.path.join(save_dir, f_name + '.npy')
    np.save(out_path, f)


def save_img(f, save_dir, f_name):
    out_path = os.path.join(save_dir, f_name + '.png')
    cv2.imwrite(out_path, cv2.cvtColor(f, cv2.COLOR_RGB2BGR) * 255)


def preprocess_raw(f):
    with open(f, 'rb') as raw_file:
        img = np.fromfile(raw_file, dtype=np.uint8)
        img = img.reshape((1024, 1224))
        img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)
        img = np.float32(img)
        img /= 255.
        img = img[:, 100:-100, :]  # (1024, 1024)
        return img


if __name__ == '__main__':
    in_dir = '../raw_images/for_test'
    out_base_dir = '../raw_images/data_test_temp'
    # in_dir = '../raw_images/for_train'
    # out_base_dir = '../raw_images/data_train_temp'

    ensure_dirs(out_base_dir, ['L1', 'L2', 'L3', 'L4', 'I1', 'I2', 'I3', 'I4'])
    lut_angle = {'0': '1', '45': '2', '90': '3', '135': '4'}

    for f in fnmatch.filter(os.listdir(in_dir), '*.raw'):
        print(f)
        img = preprocess_raw(os.path.join(in_dir, f))
        scene, time, angle_postfix = f.split('_')
        angle = angle_postfix[:-5]
        if int(time) < 10000:
            # short time
            mode = 'L'
        else:
            # long time
            mode = 'I'
        out_dir = os.path.join(out_base_dir, '{}{}'.format(mode, lut_angle[angle]))
        save_img(img, out_dir, scene + '_' + time)
