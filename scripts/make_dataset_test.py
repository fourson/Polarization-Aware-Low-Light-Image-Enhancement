import os

import numpy as np
import cv2


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def save_npy(f, save_dir, f_name):
    out_path = os.path.join(save_dir, f_name + '.npy')
    print(out_path)
    np.save(out_path, f)


def save_img(f, save_dir, f_name):
    out_path = os.path.join(save_dir, f_name + '.png')
    print(out_path)
    cv2.imwrite(out_path, cv2.cvtColor(f, cv2.COLOR_RGB2BGR) * 255)


def read_img(path):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = np.float32(img) / 255.
    return img


def compute_Si_from_Ii(I_list, compute_DoP_AoP=False):
    I1, I2, I3, I4 = I_list
    S0 = (I1 + I2 + I3 + I4) / 2  # I
    S1 = I3 - I1  # I*p*cos(2*theta)
    S2 = I4 - I2  # I*p*sin(2*theta)
    if compute_DoP_AoP:
        DoP = np.clip(np.sqrt(S1 ** 2 + S2 ** 2) / (S0 + 1e-7), a_min=0, a_max=1)  # in [0, 1]
        AoP = np.arctan2(S2, S1) / 2  # in [-pi/2, pi/2]
        AoP = (AoP < 0) * np.pi + AoP  # convert to [0, pi] by adding pi to negative values
        AoP = AoP.astype(np.float32)
        return [S0, S1, S2], [DoP, AoP]
    else:
        return [S0, S1, S2]


def compute_Ii_from_Si(S_list, compute_DoP_AoP=False):
    S0, S1, S2 = S_list
    I1 = (S0 - S1) / 2
    I2 = (S0 - S2) / 2
    I3 = (S0 + S1) / 2
    I4 = (S0 + S2) / 2
    if compute_DoP_AoP:
        DoP = np.clip(np.sqrt(S1 ** 2 + S2 ** 2) / (S0 + 1e-7), a_min=0, a_max=1)  # in [0, 1]
        AoP = np.arctan2(S2, S1) / 2  # in [-pi/2, pi/2]
        AoP = (AoP < 0) * np.pi + AoP  # convert to [0, pi] by adding pi to negative values
        AoP = AoP.astype(np.float32)
        return [I1, I2, I3, I4], [DoP, AoP]
    else:
        return [I1, I2, I3, I4]


def crop_and_scale(L_list, I_list, mode):
    # input size: 1024*1024
    if mode == 'h':
        func = lambda x: x[0::2, 0::2, :]
    elif mode == 'tl':
        func = lambda x: x[0:512, 0:512, :]
    elif mode == 'tr':
        func = lambda x: x[0:512, 512:1024, :]
    elif mode == 'bl':
        func = lambda x: x[512:1024, 0:512, :]
    elif mode == 'br':
        func = lambda x: x[512:1024, 512:1024, :]
    elif mode == 'mid':
        func = lambda x: x[256:768, 256:768, :]
    else:
        func = lambda x: x

    L_list_new = [func(L) for L in L_list]
    I_list_new = [func(I) for I in I_list]
    return L_list_new, I_list_new


if __name__ == '__main__':
    in_dir = '../raw_images/data_test_temp'
    out_dir = '../data/test'

    out_dir_L_list = [ensure_dir(os.path.join(out_dir, 'L{}'.format(i))) for i in range(1, 5)]
    out_dir_I_list = [ensure_dir(os.path.join(out_dir, 'I{}'.format(i))) for i in range(1, 5)]

    in_dir_L_list = [os.path.join(in_dir, 'L{}'.format(i)) for i in range(1, 5)]
    in_dir_I_list = [os.path.join(in_dir, 'I{}'.format(i)) for i in range(1, 5)]
    for f in os.listdir(in_dir_L_list[0]):
        print(f)
        scene, time_postfix = f.split('_')
        time = int(time_postfix[:-4])

        time_gt = time * 10
        f_gt = '{}_{}.png'.format(scene, time_gt)

        L_list = [read_img(os.path.join(in_dir_L, f)) for in_dir_L in in_dir_L_list]
        I_list = [read_img(os.path.join(in_dir_I, f_gt)) for in_dir_I in in_dir_I_list]

        # data augmentation and save
        for crop_and_scale_mode in ['h']:
            L_list_new, I_list_new = crop_and_scale(L_list, I_list, crop_and_scale_mode)

            f_new = '{}_{}_10x'.format(scene, crop_and_scale_mode)
            f_new_gt = '{}_{}_10x'.format(scene, crop_and_scale_mode)

            for d, img in zip(out_dir_L_list, L_list_new):
                save_img(img, d, f_new)
            for d, img in zip(out_dir_I_list, I_list_new):
                save_img(img, d, f_new_gt)
