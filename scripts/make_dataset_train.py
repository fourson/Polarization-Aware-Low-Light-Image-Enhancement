import os
import random

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


def rotate(L_list, I_list, S_hat_list, S_list, mode):
    if mode == 'td':
        func = lambda x: np.flip(x, axis=0)
    elif mode == 'lr':
        func = lambda x: np.flip(x, axis=1)
    elif mode == 'a90':
        func = lambda x: np.rot90(x, 1)
    elif mode == 'a180':
        func = lambda x: np.rot90(x, 2)
    elif mode == 'a270':
        func = lambda x: np.rot90(x, 3)
    else:
        func = lambda x: x

    L_list_new = [func(L) for L in L_list]
    I_list_new = [func(I) for I in I_list]
    S_hat_list_new = [func(S_hat) for S_hat in S_hat_list]
    S_list_new = [func(S) for S in S_list]
    return L_list_new, I_list_new, S_hat_list_new, S_list_new


def crop_and_scale(L_list, I_list, S_hat_list, S_list, mode):
    # input size: 1024*1024
    if mode == 'crop':
        H_offset = random.randrange(0, 768)
        W_offset = random.randrange(0, 768)
        func = lambda x: x[H_offset: H_offset + 256, W_offset: W_offset + 256, :]
    elif mode == 'htl':
        func = lambda x: x[0::2, 0::2, :][0:256, 0:256, :]
    elif mode == 'htr':
        func = lambda x: x[0::2, 0::2, :][0:256, 256:512, :]
    elif mode == 'hbl':
        func = lambda x: x[0::2, 0::2, :][256:512, 0:256, :]
    elif mode == 'hbr':
        func = lambda x: x[0::2, 0::2, :][256:512, 256:512, :]
    elif mode == 'hmid':
        func = lambda x: x[0::2, 0::2, :][128:384, 128:384, :]
    else:
        func = lambda x: x

    L_list_new = [func(L) for L in L_list]
    I_list_new = [func(I) for I in I_list]
    S_hat_list_new = [func(S_hat) for S_hat in S_hat_list]
    S_list_new = [func(S) for S in S_list]
    return L_list_new, I_list_new, S_hat_list_new, S_list_new


if __name__ == '__main__':
    in_dir = '../raw_images/data_train_temp'
    out_dir = '../data/train'
    save_stokes = False

    out_dir_L_list = [ensure_dir(os.path.join(out_dir, 'L{}'.format(i))) for i in range(1, 5)]
    out_dir_I_list = [ensure_dir(os.path.join(out_dir, 'I{}'.format(i))) for i in range(1, 5)]
    if save_stokes:
        out_dir_S_hat_list = [ensure_dir(os.path.join(out_dir, 'S{}_hat'.format(i))) for i in range(0, 3)]
        out_dir_S_list = [ensure_dir(os.path.join(out_dir, 'S{}'.format(i))) for i in range(0, 3)]
    else:
        out_dir_S_hat_list = [os.path.join(out_dir, 'S{}_hat'.format(i)) for i in range(0, 3)]
        out_dir_S_list = [os.path.join(out_dir, 'S{}'.format(i)) for i in range(0, 3)]

    in_dir_L_list = [os.path.join(in_dir, 'L{}'.format(i)) for i in range(1, 5)]
    in_dir_I_list = [os.path.join(in_dir, 'I{}'.format(i)) for i in range(1, 5)]
    for f in os.listdir(in_dir_L_list[0]):
        print(f)
        scene, time_postfix = f.split('_')
        time = int(time_postfix[:-4])

        time_gt = time * 10
        f_gt = '{}_{}.png'.format(scene, time_gt)

        L_list = [read_img(os.path.join(in_dir_L, f)) for in_dir_L in in_dir_L_list]
        S_hat_list = compute_Si_from_Ii(L_list)
        I_list = [read_img(os.path.join(in_dir_I, f_gt)) for in_dir_I in in_dir_I_list]
        S_list = compute_Si_from_Ii(I_list)

        # data augmentation and save
        for idx, crop_and_scale_mode in enumerate(5 * ['crop'] + ['htl', 'htr', 'hbl', 'hbr', 'hmid']):
            for rotate_mode in ['td', 'lr', 'a90', 'a180', 'a270', 'a0']:
                L_list_new, I_list_new, S_hat_list_new, S_list_new = crop_and_scale(L_list, I_list, S_hat_list, S_list,
                                                                                    crop_and_scale_mode)
                L_list_new, I_list_new, S_hat_list_new, S_list_new = rotate(L_list_new, I_list_new, S_hat_list_new,
                                                                            S_list_new, rotate_mode)

                if crop_and_scale_mode == 'crop':
                    crop_and_scale_mode_ = 'crop{}'.format(idx)
                else:
                    crop_and_scale_mode_ = crop_and_scale_mode

                f_new = '{}_{}_{}_{}'.format(scene, crop_and_scale_mode_, rotate_mode, time)
                f_new_gt = '{}_{}_{}_{}'.format(scene, crop_and_scale_mode_, rotate_mode, time_gt)

                for d, img in zip(out_dir_L_list, L_list_new):
                    save_img(img, d, f_new)
                for d, img in zip(out_dir_I_list, I_list_new):
                    save_img(img, d, f_new_gt)
                if save_stokes:
                    for d, img in zip(out_dir_S_hat_list, S_hat_list_new):
                        save_npy(img, d, f_new)
                    for d, img in zip(out_dir_S_list, S_list_new):
                        save_npy(img, d, f_new_gt)
