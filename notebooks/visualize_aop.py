import os
import numpy as np
import cv2

rgb_split = False

for f in os.listdir(os.getcwd()):
    if f.endswith('.npy'):
        name = f.split('.')[0]
        aop = np.load(f) / np.pi
        if len(aop.shape) == 3:
            if rgb_split:
                aop_color_R = cv2.applyColorMap(np.uint8(aop[:, :, 0] * 255), cv2.COLORMAP_JET)
                aop_color_G = cv2.applyColorMap(np.uint8(aop[:, :, 1] * 255), cv2.COLORMAP_JET)
                aop_color_B = cv2.applyColorMap(np.uint8(aop[:, :, 2] * 255), cv2.COLORMAP_JET)
                cv2.imwrite(name + '_R.png', aop_color_R)
                cv2.imwrite(name + '_G.png', aop_color_G)
                cv2.imwrite(name + '_B.png', aop_color_B)
            else:
                aop_color_gray = cv2.applyColorMap(cv2.cvtColor(np.uint8(aop * 255), cv2.COLOR_RGB2GRAY), cv2.COLORMAP_JET)
                cv2.imwrite(name + '.png', aop_color_gray)
        else:
            aop_color_gray = cv2.applyColorMap(np.uint8(aop * 255), cv2.COLORMAP_JET)
            cv2.imwrite(name + '.png', aop_color_gray)