import os
import numpy as np
import cv2

rgb_split = False

for f in os.listdir(os.getcwd()):
    if f.endswith('.npy'):
        name = f.split('.')[0]
        dop = np.load(f)
        if len(dop.shape) == 3:
            if rgb_split:
                dop_color_R = cv2.applyColorMap(np.uint8(dop[:, :, 0] * 255), cv2.COLORMAP_JET)
                dop_color_G = cv2.applyColorMap(np.uint8(dop[:, :, 1] * 255), cv2.COLORMAP_JET)
                dop_color_B = cv2.applyColorMap(np.uint8(dop[:, :, 2] * 255), cv2.COLORMAP_JET)
                cv2.imwrite(name + '_R.png', dop_color_R)
                cv2.imwrite(name + '_G.png', dop_color_G)
                cv2.imwrite(name + '_B.png', dop_color_B)
            else:
                dop_color_gray = cv2.applyColorMap(cv2.cvtColor(np.uint8(dop * 255), cv2.COLOR_RGB2GRAY), cv2.COLORMAP_JET)
                cv2.imwrite(name + '.png', dop_color_gray)
        else:
            dop_color_gray = cv2.applyColorMap(np.uint8(dop * 255), cv2.COLORMAP_JET)
            cv2.imwrite(name + '.png', dop_color_gray)
