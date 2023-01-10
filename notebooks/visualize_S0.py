import os
import numpy as np
import cv2

for f in os.listdir(os.getcwd()):
    if f.endswith('.npy'):
        name = f.split('.')[0]
        image = np.load(f) / 2

        if len(image.shape) == 3:
            cv2.imwrite(name + '.png', cv2.cvtColor(image * 255, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(name + '.png', image * 255)