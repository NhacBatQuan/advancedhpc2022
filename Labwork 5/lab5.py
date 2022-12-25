from matplotlib.image import imread
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import numpy as np
import math
from numba import cuda

NUMBA_CUDA_DRIVER="/usr/lib/wsl/lib/libcuda.so.1"
LD_LIBRARY_PATH="/usr/lib/wsl/lib/"
image = plt.imread("/home/mocavandao/Lab/lechonk.jpg")

filter = np.array([[0,  0,   1,  2,  1,  0,  0],
       [0,  3,  13, 22, 13,  3,  0],
       [1, 13,  59, 97, 59, 13,  1],
       [2, 22,  97,159, 97, 22,  2],
       [1, 13,  59, 97, 59, 13,  1],
       [0,  3,  13, 22, 13,  3,  0],
       [0,  0,   1,  2,  1,  0,  0]
       ])

filter = filter/ np.sum(filter)

blur = image.copy()
def conv(image,image2):
    for x in range(3, image.shape[0]-3):
        for y in range(3, image.shape[1]-3):
            r = 0
            g = 0
            b = 0
            for i in range(-3, 3):
                for j in range(-3,3):
                    r = (r + image[x+i,y+j,0] * filter[i, j])
                    g = (g + image[x+i,y+j,1] * filter[i, j])
                    b = (b + image[x+i,y+j,2] * filter[i, j])
            image2[x,y,0] = r
            image2[x,y,1] = g
            image2[x,y,2] = b

conv(image,blur)
plt.imsave('blur.png',blur)
