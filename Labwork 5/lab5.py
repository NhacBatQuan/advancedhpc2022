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

height, width = image.shape[0], image.shape[1]
pixelCount = width * height
blockSize = (2, 2)
gridSize = (math.ceil(height/blockSize[0]), math.ceil(width/blockSize[1]))
reshaped_img = np.reshape(image, (height * width, 3))
devOutput = cuda.device_array((height,width,3), np.uint8)
devInput = cuda.to_device(image)

filter = np.array([[0,  0,   1,  2,  1,  0,  0],
       [0,  3,  13, 22, 13,  3,  0],
       [1, 13,  59, 97, 59, 13,  1],
       [2, 22,  97,159, 97, 22,  2],
       [1, 13,  59, 97, 59, 13,  1],
       [0,  3,  13, 22, 13,  3,  0],
       [0,  0,   1,  2,  1,  0,  0]
       ])

filter2 = filter/ 1003
devFilter = cuda.to_device(filter2)

blur = image.copy()
def conv(image,image2):
    for x in range(3, height - 3):
        for y in range(3, width - 3):
            r = 0
            g = 0
            b = 0
            for i in range(-3, 4):
                for j in range(-3,4):
                    r = (r + image[x+i,y+j,0] * filter2[i, j])
                    g = (g + image[x+i,y+j,1] * filter2[i, j])
                    b = (b + image[x+i,y+j,2] * filter2[i, j])
            image2[x, y, 0] = r
            image2[x, y, 1] = g
            image2[x, y, 2] = b

conv(image,blur)
plt.imsave('blur.png',blur)

@cuda.jit
def blur(src, dst, fil):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    r = 0
    g = 0
    b = 0
    for i in (range(-3,4)):
        for j in (range(-3,4)):
            r= np.uint8(r + (src[tidx+i, tidy+j, 0]*fil[i, j]))
            g= np.uint8(g + (src[tidx+i, tidy+j, 1]*fil[i, j]))
            b= np.uint8(b + (src[tidx+i, tidy+j, 2]*fil[i, j]))
  
    dst[tidx, tidy, 0 ] = r
    dst[tidx, tidy, 1 ] = g
    dst[tidx, tidy, 2 ] = b

blur[gridSize, blockSize](devInput, devOutput,devFilter)
start_nomemo = time.time()
hostOutput = devOutput.copy_to_host()
print("No memory shared time", abs(time.time()-start_nomemo))

plt.imsave('blur_gpu.png',hostOutput)

@cuda.jit
def blurMemo(src,dst,fil):
    memoFilter = cuda.shared.array((7, 7), np.uint8)
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    cuda.syncthreads()

    r = 0
    g = 0
    b = 0
    for i in (range(-3,4)):
        for j in (range(-3,4)):
            r= np.uint8(r + (src[tidx+i, tidy+j, 0]*fil[i, j]))
            g= np.uint8(g + (src[tidx+i, tidy+j, 1]*fil[i, j]))
            b= np.uint8(b + (src[tidx+i, tidy+j, 2]*fil[i, j]))
  
    dst[tidx, tidy, 0 ] = r
    dst[tidx, tidy, 1 ] = g
    dst[tidx, tidy, 2 ] = b

blurMemo[gridSize, blockSize](devInput, devOutput,devFilter)
start_memo = time.time()
hostOutput = devOutput.copy_to_host()
print("Memory shared time", abs(time.time()-start_memo))
plt.imsave('blur_gpu_with_memo.png',hostOutput)