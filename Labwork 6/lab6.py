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
devOutput = cuda.device_array((height,width,3), np.uint8)
devOutput1 = cuda.device_array((height,width,3), np.uint8)
devInput = cuda.to_device(image)

@cuda.jit
def biGray(src,dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    f = (np.uint8((src[tidx,tidy,0] + src[tidx,tidy,1] + src[tidx,tidy,2])/3))

    if f> 255/2:
        f = 255
    else:
        f = 0
    dst[tidx,tidy] = f

start_bigray = time.time()
biGray[gridSize, blockSize](devInput, devOutput)
print("GPU binary gray time :",abs(time.time()-start_bigray))
hostOutput = devOutput.copy_to_host()

plt.imsave("Binarization gray.png",hostOutput, cmap = 'gray')

@cuda.jit
def bright(src,dst, brightness):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    fr = np.uint8(src[tidx,tidy,0] * brightness)
    fg = np.uint8(src[tidx,tidy,1] * brightness)
    fb = np.uint8(src[tidx,tidy,2] * brightness)

    dst[tidx,tidy,0] = fr
    dst[tidx,tidy,1] = fg
    dst[tidx,tidy,2] = fb

bright[gridSize, blockSize](devInput, devOutput1, 1.1)
hostOutput1 = devOutput1.copy_to_host()
plt.imsave("brightness.png",hostOutput1)