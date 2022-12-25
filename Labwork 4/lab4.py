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
timer_2d = []
timer_1d = []
height, width = image.shape[0], image.shape[1]
pixelCount = width * height
blockSize = (32, 32)
gridSize = (math.ceil(height/blockSize[0]), math.ceil(width/blockSize[1]))
reshaped_img = np.reshape(image, (height * width, 3))

#CPU grayscale
def rgbToGray(img):
    r,g,b =image[:,:,0]*0.2989,image[:,:,1]*0.5870,image[:,:,2]*0.1140
    gray = r+g+b
    return gray

#CPU time
start =time.time()
grayed = rgbToGray(image)
cputime = time.time() - start
print("GPU compute time: ", abs(cputime))
plt.imsave('RGB.png', image)
plt.imsave('grayed.png', grayed, cmap = 'gray')

devOutput = cuda.device_array((height,width,3), np.uint8)
devOutput1 = cuda.device_array((height,width,3), np.uint8)
devInput = cuda.to_device(image)
devInput1 = cuda.to_device(reshaped_img)

# 1 Dim GPU grayscale
@cuda.jit
def grayscale_1d(src, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    g = np.uint8((src[tidx, 0] + src[tidx, 1] + src[tidx, 2]) / 3)
    dst[tidx, 0] = dst[tidx, 1] = dst[tidx, 2] = g

# 2 Dim GPU grayscale
@cuda.jit
def grayscale_2d(src, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    g = np.uint8((src[tidx,tidy, 0] + src[tidx,tidy, 1] + src[tidx,tidy, 2]) / 3)
    dst[tidx,tidy,0] = dst[tidx,tidy,1] = dst[tidx,tidy,2] = g

#GPU 2 dim image
grayscale_2d[gridSize, blockSize](devInput, devOutput)
hostOutput = devOutput.copy_to_host()
hostOutput = np.reshape(hostOutput, (height, width, 3))
plt.imsave('gpu_2d.png',hostOutput, cmap = 'gray')

#GPU 1 dim image
grayscale_1d[gridSize, blockSize](devInput1, devOutput1)
hostOutput1 = devOutput1.copy_to_host()
hostOutput1 = np.reshape(hostOutput1, (height, width, 3))
plt.imsave('gpu_1d.png',hostOutput1, cmap = 'gray')

#block sizes
blockSizes = [(2,2),(4,4),(8,8),(16,16),(32,32)]
for i in blockSizes:
    gridsize = (32,32)
    start = time.time()
    grayscale_2d[gridsize, i](devInput, devOutput)
    stop = time.time()
    timer_2d.append(abs(start-stop))
    start1 = time.time()
    grayscale_1d[gridsize, i](devInput1, devOutput1)
    stop1 = time.time()
    timer_1d.append(abs(start1-stop1))

print(timer_2d)
gputime_2d = timer_2d[2]
gputime_1d = timer_1d[2]
print("GPU compute time 2d: ", gputime_2d)
print("GPU compute time 1d: ", gputime_1d)
fig, ax = plt.subplots()
ax.plot(blockSizes, timer_2d)
ax.set_xlabel('Block size')
ax.set_ylabel('Compute time')
plt.savefig('Execution over block size.png')
plt.show()





