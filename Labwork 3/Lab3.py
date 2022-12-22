from matplotlib.image import imread
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import numpy as np
import math
from numba import cuda

NUMBA_CUDA_DRIVER="/usr/lib/wsl/lib/libcuda.so.1"
LD_LIBRARY_PATH="/usr/lib/wsl/lib/"
cuda.detect()

image = plt.imread("/home/mocavandao/Lab/lechonk.jpg")

height, width = image.shape[0], image.shape[1]
pixelCount = width * height
blockSize = 64
gridSize = math.floor(pixelCount) / blockSize
reshaped_img = np.reshape(image, (height * width, 3))

def rgbToGray(img):
    r,g,b =image[:,:,0]*0.2989,image[:,:,1]*0.5870,image[:,:,2]*0.1140
    gray = r+g+b
    return gray

#CPU time
start =time.time()
grayed = rgbToGray(image)
print("CPU : ",abs(time.time()-start))

plt.imsave('test.png', image)
plt.imsave('grayed.png', grayed, cmap = 'gray')

devOutput = cuda.device_array((pixelCount), np.float64)
devInput = cuda.to_device(reshaped_img)
gridSize1 = math.floor(gridSize)

@cuda.jit
def grayscale(src, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    g = (src[tidx, 0] + src[tidx, 1] + src[tidx, 2]) / 3
    dst[tidx] = g

#GPU time
start = time.time()
grayscale[gridSize1, blockSize](devInput, devOutput)
print("GPU: ", abs(time.time()-start))

hostOutput = devOutput.copy_to_host()
hostOutput = np.reshape(hostOutput, (height, width))

plt.imsave('gpu.png',hostOutput, cmap = 'gray')
timer = []

#Block sizes
blockSizes = [16, 32, 64, 128, 256, 512, 1024]
for i in blockSizes:
    gridsize = pixelCount/i
    gridsize1 = math.floor(gridSize)
    start = time.time()
    grayscale[gridsize1, i](devInput, devOutput)
    stop = time.time()
    timer.append(abs(start-stop))
    print(timer)

fig, ax = plt.subplots()
ax.plot(blockSizes, timer)
ax.set_xlabel('Block size')
ax.set_ylabel('Compute time')
plt.savefig('Execution over block size.png')
plt.show()












# def rgbToGray(image):
#     img2, img1 = fig.add_subplot(121), fig.add_subplot(122)
#     img1.imshow(image)
#     img2.imshow(grayscale)

# fig.show()

# def rgbGray(image):
#     pixels = image.height * image.width
#     blockSize = 64
#     gridSize = pixels/blockSize
#     grayscale

# img = imread('/home/mocavandao/Lab/image.jpg')

# # print(img)
# print(img.shape)
# img2 = np.array(img).reshape(,1)
# print(len(img2))

# for i in range(len(img2)):
#     print(img2[i] /3)






