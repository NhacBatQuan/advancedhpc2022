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
blockSize = (20, 20)
gridSize = (math.ceil(height/blockSize[0]), math.ceil(width/blockSize[1]))
devOutput1 = cuda.device_array(np.shape(image),np.uint8)
devOutput2 = cuda.device_array(np.shape(image),np.uint8)
devInput1 = cuda.to_device(image)
image1 = np.zeros((height, width, 3),np.uint8)
image2 = np.zeros((height, width, 3),np.uint8)

def hsvcpu(image): 
    for i in range(height):
        for j in range(width):
            r = np.float64(image[i,j,0]/255)
            g = np.float64(image[i,j,1]/255)
            b = np.float64(image[i,j,2]/255)

            Max = max(r,g,b)
            Min = min(r,g,b)

            delta = Max - Min
            if delta == 0:
                h = 0
            elif Max == r:
                h = ((((g - b) / delta) % 6) * 60 )
            elif Max == g:
                h = ((((b - r) / delta) + 2) * 60 )
            elif Max == b: 
                h = ((((r - g) / delta) + 4) * 60 )
            if Max == 0:
                s = 0
            if Max != 0:
                s = delta/ Max
            v = Max

            image1[i,j,0] = h %360
            image1[i,j,1] = s*100
            image1[i,j,2] = v*100
    return image1

def cpukuwa(image,v,size): #WIP
    for wi in range(height+1):
        for wj in range(width+1):
            sizes = (
                ((wi-size,wi),(wj-size,wj)),
                ((wi,wi+size),(wj-size,wj)),
                ((wi-size,wi),(wj,wj+size)),
                ((wi,wi+size),(wj,wj+size))
    )    
            sum = 0
            sumsum = 0
            for i in range(*sizes[0][0]):
                for j in range(*sizes[0][1]):
                    sum = sum + (v[i,j])
                    sumsum = sumsum + (v[i,j] * v[i,j])
            stadev1 = math.sqrt(abs(sumsum / (size **2) - (sum / (size **2)) **2)) 

            sum = 0
            sumsum = 0
            for i in range(*sizes[1][0]):
                for j in range(*sizes[1][1]):
                    sum = sum + v[i,j]
                    sumsum = sumsum + v[i,j] * v[i,j]
            stadev2 = math.sqrt(abs(sumsum / (size **2) - (sum / (size **2)) **2)) 

            sum = 0
            sumsum = 0
            for i in range(*sizes[2][0]):
                for j in range(*sizes[2][1]):
                    sum = sum + v[i,j]
                    sumsum = sumsum + v[i,j] * v[i,j]
            stadev3 = math.sqrt(abs(sumsum / (size **2) - (sum / (size **2)) **2)) 

            sum = 0
            sumsum = 0
            for i in range(*sizes[3][0]):
                for j in range(*sizes[3][1]):
                    sum = sum + v[i,j]
                    sumsum = sumsum + v[i,j] * v[i,j]
            stadev4 = math.sqrt(abs(sumsum / (size **2) - (sum / (size **2)) **2)) 
            Min = min(stadev1,stadev2,stadev3,stadev4)
            stadev = (stadev1,stadev2,stadev3,stadev4)
            r = 0
            g = 0
            b = 0

            for si in (range(4)):
                for sj in range(4):
                    if Min == stadev[si]:
                        for wi in (sizes[sj][0]):
                            for wj in (sizes[sj][1]):
                                r = r + image[wi,wj,2]
                                g = g + image[wi,wj,1]
                                b = b + image[wi,wj,0]
    image2[wi,wj,2] = r / size**2
    image2[wi,wj,1] = g / size**2
    image2[wi,wj,0] = b / size**2  

# kuwahared = cpukuwa(image1,v_ed,8)
# plt.imsave("kuwahared.png",kuwahared)

@cuda.jit
def rgb_hsv(src, dst):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    r = src[tidx, tidy, 0]/255
    g = src[tidx, tidy, 1]/255
    b = src[tidx, tidy, 2]/255
    Max = max(r,g,b)
    Min = min(r,g,b)
    delta = Max - Min
    if delta == 0:
        h = 0
    elif Max == r:
        h = (60*(((g - b) / delta) % 6))
    elif Max == g:
        h = (60*(((b - r) / delta) + 2))
    elif Max == b: 
        h = (60*(((r - g) / delta) + 4))
    if Max == 0:
        s = 0
    if Max != 0:
        s = delta/ Max
    v = Max
    dst[tidx, tidy, 0] = h %360
    dst[tidx, tidy, 1] = s *100  
    dst[tidx, tidy, 2] = v *100

start_hsv = time.time()
rgb_hsv[gridSize, blockSize](devInput1, devOutput1)
hostOutput1 = devOutput1.copy_to_host()
print("GPU hsv time :",abs(time.time()-start_hsv))
plt.imsave("HSV_lab10.png",hostOutput1)
devInput2 = cuda.to_device(hostOutput1)

@cuda.jit
def kuwahara(src, v, dst, size):
    tidx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tidy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    sizes = (
        ((tidx-size,tidx),(tidy-size,tidy)),
        ((tidx,tidx+size),(tidy-size,tidy)),
        ((tidx-size,tidx),(tidy,tidy+size)),
        ((tidx,tidx+size),(tidy,tidy+size))
    )

    # for i in range(4):
    #     sum = 0
    #     sumsum = 0
    #     for e in range(*sizes[i][0]):
    #         for f in range(*sizes[i][1]):
    #             sum = sum + (v[e][f])
    #             sumsum = sumsum + (v[e][f]) **2
    #     stadev[i] = math.sqrt(sumsum / (size **2) - (sum / (size **2)) **2)
    # Min = min(stadev)

    sum = 0
    sum2 = 0
    for i in  range(*sizes[0][0]):
        for j in range(*sizes[1][0]):
                sum = sum + (v[i,j])
                sum2 = sum2 + (v[i,j] * v[i,j])
    stadev1 = math.sqrt(sum2 / (size **2) - (sum / (size **2)) **2)
    
    sum = 0
    sum2 = 0
    for i in range(*sizes[1][0]):
        for j in range(*sizes[1][1]):
                sum = sum + v[i,j]
                sum2 = sum2 + v[i,j] * v[i,j]
    stadev2 = math.sqrt(sum2 / (size **2) - (sum / (size **2)) **2) 

    sum = 0
    sum2 = 0
    for i in range(*sizes[2][0]):
        for j in range(*sizes[2][1]):
                sum = sum + v[i,j]
                sum2 = sum2 + v[i,j] * v[i,j]
    stadev3 = math.sqrt(sum2 / (size **2) - (sum / (size **2)) **2) 

    sum = 0
    sum2 = 0
    for i in range(*sizes[3][0]):
        for j in range(*sizes[3][1]):
                sum = sum + v[i,j]
                sum2 = sum2 + v[i,j] * v[i,j]
    stadev4 = math.sqrt(sum2 / (size **2) - (sum / (size **2)) **2) 

    Min = min(stadev1,stadev2,stadev3,stadev4)
    stadev = (stadev1,stadev2,stadev3,stadev4)

    r = 0
    g = 0
    b = 0

    for j in range(4):
        if Min == stadev[j]:
            temp = j
        for i in range(4):
            for wi in (sizes[temp][0]):
                for wj in (sizes[temp][1]):
                    r = r + src[wi,wj,2]
                    g = g + src[wi,wj,1]
                    b = b + src[wi,wj,0]

    dst[tidx,tidy,2] = r / size**2
    dst[tidx,tidy,1] = g / size**2
    dst[tidx,tidy,0] = b / size**2

v_hsv = hostOutput1[:,:,2]
v_hsv = np.ascontiguousarray(v_hsv)

start_kuwa = time.time()
kuwahara[gridSize, blockSize](devInput1, v_hsv,devOutput2,8)
hostOutput2 = devOutput2.copy_to_host()
print("GPU kuwa time :",abs(time.time()-start_hsv))
plt.imsave("kuwa.png",hostOutput2)




