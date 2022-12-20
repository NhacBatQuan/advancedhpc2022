from numba import cuda
from numba.cuda.cudadrv import enums

print(cuda.detect())
print(cuda.gpus)

device = cuda.select_device(0)
num_processor = device.MULTIPROCESSOR_COUNT
compute_capability = device.compute_capability
print("compute capability :", compute_capability)
cores_per_sm = 128
cores = cores_per_sm * num_processor
memory =device.get_primary_context().get_memory_info()
print("number of processor :", num_processor)
print("clock rate :", device.CLOCK_RATE)
print("number of cores :", cores)
print("device memory :",memory)
