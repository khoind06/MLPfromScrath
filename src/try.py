import cupy as cp

props = cp.cuda.runtime.getDeviceProperties(0)
print(props['name'])  # In tên GPU
print(props)  # In toàn bộ properties (compute capability, memory, v.v.)