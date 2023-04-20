from utils import Utils, GPUTools

gpu_id = GPUTools.detect_available_gpu_id()
print(gpu_id)

print(GPUTools.get_available_gpu_ids())