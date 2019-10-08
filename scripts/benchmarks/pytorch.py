import torch
import time

def torch_timing(device_name):
    start_time = time.time()
    dtype = torch.double
    device = torch.device(device_name)

    norm = torch.randn(100, device=device, dtype=dtype)

    x = torch.randn(1000, 4, device=device, dtype=dtype)
    y = torch.randn(1000000, 4, device=device, dtype=dtype)

    mem_create_time = time.time()

    x = x.unsqueeze(1)
    y = y.unsqueeze(0)

    result = norm[(x-y).long()]
    
    compute_time = time.time()

    return start_time, mem_create_time, compute_time


cpu_times = torch_timing('cpu')
print('cpu')
for ii in range(len(cpu_times)-1):
    print(' (time', ii, '):', cpu_times[ii+1] - cpu_times[ii])
print(cpu_times[-1] - cpu_times[0])

torch.cuda.empty_cache()
gpu_times = torch_timing('cuda')
print('gpu')
for ii in range(len(gpu_times)-1):
    print(' (time', ii, '):', gpu_times[ii+1] - gpu_times[ii])
print(gpu_times[-1] - gpu_times[0])
