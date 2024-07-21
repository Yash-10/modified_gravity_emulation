import os
import numpy as np
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import torch.utils.benchmark as benchmark

from models.networks import get_norm_layer, get_non_linearity, G_Unet_add_all, E_ResNet

from data.aligned_dataset import andres_forward

PATH = './checkpoints/CHECK_F4_256X256/40_net_G.pth'
PATHE = './checkpoints/CHECK_F4_256X256/40_net_E.pth'
input_nc = 1
output_nc = 1
nz = 128
ngf = 128
ndf = 128
nef = 128
norm_layer = get_norm_layer(norm_type='instance')
nl_layer = get_non_linearity(layer_type='relu')
use_dropout = False
upsample = 'basic'


device = 'cpu'
def get_z_random(batch_size, nz, random_type='gauss'):
    if random_type == 'uni':
        z = torch.rand(batch_size, nz) * 2.0 - 1.0
    elif random_type == 'gauss':
        z = torch.randn(batch_size, nz)
    return z.detach().to(device)

"""
Some examples to try:
[dc-gond1@login5a test]$ ls -U | head -10
Run62_881_4.npy
Run62_740_2.npy
Run62_1458_2.npy
Run62_243_2.npy
Run62_1461_4.npy
Run62_358_3.npy
Run62_845_2.npy
Run62_784_4.npy
Run62_1356_1.npy
Run62_1162_4.npy
```
"""
def combine_256_maps_into_512(name, gr=True):
    prefix = '../../official_pix2pix_data_F4n1_GR_256X256/test/'
    map1 = np.load(prefix+name+'_1.npy')
    map2 = np.load(prefix+name+'_2.npy')
    map3 = np.load(prefix+name+'_3.npy')
    map4 = np.load(prefix+name+'_4.npy')

    if gr:
        map1 = map1[:,:256]
        map2 = map2[:,:256]
        map3 = map3[:,:256]
        map4 = map4[:,:256]
    else:
        map1 = map1[:,256:]
        map2 = map2[:,256:]
        map3 = map3[:,256:]
        map4 = map4[:,256:]

    arr = np.vstack(
            (
                np.hstack((map1, map2)),
                np.hstack((map3, map4))
            )
    )
    assert arr.shape == (512, 512)
    return arr

#name = 'Run62_881_4.npy'
name = 'Run62_1162'
#np_array = andres_forward(np.load(f'../../official_pix2pix_data_F4n1_GR_256X256/test/{name}'), a=7)
np_array = np.hstack(
        (combine_256_maps_into_512(name, gr=True), combine_256_maps_into_512(name, gr=False)
))
real_A = np_array[:,:512]
real_B = np_array[:,512:]
real_A = torch.from_numpy(real_A).unsqueeze(0).unsqueeze(0)
real_B = torch.from_numpy(real_B).unsqueeze(0).unsqueeze(0)


bs = 16
real_A = real_A.repeat(bs, 1,1,1)
real_B = real_B.repeat(bs, 1,1,1)

encode = False
vaeLike = True
netE = E_ResNet(output_nc, nz, nef, n_blocks=5, norm_layer=norm_layer, nl_layer=nl_layer, vaeLike=vaeLike)

net = G_Unet_add_all(input_nc, output_nc, nz, 8, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                                             use_dropout=use_dropout, upsample=upsample)

net.load_state_dict(torch.load(PATH))
net.eval()
netE.load_state_dict(torch.load(PATHE))
netE.eval()

# For GPU exec time measurement, see https://discuss.pytorch.org/t/time-consuming-of-gpu-model/22285
use_gpu = False
device = torch.device('cuda:0') if use_gpu else torch.device('cpu')  # get device name: CPU or GPU

#from timeit import default_timer as timer
#start = timer()
#net(real_A, z0)
#end = timer()
#print(end-start)
#activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA] if use_gpu else [ProfilerActivity.CPU]

import time
#torch.cuda.synchronize()  # Uncomment if using GPU.
start_time = time.time()

#with profile(activities=activities, record_shapes=True) as prof:
#with record_function("model_inference"):
for _ in range(512//bs):
    if encode:
        #netE.to(device)  # Uncomment if using GPU.
        #real_B = real_B.to(device)  # Uncomment if using GPU.
        z0, _ = netE(real_B)
    else:
        #real_A = real_A.to(device)  # Uncomment if using GPU.
        z0 = get_z_random(real_A.size(0), nz)
    #net.to(device)  # Uncomment if using GPU.
    net(real_A, z0)

#torch.cuda.synchronize()  # Uncomment if using GPU.
end_time = time.time()
wall_clock_time = end_time - start_time
print(wall_clock_time)
#print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=-1))

#profiling_results = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1)
#total_time = sum(row.self_cpu_time_total for row in prof.key_averages())
#print("Total execution time of forward pass:", total_time/1e6, "seconds")

# Calculate total memory usage in bytes
#total_memory = sum(row.self_cpu_memory_usage for row in prof.key_averages())
#print([row.self_cpu_memory_usage for row in prof.key_averages()])
#print("Total memory usage:", total_memory, 'bytes')
