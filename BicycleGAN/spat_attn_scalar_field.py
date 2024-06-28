import torch
import numpy as np
from torch_intermediate_layer_getter import IntermediateLayerGetter as MidGetter
from models.networks import get_norm_layer, D_NLayersMulti 

from models.networks import get_norm_layer, get_non_linearity, G_Unet_add_all, E_ResNet, D_NLayersMulti

from data.aligned_dataset import andres_forward

PATH = './checkpoints/CHECK_F4_256X256/40_net_G.pth'
PATHE = './checkpoints/CHECK_F4_256X256/40_net_E.pth'
PATHD = './checkpoints/CHECK_F4_256X256/40_net_D.pth'
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
num_Ds = 2

name = 'Run62_1009_1'
np_array = andres_forward(np.load(f'../../official_pix2pix_data_F6n1_GR_256X256/test/{name}.npy'), a=7)
real_A = torch.from_numpy(np_array[:,:256]).unsqueeze(0).unsqueeze(0)
real_B = torch.from_numpy(np_array[:,256:]).unsqueeze(0).unsqueeze(0)

vaeLike = True
netE = E_ResNet(output_nc, nz, nef, n_blocks=5, norm_layer=norm_layer, nl_layer=nl_layer, vaeLike=vaeLike)
net = G_Unet_add_all(input_nc, output_nc, nz, 8, ngf, norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
netD = D_NLayersMulti(input_nc=input_nc, ndf=ndf, n_layers=3, norm_layer=norm_layer, num_D=num_Ds)

z0, _ = netE(real_B)

net.load_state_dict(torch.load(PATH))
net.eval()
netE.load_state_dict(torch.load(PATHE))
netE.eval()
netD.load_state_dict(torch.load(PATHD))
netD.eval()

param_size = 0
for param in net.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in net.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('model size: {:.3f}MB'.format(size_all_mb))

import sys
sys.exit()
#from types import MethodType
#from functools import partial
#net.forward = MethodType(partial(net.forward, z=z0), net)

return_layers = {
    'model_0.5': 'model_0.5',
    'model_0.6': 'model_0.6',
    'model_0.7': 'model_0.7',
    'model_0.8': 'model_0.8',
    'model_1.5': 'model_1.5',
    'model_1.6': 'model_1.6',
    'model_1.7': 'model_1.7',
    'model_1.8': 'model_1.8'
}


#return_layers = {'model.up.3': 'model.up.3', 'model.up.1': 'model.up.1'}
#return_layers = {'model.up.2': 'model.up.2', 'model.up.1': 'model.up.1'}

mid_getter = MidGetter(netD, return_layers=return_layers, keep_output=True)


#batch = []
#batch.append(torch.from_numpy(andres_forward(np.load('../../official_pix2pix_data_F4n1_GR_256X256/train/Run1_1448_1.npy')[:,256:], a=7)))
#batch.append(torch.from_numpy(andres_forward(np.load('../../official_pix2pix_data_F4n1_GR_256X256/train/Run1_1448_3.npy')[:,256:], a=7)))
#batch = torch.vstack(batch)

#batch = batch.unsqueeze(0).unsqueeze(0)
fake_B = net(real_A, z0)

np.save(f'real_A_{name}.npy', real_A.cpu().detach().numpy())
np.save(f'real_B_{name}.npy', real_B.cpu().detach().numpy())
np.save(f'fake_B_{name}.npy', fake_B.cpu().detach().numpy())

fake_doutputs = netD(fake_B)
real_doutputs = netD(real_B)

fake_doutputs = [f.cpu().detach().numpy() for f in fake_doutputs]
real_doutputs = [r.cpu().detach().numpy() for r in real_doutputs]

np.save('fake_outputs_disc_model_0.npy', fake_doutputs[0])
np.save('fake_outputs_disc_model_1.npy', fake_doutputs[1])
np.save('real_outputs_disc_model_0.npy', real_doutputs[0])
np.save('real_outputs_disc_model_1.npy', real_doutputs[1])

mid_outputs = mid_getter(fake_B)

output = mid_outputs[0]['model_0.5']
print(output.shape)
np.save('disc_model_0.5_fake_B.npy', output.detach().numpy())

output = mid_outputs[0]['model_0.6']
print(output.shape)
np.save('disc_model_0.6_fake_B.npy', output.detach().numpy())

output = mid_outputs[0]['model_0.7']
print(output.shape)
np.save('disc_model_0.7_fake_B.npy', output.detach().numpy())

output = mid_outputs[0]['model_0.8']
print(output.shape)
np.save('disc_model_0.8_fake_B.npy', output.detach().numpy())

output = mid_outputs[0]['model_1.5']
print(output.shape)
np.save('disc_model_1.5_fake_B.npy', output.detach().numpy())

output = mid_outputs[0]['model_1.6']
print(output.shape)
np.save('disc_model_1.6_fake_B.npy', output.detach().numpy())

output = mid_outputs[0]['model_1.7']
print(output.shape)
np.save('disc_model_1.7_fake_B.npy', output.detach().numpy())

output = mid_outputs[0]['model_1.8']
print(output.shape)
np.save('disc_model_1.8_fake_B.npy', output.detach().numpy())


mid_outputs = mid_getter(real_B)

output = mid_outputs[0]['model_0.5']
print(output.shape)
np.save('disc_model_0.5_real_B.npy', output.detach().numpy())

output = mid_outputs[0]['model_0.6']
print(output.shape)
np.save('disc_model_0.6_real_B.npy', output.detach().numpy())

output = mid_outputs[0]['model_0.7']
print(output.shape)
np.save('disc_model_0.7_real_B.npy', output.detach().numpy())

output = mid_outputs[0]['model_0.8']
print(output.shape)
np.save('disc_model_0.8_real_B.npy', output.detach().numpy())

output = mid_outputs[0]['model_1.5']
print(output.shape)
np.save('disc_model_1.5_real_B.npy', output.detach().numpy())

output = mid_outputs[0]['model_1.6']
print(output.shape)
np.save('disc_model_1.6_real_B.npy', output.detach().numpy())

output = mid_outputs[0]['model_1.7']
print(output.shape)
np.save('disc_model_1.7_real_B.npy', output.detach().numpy())

output = mid_outputs[0]['model_1.8']
print(output.shape)
np.save('disc_model_1.8_real_B.npy', output.detach().numpy())

