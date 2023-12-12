import sys
import os

assert len(sys.argv) == 3, 'Args are wrong.'

input_path = sys.argv[1]
output_path = sys.argv[2]

assert os.path.exists(input_path), 'Input model does not exist.'
assert not os.path.exists(output_path), 'Output filename already exists.'
assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

import torch
from control_net.share import *
from control_net.cldm.model import create_model


def get_node_name(name, parent_name):
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)]
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


model = create_model(config_path='./control_net/models/cldm_v15_slt.yaml')

# These weights should not be attempted to be copied over from the pre-trained model.
# NOTE: This is only for ControlNetSLT!
exclude = set([
   "control_model.input_blocks.1.1.proj_in.weight",
   "control_model.input_blocks.2.1.proj_in.weight",
   "control_model.input_blocks.4.1.proj_in.weight",
   "control_model.input_blocks.5.1.proj_in.weight",
   "control_model.input_blocks.7.1.proj_in.weight",
   "control_model.input_blocks.8.1.proj_in.weight",
   "control_model.middle_block.1.proj_in.weight",
])

pretrained_weights = torch.load(input_path)
if 'state_dict' in pretrained_weights:
    pretrained_weights = pretrained_weights['state_dict']

scratch_dict = model.state_dict()

target_dict = {}
for k in scratch_dict.keys():
    is_control, name = get_node_name(k, 'control_')
    if is_control:
        copy_k = 'model.diffusion_' + name
    else:
        copy_k = k
    if (copy_k in pretrained_weights) and (k not in exclude):
        target_dict[k] = pretrained_weights[copy_k].clone()
    else:            
        target_dict[k] = scratch_dict[k].clone()
        print(f'These weights are newly added: {k}')

model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), output_path)
print('Done.')
