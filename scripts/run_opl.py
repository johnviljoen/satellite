import torch
import utils.pytorch as ptu
from dynamics import state_dot
from params import get_params

ptu.init_gpu(use_gpu=True, gpu_id=0)
ptu.init_dtype(set_dtype=torch.float32)

params = get_params()
Ti, Ts, Tf = 0, 1.0, 100
state = torch.clone(params["init_state"])
action = ptu.tensor([[1.,0,0,0,0,0]])

for t in torch.arange(Ti, Tf, Ts):
    state += state_dot(state, action) * Ts
    
