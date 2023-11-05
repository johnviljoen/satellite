import torch
import utils.pytorch as ptu
from dynamics import state_dot
from params import get_params
from utils.orbital_elements import state_to_orbital_elements
import numpy as np
from utils.animator import Animator

# setup
# -----

ptu.init_gpu(use_gpu=True, gpu_id=0)
ptu.init_dtype(set_dtype=torch.float32)

params = get_params()
Ti, Ts, Tf = 0, 10.0, 10000
state = torch.clone(params["init_state"])
state[:,8] += 1000
state[:,9] += 100
action = ptu.tensor([[0.,0,0,0,0,0]])

# simulate
# --------

state_history = [ptu.to_numpy(state)]
times = torch.arange(Ti, Tf, Ts)
for t in times:
    state += state_dot.pytorch_batched(state, action) * Ts
    print(state_to_orbital_elements.pytorch_batched(state))
    state_history.append(ptu.to_numpy(state))
    print(t)

# animate 
# -------

state_history = np.vstack(state_history)
reference_history = np.copy(state_history)

animator = Animator(
    states=state_history[::10,:], 
    references=reference_history[::10,:], 
    times=ptu.to_numpy(times[::10]),
    sphere_rad=5000e3
)
animator.animate()

print('fin')