import torch
import utils.pytorch as ptu

ptu.init_gpu(use_gpu=True, gpu_id=0)
ptu.init_dtype(set_dtype=torch.float32)

"""
state = {x, y, z, q0, q1, q2, q3, xdot, ydot, zdot, p, q, r}
         0  1  2  3   4   5   6   7     8     9     10 11 12
action = {x2dot, y2dot, z2dot, pdot, qdot, rdot}
          0      1      2      3     4     5
"""

def state_dot(state: torch.Tensor, action: torch.Tensor):
    
    # state:  [minibatch, nx]
    # action: [minibatch, nu]
    
    return torch.stack([
        state[:,7],
        state[:,8],
        state[:,9],
        -0.5 * state[:,10] * state[:,4] - 0.5 * state[:,11] * state[:,5] - 0.5 * state[:,6] * state[:,12],
        0.5 *  state[:,10] * state[:,3] - 0.5 * state[:,11] * state[:,6] + 0.5 * state[:,5] * state[:,12],
        0.5 *  state[:,10] * state[:,6] + 0.5 * state[:,11] * state[:,3] - 0.5 * state[:,4] * state[:,12],
        -0.5 * state[:,10] * state[:,5] + 0.5 * state[:,11] * state[:,4] + 0.5 * state[:,3] * state[:,12],
        action[:,0],
        action[:,1],
        action[:,2],
        action[:,3],
        action[:,4],
        action[:,5],
    ])


