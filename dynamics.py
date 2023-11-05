import torch
import utils.pytorch as ptu
from utils.rotation import quaternion_derivative

def state_dot(state: torch.Tensor, action: torch.Tensor, G=6.67430e-11, M=5.972e+24):
    """
    Compute the time derivative of the state.

    Parameters:
    - state: Tensor containing the state.               [minibatch, nx]
            {x, y, z, q0, q1, q2, q3, xdot, ydot, zdot, p, q, r}
             0  1  2  3   4   5   6   7     8     9     10 11 12
    - action: Tensor containing the control inputs.     [minibatch, nu]
            {x2dot, y2dot, z2dot, pdot, qdot, rdot}
             0      1      2      3     4     5
    - G: Gravitational constant.
    - M: Mass of the Earth.
    
    Returns:
    - Tensor containing the time derivative of the state.
    """

        # Extract positions and velocities
    pos = state[:, :3]  # [minibatch, 3]
    vel = state[:, 7:10]  # [minibatch, 3]

    # Compute the square of the distances
    r_squared = torch.sum(pos * pos, dim=1, keepdim=True)  # [minibatch, 1]

    # Compute the gravitational acceleration
    r = torch.sqrt(r_squared)  # [minibatch, 1]
    acc_gravity = -G * M / r_squared * pos / r  # [minibatch, 3]

    # Quaternion derivatives need to be calculated with respect to angular rates
    # This quaternion derivative will assume q = [q0, q1, q2, q3] where q0 is the scalar part
    q = state[:, 3:7]  # Extract quaternions from the state
    omega = state[:, 10:13]  # Extract angular velocities (p, q, r)

    q_dot = quaternion_derivative(q, omega)

    # Compute the time derivative of the state
    state_dot = torch.hstack([
        vel,
        q_dot,
        acc_gravity + action[:, :3],
        action[:, 3:]
    ])

    return state_dot
    
# return torch.hstack([
#     state[:,7:8],
#     state[:,8:9],
#     state[:,9:10],
#     -0.5 * state[:,10:11] * state[:,4:5] - 0.5 * state[:,11:12] * state[:,5:6] - 0.5 * state[:,6:7] * state[:,12:13],
#     0.5 *  state[:,10:11] * state[:,3:4] - 0.5 * state[:,11:12] * state[:,6:7] + 0.5 * state[:,5:6] * state[:,12:13],
#     0.5 *  state[:,10:11] * state[:,6:7] + 0.5 * state[:,11:12] * state[:,3:4] - 0.5 * state[:,4:5] * state[:,12:13],
#     -0.5 * state[:,10:11] * state[:,5:6] + 0.5 * state[:,11:12] * state[:,4:5] + 0.5 * state[:,3:4] * state[:,12:13],
#     action[:,0:1],
#     action[:,1:2],
#     action[:,2:3],
#     action[:,3:4],
#     action[:,4:5],
#     action[:,5:6],
# ])


if __name__ == "__main__":

    # example usage

    ptu.init_gpu(use_gpu=True, gpu_id=0)
    ptu.init_dtype(set_dtype=torch.float32)