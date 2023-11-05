import torch
import utils.pytorch as ptu

def get_params():

    init_state = ptu.tensor([[
        7000e3, 0, 0,    # position
        1, 0, 0, 0,      # quaternion
        0, 7.5e3, 0,     # velocity
        0, 0, 0         # angular rates
    ]])

    return locals()

if __name__ == "__main__":

    # example usage

    test = get_params()

    print('fin')