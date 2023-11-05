import torch
import utils.pytorch as ptu

def state_to_orbital_elements(state, mu=398600.4418):
    
    r_vec = state[:,:3]
    v_vec = state[:,7:10]

    # Calculate specific angular momentum
    h_vec = torch.cross(r_vec, v_vec, dim=1)
    h = torch.norm(h_vec, dim=1)
    
    # Avoid division by zero
    h[h == 0] = 1e-10

    # Calculate the eccentricity vector
    r = torch.norm(r_vec, dim=1, keepdim=True)
    v = torch.norm(v_vec, dim=1, keepdim=True)
    term1 = torch.cross(v_vec, h_vec, dim=1) / mu
    term2 = r_vec / r
    e_vec = term1 - term2
    e = torch.norm(e_vec, dim=1)

    # Calculate the semi-major axis
    a = 1 / (2 / r - v.pow(2) / mu)
    
    # Calculate inclination
    i = torch.acos(h_vec[:, 2] / h)
    
    # Calculate the node line
    n_vec = torch.cross(ptu.tensor([0, 0, 1]).expand_as(r_vec), h_vec, dim=1)
    n = torch.norm(n_vec, dim=1)

    # Right ascension of the ascending node (RAAN)
    Omega = torch.zeros_like(n)
    mask = n != 0
    Omega[mask] = torch.acos(n_vec[mask, 0] / n[mask])
    Omega[n_vec[:, 1] < 0] = 2 * torch.pi - Omega[n_vec[:, 1] < 0]
    
    # Argument of periapsis
    omega = torch.zeros_like(e)
    mask = (n != 0) & (e != 0)
    omega[mask] = torch.acos(torch.einsum('bi,bi->b', n_vec[mask], e_vec[mask]) / (n[mask] * e[mask]))
    omega[e_vec[:, 2] < 0] = 2 * torch.pi - omega[e_vec[:, 2] < 0]
    
    # True anomaly
    nu = torch.zeros_like(e)
    mask = e != 0
    nu[mask] = torch.acos(torch.einsum('bi,bi->b', e_vec[mask], r_vec[mask]) / (e[mask] * r[mask].squeeze()))
    radial_velocity = torch.einsum('bi,bi->b', r_vec, v_vec)
    nu[radial_velocity < 0] = 2 * torch.pi - nu[radial_velocity < 0]
    
    # Handling edge cases to prevent NaNs
    Omega = torch.nan_to_num(Omega, nan=0.0)
    omega = torch.nan_to_num(omega, nan=0.0)
    nu = torch.nan_to_num(nu, nan=0.0)
    a = torch.nan_to_num(a, nan=0.0)
    e = torch.nan_to_num(e, nan=0.0)
    i = torch.nan_to_num(i, nan=0.0)

    return {
        'semi_major_axis': a,
        'eccentricity': e,
        'inclination': i,
        'ra_of_ascending_node': Omega,
        'argument_of_periapsis': omega,
        'true_anomaly': nu
    }

if __name__ == "__main__":

    import utils.pytorch as ptu
    
    # Example usage
    r_vec = ptu.tensor([[7000e3, 0, 0], [8000e3, 0, 0]])
    v_vec = ptu.tensor([[0, 7.5e3, 0], [0, 7.8e3, 0]])

    elements = state_to_orbital_elements(r_vec, v_vec)
    print(elements)


