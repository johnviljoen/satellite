import torch

def euler_to_quaternion(euler_angles):
    # euler_angles is a tensor of shape (batch_size, 3)
    # where each row is (roll, pitch, yaw)
    rolls = euler_angles[:, 0]
    pitches = euler_angles[:, 1]
    yaws = euler_angles[:, 2]

    cys = torch.cos(yaws * 0.5)
    sys = torch.sin(yaws * 0.5)
    cps = torch.cos(pitches * 0.5)
    sps = torch.sin(pitches * 0.5)
    crs = torch.cos(rolls * 0.5)
    srs = torch.sin(rolls * 0.5)

    q0s = crs * cps * cys + srs * sps * sys
    q1s = srs * cps * cys - crs * sps * sys
    q2s = crs * sps * cys + srs * cps * sys
    q3s = crs * cps * sys - srs * sps * cys

    quaternions = torch.stack((q0s, q1s, q2s, q3s), dim=1)
    # Normalize each quaternion
    norms = torch.norm(quaternions, p=2, dim=1, keepdim=True)
    quaternions_normalized = quaternions / norms
    
    return quaternions_normalized

def quaternion_to_euler(quaternions):
    # quaternions is a tensor of shape (batch_size, 4)
    # where each row is [q0, q1, q2, q3] with q0 being the scalar part

    # Normalize the quaternion
    quaternions = quaternions / torch.norm(quaternions, dim=1, keepdim=True)
    
    # Extracting the values from each quaternion
    q0 = quaternions[:, 0]
    q1 = quaternions[:, 1]
    q2 = quaternions[:, 2]
    q3 = quaternions[:, 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (q0 * q1 + q2 * q3)
    cosr_cosp = 1 - 2 * (q1.pow(2) + q2.pow(2))
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (q0 * q2 - q3 * q1)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * torch.pi / 2,
        torch.asin(sinp)
    )

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (q0 * q3 + q1 * q2)
    cosy_cosp = 1 - 2 * (q2.pow(2) + q3.pow(2))
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.vstack([roll, pitch, yaw]).T

if __name__ == "__main__":
    
    import pytorch_utils as ptu

    # Example usage:
    # Define a batch of Euler angles (roll, pitch, yaw) in radians
    euler_angles_batch = ptu.tensor([
        [30, 45, 60],
        [10, 20, 30],
        # ... more angles
    ]) * torch.pi/180  # Convert degrees to radians

    # Convert to quaternions
    quaternions_batch = euler_to_quaternion_batch_torch(euler_angles_batch)

    # Print the quaternions
    print("Quaternions:\n", quaternions_batch)

    # Convert to Euler angles
    eul_batch = quaternion_to_euler_batch_torch(quaternions_batch)

    # Print the Euler angles in radians
    print(f"eul_batch: {eul_batch * 180/torch.pi}")