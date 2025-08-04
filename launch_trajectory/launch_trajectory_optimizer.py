import numpy as np
import torch
import torch.nn as nn
from constants import R as earth_radius, earth_rotation_rate, burn_time, launch_lat, launch_lon, rocket_mass, ORBITAL_ELEMENTS,a, i, raan, arg_of_perigee, e

def convert_to_eci(lat, lon, t=0):
    """
    Convert launch site to ECI coordinates at time t. t-> time since epoch
    t in seconds from reference epoch (e.g., t=0 = launch)
    """

    # Earth radius is in metres
    # Enter latitude and longitude in radians

    theta = lon + earth_rotation_rate * t
    x = earth_radius * np.cos(lat) * np.cos(theta)
    y = earth_radius * np.cos(lat) * np.sin(theta)
    z = earth_radius * np.sin(lat)

    return np.array([x, y, z], dtype=np.float64)

def rotation_matrix(i, raan, arg_of_perigee):
    # 3 - 1 - 3 Euler rotation matrix -> sequence of rotations about the z-axis, then the x-axis, then again the z-axis.

    R3_W =np.array(
        [
            [np.cos(-raan), -np.sin(-raan), 0],
            [np.sin(-raan), np.cos(-raan), 0],
            [0, 0, 1]
        ]
    )

    R1_i = np.array(
        [
            [1, 0, 0],
            [0, np.cos(-i), -np.sin(-i)],
            [0, np.cos(-i), np.sin(-i)]
        ]
    )

    R3_w = np.array(
        [
            [np.cos(-arg_of_perigee), -np.sin(-arg_of_perigee), 0],
            [np.sin(-arg_of_perigee), np.cos(-arg_of_perigee), 0],
            [0, 0, 1]
        ]
    )

    return R3_W @ R1_i @ R3_w

def position_on_orbit(a, e, true_anomaly):
    # to return -> position in perifocal coordinates

    """
    What is Perifocal coordinates?

    Perifocal coordinates are a special 3D coordinate system used to describe a spacecraft’s position and motion in its orbital plane.

        - The origin is at the center of the planet or star.
        - The x-axis points to the closest point in the orbit (called periapsis).
        - The y-axis points in the direction the spacecraft moves at periapsis.
        - The z-axis points up from the orbital plane (perpendicular to it).

    This system makes orbit math easier, especially for elliptical orbits.
    """

    r = a * (1 + np.power(np.e, 2)) / (1 + np.e * np.cos(true_anomaly))

    return np.array([
        r * np.cos(true_anomaly),
        r * np.sin(true_anomaly),
        0
    ])

def compute_optimal_true_anomaly(orbital_elements, launch_lat, launch_lon, t_launch):
    # assuming latitude and longitude are in radians
    a = orbital_elements[0]
    e = orbital_elements[1]
    i = orbital_elements[2]
    raan = orbital_elements[3]
    arg_of_perigee = orbital_elements[4]

    # ECI conversion of launch coordinates
    launch_eci = convert_to_eci(launch_lat, launch_lon, t_launch)
    launch_dir = launch_eci / np.linalg.norm(launch_eci)

    # converting perifocal to ECI using rotation matrix
    Q_p_to_i = rotation_matrix(i, raan, arg_of_perigee)

    # Trialing true anomalies
    best_angle = np.inf
    best_nu = 0  # Trialing and validating true anomalies

    for nu in np.linspace(0, 2*np.pi, 50): # Trialling 1000 values, add more for fine tuning
        r_perifocal = position_on_orbit(a, e, nu)
        r_eci = Q_p_to_i @ r_perifocal
        r_dir = r_eci / np.linalg.norm(r_eci)

        angle = np.arccos(np.clip(np.dot(r_dir, launch_dir), -1, 1))
        if angle < best_angle:
            best_angle = angle
            best_nu = nu

    return best_nu


# Trial code -> remove at last
"""
if __name__ == "__main__":
    from constants import ORBITAL_ELEMENTS, launch_lon, launch_lat
    from datetime import datetime
    from astropy.time import Time

    # Get current time (or time of launch)
    t = datetime.now()

    # Convert to seconds since J2000 using astropy
    j2000_epoch = datetime(2000, 1, 1, 12, 0, 0)  # J2000 epoch
    time_difference = t - j2000_epoch  # Calculate time difference
    t_launch_seconds = time_difference.total_seconds()  # Convert to seconds

    # Now pass the numeric value (t_launch_seconds) to your function
    nu = compute_optimal_true_anomaly(ORBITAL_ELEMENTS, launch_lat, launch_lon, t_launch_seconds)

    print("Optimal true anomaly:", nu)
"""


# PINN design
"""
Input: time 
Outputs: position r(x, y, z), velocity v(x, y, z), mass m(t)

Constraints:
    - Satisfy Equations of Motion
    Initial conditions:
        - r(x, y, z) -> launch site ECI
        - v(x, y, z) -> 0 or launch pad direction
        - m(t) -> initial mass of rocket
        
    Terminal conditions:
    - r(x, y, z) -> r(target)
    - v(x, y, z) -> orbital velocity for altitude
    - m(t) -> initial mass of rocket
"""

# Loss Function
"""
# Physics loss {EOM Loss} -> DE's to be solved
1. dr/dt = v
2. dv/dt = a{gravity} + a{thrust}
3. dm/dt = -T(t)/(Isp*g0) -> Isp: Specific impulse

L{total} =  w{EOM} * L{EOM} + 
            w{initial condition} * L{initial condition} + 
            w{target} * L{target} + 
            w{fuel} * L{fuel} # Mass depletion rate loss
"""

# Defining the NN
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.net = nn.Sequential()

        # Building the layers
        for i in range(len(layers)-1):
            self.net.add_module(f"linear{i}", nn.Linear(layers[i], layers[i+1]))
            if i < len(layers)-2 and layers[i+1]:
                self.net.add_module(f"tanh{i}", nn.Tanh())

    def forward(self, t):
        return self.net(t)

# Sampling time points
def sample_time_points(n_points, t_final):
    tensor = torch.linspace(0, t_final, n_points)
    t = tensor.view(-1,1) # defines the tensor into the shape (row, col). row = -1 implies, automatic selection of number of rows
    t.requires_grad_(True) # Enabling automatic differentiation

# Function to unpack outputs
def unpack(output):
    """
    Splitting output into the form r, v, m
    """
    r = output[: , 0:3] # position vector (r) -> {x, y, z} | Shape: (batch, 3)
    v = output[: , 3:6] # velocity vector (v) -> {vx, vy, vz} | Shape: (batch, 3)
    m = output[: , 6:7] # mass (m)| Shape: (batch, 1)

    return r, v, m

# defining the residual for EOM and other physical contraints
def physics_loss(t, model, r, v, m):
    # constants
    from constants import G, M, Isp, g, T

    # first-order derivative
    dr = torch.autograd.grad(r, t , grad_outputs=torch.ones_like(r),create_graph=True)[0]
    dv = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v),create_graph=True)[0]
    dm = torch.autograd.grad(m, t, grad_outputs=torch.ones_like(m), create_graph=True)[0]

    # residual for r : (refer 1. of EOM loss)
    res_r = dr - v

    # residual for v : (refer 2. of EOM loss)
    norm_r = torch.norm(r, dim=1, keepdim=True)
    a_gravity = - (G*M/norm_r**3) * r # acceleration

    # tensor representing a unit thrust direction vector
    thrust_unit_dir = torch.tensor([0.0, 0.0, 0.1]) # represents thrust pointing in +ve z axis
    thrust_dir = thrust_unit_dir.to(t.device).reshape(1,3) # move to same device (CPU, GPU) : resize for compatibility
    a_thrust = (T / m) * thrust_dir

    res_v = dv - a_gravity - a_thrust


    # residual for mass : (refer 3. of EOM loss)
    res_m = dm + (T/(Isp*g))


    # Mean squared loss
    loss_r = torch.mean(res_r**2)
    loss_v = torch.mean(res_v ** 2)
    loss_m = torch.mean(res_m ** 2)

    return loss_r + loss_v+ loss_m


# initial and final condition loss (refer PINN design)
def initial_loss(model, t0, r0, v0, m0, pred_r, pred_v, pred_m):

    loss_r = torch.mean((pred_r - r0) ** 2)
    loss_v = torch.mean((pred_v - v0) ** 2)
    loss_m = torch.mean((pred_m - m0) ** 2)

    return loss_r + loss_v + loss_m

def terminal_loss(model, tf, r_target, pred_r, pred_v):
    from constants import orbital_velocity
    v_target = orbital_velocity

    loss_r = torch.mean((pred_r - r_target)**2)
    loss_v = torch.mean((pred_v - v_target)**2)

    return loss_r + loss_v

def fuel_efficiency_loss(predicted_mf, initial_mass):
    """
    Encourage the network to use minimal fuel
    predicted_mf: predicted final mass (torch tensor)
    initial_mass: known starting mass (scalar)
    """
    return torch.mean((initial_mass - predicted_mf)**2)



def total_loss(model, t_phys, t0, tf, r0, v0, m0, r_target):
    """
    t_phys : time points for physics loss (interior of domain)
    t0     : initial time (usually 0), shape (1, 1)
    tf     : final time, shape (1, 1)
    r0, v0, m0 : known initial conditions (as tensors)
    r_target   : known final position (ECI coordinate)
    weights    : dict with weights for each loss component
    """

    # weights
    from constants import weights

    # Predict outputs
    out_phys = model(t_phys)
    out_init = model(t0)
    out_final = model(tf)

    r_phys, v_phys, m_phys = unpack(out_phys)
    r_init, v_init, m_init = unpack(out_init)
    r_final, v_final, m_final = unpack(out_final)

    # Compute individual losses
    loss_phys = physics_loss(t_phys, model, r_phys, v_phys, m_phys)
    loss_init = initial_loss(model, t0, r0, v0, m0, r_init, v_init, m_init)
    loss_term = terminal_loss(model, tf, r_target, r_final, v_final)
    loss_fuel = fuel_efficiency_loss(m_final, m0)

    # Weighted total loss
    total = (weights["phys"] * loss_phys +
             weights["init"] * loss_init +
             weights["term"] * loss_term +
             weights["fuel"] * loss_fuel)

    return total

def train(model, epochs, optimizer,t_phys, t0, tf,r0, v0, m0,r_target):

    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()

        loss = total_loss(model, t_phys, t0, tf,
                          r0, v0, m0, r_target)

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

if __name__ == "__main__":
    # Model definition
    model = PINN(layers=[1, 64, 64, 64, 7])  # 1 input (time), 7 outputs
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Time points
    t0 = torch.tensor([[0.0]], requires_grad=True)
    tf = torch.tensor([[burn_time]], requires_grad=True)
    t_phys = torch.linspace(0, burn_time, 100).view(-1, 1)
    t_phys.requires_grad_(True)

    # Initial conditions (from constants)
    r0 = torch.tensor(convert_to_eci(launch_lat, launch_lon), dtype=torch.float32).view(1, 3)
    v0 = torch.zeros_like(r0)
    m0 = torch.tensor([[rocket_mass]], dtype=torch.float32)

    # Target orbit position (using optimal true anomaly)
    nu = compute_optimal_true_anomaly(ORBITAL_ELEMENTS, launch_lat, launch_lon, 0)
    r_target = torch.tensor(rotation_matrix(i, raan, arg_of_perigee) @ position_on_orbit(a, e, nu),
                            dtype=torch.float32).view(1, 3)

    # Train
    train(model, epochs=500000, optimizer=optimizer,
          t_phys=t_phys, t0=t0, tf=tf,
          r0=r0, v0=v0, m0=m0,
          r_target=r_target)

