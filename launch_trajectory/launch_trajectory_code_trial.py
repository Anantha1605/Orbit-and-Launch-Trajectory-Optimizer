import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import copy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from constants import R, earth_rotation_rate, burn_time, launch_lat, launch_lon, fuel_mass, \
    ORBITAL_ELEMENTS, a, i, raan, arg_of_perigee, e, orbital_velocity, Isp, g, v_ground, \
    v0_eci_tensor, r0_eci_tensor, weights, individual_loss_threshold, phys_scale, init_scale, fuel_scale, term_scale

earth_radius = R
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

    return np.array([x, y, z], dtype=np.float32)

def rotation_matrix(i, raan, arg_of_perigee):
    # 3 - 1 - 3 Euler rotation matrix -> sequence of rotations about the z-axis, then the x-axis, then again the z-axis.

    R3_W = np.array(
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
            [0, np.sin(-i), np.cos(-i)]
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

    r = a * (1 - np.power(e, 2)) / (1 + e * np.cos(true_anomaly))

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

    for nu in np.linspace(0, 2 * np.pi, 50):  # change num: for fine-tuning
        r_perifocal = position_on_orbit(a, e, nu)
        r_eci = Q_p_to_i @ r_perifocal
        r_dir = r_eci / np.linalg.norm(r_eci)

        angle = np.arccos(np.clip(np.dot(r_dir, launch_dir), -1, 1))
        if angle < best_angle:
            best_angle = angle
            best_nu = nu

    return best_nu

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
        dropout_prob = 0.001  # Example: 0.1% dropout rate

        # Building the layers
        for i in range(len(layers) - 1):
            linear_layer = nn.Linear(layers[i], layers[i + 1])
            self.net.add_module(f"linear{i}", linear_layer)

            # Apply Xavier Normal Initialization to the weights
            gain = 1.0
            init.xavier_normal_(tensor=linear_layer.weight, gain=gain)
            # Apply Orthogonal Initialization to the weights
            # init.orthogonal_(linear_layer.weight)


            # initialize the bias -> initializing with all 0's
            # nn.init.zeros_(linear_layer.bias)

            if i < len(layers) - 2:
                self.net.add_module(f"softsign{i}", nn.Softsign())
                # add dropout layers
                self.net.add_module(f"dropout{i}", nn.Dropout(p=dropout_prob))


        """
        self.scale_r = 7e6  # ~Earth radius + typical orbital altitude
        self.scale_v = 8e3  # Slightly above orbital velocity
        self.scale_m = 75e4  # 500 tons max (adjust to your fuel_mass)
        """

        self.scale_r = 7.28e6   # Earth + 400km + margin
        self.scale_v = 9.5e3    # 1.2x orbital velocity
        self.scale_m = 549000   # Actual rocket mass

    def forward(self, t):
        # network predicts normalized outputs in [-1, 1]
        # r = x[:, 0:3]  -> normalized position
        # v = x[:, 3:6]  -> normalized velocity
        # m = x[:, 6:7]  -> normalized mass

        x = self.net(t) # t is normalized [0,1]

        return x

# Sampling time points
def sample_time_points(n_points):
    t = torch.linspace(0.0, 1.0, n_points)
    t = t.unsqueeze(1)
    t.requires_grad_(True)  # Enabling automatic differentiation
    return t

# Function to unpack outputs
def unpack(output):
    """
    Splitting output into the form r, v, m
    """
    r = output[:, 0:3]  # position vector (r) -> {x, y, z} | Shape: (batch, 3)
    v = output[:, 3:6]  # velocity vector (v) -> {vx, vy, vz} | Shape: (batch, 3)
    m = output[:, 6:7]  # mass (m)| Shape: (batch, 1)

    return r, v, m

# physics-based thrust direction initialization
def gravity_turn_thrust_direction(r, v, t_norm, burn_time):
    """
    Physics-based gravity turn - gradually tilts from vertical to horizontal.
    Ensures output shape is [N, 3] for compatibility with downstream code.
    """
    # Convert normalized time to physical time
    t_phys = t_norm * burn_time

    # Gravity turn parameters
    turn_start = 10.0  # Start turn after 10 seconds
    turn_duration = 60.0  # Complete turn over 60 seconds

    # Calculate turn angle (0 = vertical, π/2 = horizontal)
    turn_progress = torch.clamp((t_phys - turn_start) / turn_duration, 0, 1)
    pitch_angle = turn_progress * (torch.pi / 2)

    # Local coordinate system
    r_norm = torch.norm(r, dim=1, keepdim=True)
    r_unit = r / (r_norm + 1e-8)  # Radial unit vector (up)

    if r.shape[1] == 3:
        # 3D case
        z_axis = torch.tensor([0., 0., 1.], device=r.device).expand_as(r)
        tangent = torch.cross(z_axis, r_unit, dim=1)
        tangent_norm = torch.norm(tangent, dim=1, keepdim=True)
        tangent_unit = tangent / (tangent_norm + 1e-8)
    else:
        # 2D case
        tangent_unit = torch.stack([-r_unit[:, 1], r_unit[:, 0]], dim=1)

        # Pad r_unit and tangent_unit to 3D
        r_unit = torch.cat([r_unit, torch.zeros(r_unit.size(0), 1, device=r.device)], dim=1)
        tangent_unit = torch.cat([tangent_unit, torch.zeros(tangent_unit.size(0), 1, device=r.device)], dim=1)

    # Combine radial and tangential components
    thrust_dir = (torch.cos(pitch_angle).unsqueeze(-1) * r_unit +
                  torch.sin(pitch_angle).unsqueeze(-1) * tangent_unit)

    return thrust_dir  # Always shape [N, 3]

# defining the residual for EOM and other physical constraints
def physics_loss(t_norm, r_phys, v_phys, m_phys, scale_r, scale_v, scale_m):
    # constants
    from constants import G, M, Isp, g, T, burn_time

    # === Convert to physical units ===
    r = r_phys * scale_r  # meters
    v = v_phys * scale_v  # m/s
    m = m_phys * scale_m  # kg



    # first-order derivative wrt to t
    # Note: derivatives of scaled quantities need chain rule correction since time is normalized
    dr = torch.autograd.grad(r, t_norm, grad_outputs=torch.ones_like(r), create_graph=True)[0]
    dv = torch.autograd.grad(v, t_norm, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    dm = torch.autograd.grad(m, t_norm, grad_outputs=torch.ones_like(m), create_graph=True)[0]

    # Apply chain rule: d/dt = (d/dt_norm) * (dt_norm/dt) = (d/dt_norm) / burn_time
    dr = dr / burn_time
    dv = dv / burn_time
    dm = dm / burn_time

    # residual for r : (refer 1. of EOM loss)
    res_r = dr - v

    # residual for v : (refer 2. of EOM loss)
    norm_r = torch.norm(r, dim=1, keepdim=True)
    eps = 1e-6  # Prevent division by zero
    safe_norm_r = torch.clamp(norm_r, min=eps)
    a_gravity = -(G * M / (safe_norm_r ** 3)) * r  # acceleration

    # tensor representing a unit thrust direction vector
    thrust_unit_dir = gravity_turn_thrust_direction(r, v, t_norm, burn_time)
    #thrust_unit_dir = torch.tensor(thrust_unit_dir, device=t_norm.device)  # represents thrust
    thrust_dir = thrust_unit_dir  # move to same device (CPU, GPU) : resize for compatibility
    eps = 1e-6  # small epsilon to prevent divide-by-zero
    safe_m = torch.clamp(m, min=eps)
    a_thrust = (T / safe_m) * thrust_dir

    res_v = dv - a_gravity - a_thrust

    # residual for mass : (refer 3. of EOM loss)
    res_m = dm + T / (Isp * g)

    # dm currently is d(m_phys)/dt_phys (you divide by burn_time already)
    # penalize positive dm values (mass increasing)
    pos_dm_penalty = torch.mean(torch.relu(dm) ** 2)
    # choose small weight beta
    beta = 1e4

    # Mean squared loss
    loss_r = torch.mean(res_r ** 2)
    loss_v = torch.mean(res_v ** 2)
    loss_m = torch.mean(res_m ** 2)

    # Normalize each loss component
    loss_r_norm = loss_r / (scale_r ** 2)
    loss_v_norm = loss_v / (scale_v ** 2)
    loss_m_norm = loss_m / (scale_m ** 2)

    return loss_r_norm + loss_v_norm + loss_m_norm + beta * pos_dm_penalty


# initial and final condition loss (refer PINN design)
def initial_loss(model, t0, r0, v0, m0, pred_r, pred_v, pred_m):
    r_pred = pred_r * model.scale_r
    v_pred = pred_v * model.scale_v
    # m_pred = abs(pred_m) * model.scale_m -> abs() can cause gradient issues

    m_pred = torch.clamp(pred_m, min=0) * model.scale_m  # Use clamp instead

    loss_r = torch.mean((r_pred - r0) ** 2)
    loss_v = torch.mean((v_pred - v0) ** 2)
    loss_m = torch.mean((m_pred - m0) ** 2)

    # Normalize each loss component
    loss_r_norm = loss_r / (model.scale_r ** 2)
    loss_v_norm = loss_v / (model.scale_v ** 2)
    loss_m_norm = loss_m / (model.scale_m ** 2)

    return loss_r_norm + loss_v_norm + loss_m_norm


def terminal_loss(model, tf, r_target, pred_r, pred_v):
    """
    Fixed terminal loss for circular LEO orbits
    Normalized relative to orbital radius to allow stable training
    """
    from constants import G, M, a  # Use semi-major axis for normalization

    # Scale predictions to physical units
    r_pred_physical = pred_r * model.scale_r
    v_pred_physical = pred_v * model.scale_v

    # Calculate target orbital velocity for circular orbit
    r_target_norm = torch.norm(r_target, dim=1, keepdim=True)
    v_orbital_magnitude = torch.sqrt(G * M / r_target_norm)  # Circular orbital velocity

    # Tangential velocity direction (perpendicular to position)
    r_target_unit = r_target / (r_target_norm + 1e-8)

    if r_target.shape[1] == 3:
        tangent_raw = torch.stack([
            -r_target[:, 1],
            r_target[:, 0],
            torch.zeros_like(r_target[:, 2])
        ], dim=1)
        tangent_norm = torch.norm(tangent_raw, dim=1, keepdim=True)
        tangent_unit = tangent_raw / (tangent_norm + 1e-8)
    else:
        tangent_unit = torch.stack([
            -r_target_unit[:, 1],
            r_target_unit[:, 0]
        ], dim=1)

    # Target velocity vector
    v_target = v_orbital_magnitude * tangent_unit

    # ---- Normalized losses ----
    # Use orbital radius (a) for scaling, instead of absolute meters
    loss_r = torch.mean(((r_pred_physical - r_target) / a) ** 2)
    loss_v = torch.mean(((v_pred_physical - v_target) / v_orbital_magnitude) ** 2)

    return loss_r + loss_v

def fuel_efficiency_loss(predicted_mf_phys, initial_mass, model):
    """
    Physics-based fuel consumption loss using rocket equation
    Drop-in replacement that enforces realistic fuel consumption

    predicted_mf_phys: predicted final mass (torch tensor, normalized)
    initial_mass: known starting mass (scalar or tensor)
    model: model with scale_m attribute for denormalization
    """

    # Import constants (assuming constants.py is in same directory)

    # Calculate target velocity (delta-v) from orbital mechanics
    # For circular orbit: v_orbital = sqrt(GM/r)
    # Ground velocity component from Earth rotation: v_ground = ω*R*cos(lat) ≈ 464 m/s at equator
    ground_velocity = v_ground  # m/s, approximate rotational velocity at launch latitude
    target_velocity = orbital_velocity - ground_velocity  # Net delta-v needed

    specific_impulse = Isp  # Use from constants
    g0 = g  # Use from constants

    # Denormalize predicted final mass
    predicted_mf = predicted_mf_phys * model.scale_m

    # Convert initial_mass to tensor and match dimensions (same as original)
    if not isinstance(initial_mass, torch.Tensor):
        initial_mass_tensor = torch.tensor(initial_mass, dtype=predicted_mf.dtype, device=predicted_mf.device)
    else:
        initial_mass_tensor = initial_mass.detach().clone().to(dtype=predicted_mf.dtype, device=predicted_mf.device)

    # Reshape to match predicted_mf (same as original)
    if initial_mass_tensor.dim() == 0:
        initial_mass_tensor = initial_mass_tensor.unsqueeze(0).unsqueeze(0)
    elif initial_mass_tensor.dim() == 1:
        initial_mass_tensor = initial_mass_tensor.unsqueeze(0)

    initial_mass_tensor = initial_mass_tensor.expand_as(predicted_mf)

    # Calculate physics-based target final mass using rocket equation
    # m_final = m_initial / exp(Δv / v_exhaust)
    ve = specific_impulse * g0  # exhaust velocity
    ideal_mass_ratio = np.exp(target_velocity / ve)  # ~7.4 for LEO
    target_final_mass = initial_mass_tensor / ideal_mass_ratio

    # PRIMARY LOSS: Encourage realistic fuel consumption
    # Penalize deviation from physics-based target
    physics_loss = torch.mean(((predicted_mf - target_final_mass)/model.scale_m) ** 2)

    # CONSTRAINT PENALTIES: Enforce physical bounds
    alpha = 1e2  # Keep same penalty weight as original

    # Heavily penalize impossible scenarios
    impossible_penalty = torch.mean(torch.relu(predicted_mf - initial_mass_tensor) ** 2) / (model.scale_m ** 2)  # mf > m0

    # Penalize unrealistic dry mass (< 5% or > 50% of initial mass)
    min_realistic_mass = 0.05 * initial_mass_tensor
    max_realistic_mass = 0.50 * initial_mass_tensor

    too_light_penalty = torch.mean(torch.relu(min_realistic_mass - predicted_mf) ** 2) / (model.scale_m ** 2)
    insufficient_fuel_penalty = torch.mean(torch.relu(predicted_mf - max_realistic_mass) ** 2) / (model.scale_m ** 2)

    # Combine all penalties (same structure as original)
    constraint_penalties = impossible_penalty + too_light_penalty + insufficient_fuel_penalty

    return physics_loss + alpha * constraint_penalties

def total_loss(model, t_phys, t0, tf, r0, v0, m0, r_target, epoch):
    """
    t_phys : time points for physics loss (interior of domain)
    t0     : initial time (usually 0), shape (1, 1)
    tf     : final time, shape (1, 1)
    r0, v0, m0 : known initial conditions (as tensors)
    r_target   : known final position (ECI coordinate)
    weights    : dict with weights for each loss component
    """


    # Predict outputs
    out_phys = model(t_phys)
    out_init = model(t0)
    out_final = model(tf)

    r_phys, v_phys, m_phys = unpack(out_phys)
    r_init, v_init, m_init = unpack(out_init)
    r_final, v_final, m_final = unpack(out_final)

    # Compute individual losses
    loss_phys = physics_loss(t_phys, r_phys, v_phys, m_phys, model.scale_r, model.scale_v, model.scale_m)
    loss_init = initial_loss(model, t0, r0, v0, m0, r_init, v_init, m_init)
    loss_term = terminal_loss(model, tf, r_target, r_final, v_final)
    loss_fuel = fuel_efficiency_loss(m_final, m0, model)

    # Scaled
    norm_phys = loss_phys / phys_scale
    norm_init = loss_init / init_scale
    norm_term = loss_term / term_scale
    norm_fuel = loss_fuel / fuel_scale

    # Weighted total loss
    total = (weights['phys'] * norm_phys +
                  weights['init'] * norm_init +
                  weights['term'] * norm_term +
                  weights['fuel'] * norm_fuel)


    return total, {
        "physics": loss_phys.item(),
        "initial": loss_init.item(),
        "terminal": loss_term.item(),
        "fuel": loss_fuel.item()
    }


def train(model, epochs, optimizer, scheduler, t_phys, t0, tf, r0, v0, m0, r_target):

    gradient_history = {}
    loss_history = []

    model.train()

    relocation = Relocation(patience=300)

    for epoch in range(epochs):
        optimizer.zero_grad()

        loss, loss_breakdown = total_loss(model, t_phys, t0, tf, r0, v0, m0, r_target, epoch)

        loss.backward()

        # Adaptive gradient clipping
        """
        if epoch < 1000:
            max_norm = 0.5
        elif epoch < 5000:
            max_norm = 0.2
        else:
            max_norm = 0.1
        """
        max_norm = min(1.0, 1.0 / (1 + epoch * 0.0001))  # decays smoothly

        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

        # Optional: Scale gradients for boundary conditions
        if epoch > 1000:
            for name, param in model.named_parameters():
                if param.grad is not None and any(layer in name for layer in ['net.4', 'net.5', 'net.6']):
                    param.grad *= 2.0  # Emphasize boundary conditions in later layers

        optimizer.step()

        # Stepping the scheduler, so it adapts the LR if loss plateaus
        scheduler.step(loss.item())

        # Store loss history
        loss_history.append(loss.item())

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.8f}")
            print(f"  Physics: {loss_breakdown['physics']:.6f}, Initial: {loss_breakdown['initial']:.6f}")
            print(f"  Terminal: {loss_breakdown['terminal']:.6f}, Fuel: {loss_breakdown['fuel']:.6f}")
            print("\n")

            relocation(loss, model)
            if relocation.change_done:
                print("\nRelocated\n")

            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if name not in gradient_history:
                        gradient_history[name] = []
                    gradient_history[name].append(grad_norm)

        # Stop if loss below threshold {Early stopping}
        # Early stopping based on individual component losses
        if (loss_breakdown["physics"] <= individual_loss_threshold * phys_scale and
                loss_breakdown["initial"] <= individual_loss_threshold * init_scale and
                loss_breakdown["terminal"] <= individual_loss_threshold * term_scale and
                loss_breakdown["fuel"] <= individual_loss_threshold * fuel_scale):
            print(f"\nEarly stopping: All critical losses below threshold at epoch {epoch}")
            print(
                f"Physics: {loss_breakdown['physics']:.6f}, Initial: {loss_breakdown['initial']:.6f}, Terminal: {loss_breakdown['terminal']:.6f} , Fuel: {loss_breakdown['fuel']:.6f}")
            break

    # plotting results

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Loss history
    ax1.plot(loss_history)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('Training Loss')
    ax1.set_yscale('log')
    ax1.grid(True)

    # Gradient norms
    for name, norms in gradient_history.items():
        ax2.plot(norms, label=name)
    ax2.set_xlabel('Epoch (x100)')
    ax2.set_ylabel('Gradient Norm')
    ax2.set_title('Gradient Norms per Parameter')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


class Relocation:
    def __init__(self, patience=500):
        self.patience = patience
        self.improvement = 1
        self.best_score = None
        self.counter = 0
        self.best_model_state = None
        self.change_done = False

    def __call__(self, val_loss, model):
        score = -val_loss

        # First time — set baseline
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            self.improvement = 0.3 * self.best_score
            return

        # Improvement
        if score > self.best_score + self.improvement:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            self.change_done = False  # reset flag

        else:
            # Relocate
            self.counter += 1
            if self.counter >= self.patience:
                model.load_state_dict(self.best_model_state)
                self.change_done = True
                self.counter=0


def extract_trajectory_points(model, n_points=100):
    """Extract trajectory collocation points from trained PINN"""
    model.eval()

    with torch.no_grad():
        # Create time points for trajectory extraction
        t_trajectory = torch.linspace(0, 1, n_points).unsqueeze(1).to(device)

        # Get network predictions
        output = model(t_trajectory)
        r_norm, v_norm, m_norm = unpack(output)

        # Convert to physical units
        r_phys = r_norm * model.scale_r  # Position in meters
        v_phys = v_norm * model.scale_v  # Velocity in m/s
        m_phys = m_norm * model.scale_m  # Mass in kg

        # Convert time to physical units (seconds)
        t_phys = t_trajectory * burn_time

        return {
            'time': t_phys.cpu().numpy().flatten(),
            'position': r_phys.cpu().numpy(),  # Shape: (n_points, 3)
            'velocity': v_phys.cpu().numpy(),  # Shape: (n_points, 3)
            'mass': m_phys.cpu().numpy().flatten()
        }


def plot_trajectory(trajectory_data):
    """Plot the complete trajectory"""
    fig = plt.figure(figsize=(15, 10))

    # 3D trajectory plot
    ax1 = fig.add_subplot(221, projection='3d')
    pos = trajectory_data['position']
    ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b-', linewidth=2, label='Trajectory')
    ax1.scatter(pos[0, 0], pos[0, 1], pos[0, 2], c='g', s=100, label='Launch')
    ax1.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], c='r', s=100, label='Target')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()

    # Altitude vs time
    ax2 = fig.add_subplot(222)
    altitude = np.linalg.norm(pos, axis=1) - R  # Altitude above Earth surface
    ax2.plot(trajectory_data['time'], altitude / 1000, 'b-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Altitude (km)')
    ax2.set_title('Altitude Profile')
    ax2.grid(True)

    # Velocity magnitude vs time
    ax3 = fig.add_subplot(223)
    vel_mag = np.linalg.norm(trajectory_data['velocity'], axis=1)
    ax3.plot(trajectory_data['time'], vel_mag, 'r-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('Velocity Profile')
    ax3.grid(True)

    # Mass vs time
    ax4 = fig.add_subplot(224)
    ax4.plot(trajectory_data['time'], trajectory_data['mass'], 'g-', linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Mass (kg)')
    ax4.set_title('Mass Depletion')
    ax4.grid(True)

    plt.tight_layout()
    plt.show()


def save_trajectory_data(trajectory_data, filename='trajectory_collocation_points.txt'):
    """Save collocation points to file"""
    with open(filename, 'w') as f:
        f.write("Time(s)\tX(m)\tY(m)\tZ(m)\tVx(m/s)\tVy(m/s)\tVz(m/s)\tMass(kg)\n")

        for i in range(len(trajectory_data['time'])):
            t = trajectory_data['time'][i]
            r = trajectory_data['position'][i]
            v = trajectory_data['velocity'][i]
            m = trajectory_data['mass'][i]

            f.write(f"{t:.3f}\t{r[0]:.3f}\t{r[1]:.3f}\t{r[2]:.3f}\t")
            f.write(f"{v[0]:.3f}\t{v[1]:.3f}\t{v[2]:.3f}\t{m:.3f}\n")

    print(f"Trajectory collocation points saved to {filename}")





if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    # Model definition
    model = PINN(layers=[1, 128, 256, 64, 256, 128, 7]).to(device)  # 1 input (time), 7 outputs
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-3,
        betas=(0.95, 0.999)
    )
    """
    # Trying out the AdamW optimizer with modified hyperparameters, in place of the standard Adam Optimizer
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr = 1e-5,
        betas = (0.97, 0.999),
        eps = 1e-6,
        weight_decay = 1e-3,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=300,
        threshold_mode='abs',
        threshold=1e8,
        min_lr=1e-7
    )

    # Time points
    t0 = torch.tensor([[0.0]], requires_grad=True, device=device)
    tf = torch.tensor([[1.0]], requires_grad=True, device=device)
    t_phys = sample_time_points(1000).to(device)

    # Initial conditions
    r0_np = convert_to_eci(launch_lat, launch_lon, 0)  # Launch position
    r0 = r0_eci_tensor.to(device)
    v0 = v0_eci_tensor.to(device)
    m0 = torch.tensor([[fuel_mass]], dtype=torch.float32).to(device)

    # Target orbit position (using optimal true anomaly)
    nu = compute_optimal_true_anomaly(ORBITAL_ELEMENTS, launch_lat, launch_lon, 0)
    r_target_np = rotation_matrix(i, raan, arg_of_perigee) @ position_on_orbit(a, e, nu)
    r_target = torch.tensor(r_target_np, dtype=torch.float32).view(1, 3).to(device)

    print("Initial conditions:")
    print(f"  Launch position (ECI): {r0_np}")
    print(f"  Target position (ECI): {r_target_np}")
    print(f"  Initial mass: {fuel_mass} kg")
    print(f"  Burn time: {burn_time} seconds\n")

    # Training the model
    try:
        train(model, epochs=250000, optimizer=optimizer, scheduler=scheduler,
              t_phys=t_phys, t0=t0, tf=tf,
              r0=r0, v0=v0, m0=m0,
              r_target=r_target)

        print("\nTraining completed!\n")

        # Test the trained model
        with torch.no_grad():
            model.eval()
            test_t = torch.linspace(0, 1, 50).unsqueeze(1).to(device)
            test_output = model(test_t)
            test_r, test_v, test_m = unpack(test_output)

            # Convert to physical units
            test_r_phys = test_r * model.scale_r
            test_v_phys = test_v * model.scale_v
            test_m_phys = test_m * model.scale_m

            # ===== Model prediction and saving =====

            print("Model Prediction!")

            print(f"Final position: {test_r_phys[-1].cpu().numpy()}")
            print(f"Target position: {r_target.cpu().numpy()}")
            print(f"Final mass: {test_m_phys[-1].item():.1f} kg")
            print(f"Mass used: {fuel_mass - test_m_phys[-1].item():.1f} kg")
            f=open('prediction.txt', 'w')
            f.writelines([
                f"Final position: {test_r_phys[-1].cpu().numpy()}",
                f"\nTarget position: {r_target.cpu().numpy()}",
                f"\nFinal mass: {test_m_phys[-1].item():.1f} kg",
                f"\nMass used: {fuel_mass - test_m_phys[-1].item():.1f} kg"
            ])
            f.close()

            # ===== Trajectory Visualization =====
            print("\n" + "=" * 50)
            print("EXTRACTING TRAJECTORY COLLOCATION POINTS")
            print("=" * 50)

            # Extract detailed trajectory
            trajectory = extract_trajectory_points(model, n_points=300)

            # Display trajectory statistics
            print(f"\nTrajectory Analysis:")
            print(f"  Total collocation points: {len(trajectory['time'])}")
            print(f"  Mission duration: {trajectory['time'][-1]:.1f} seconds")
            print(f"  Initial altitude: {(np.linalg.norm(trajectory['position'][0]) - R) / 1000:.1f} km")
            print(f"  Final altitude: {(np.linalg.norm(trajectory['position'][-1]) - R) / 1000:.1f} km")
            print(f"  Max velocity: {np.max(np.linalg.norm(trajectory['velocity'], axis=1)):.1f} m/s")
            print(f"  Fuel consumed: {trajectory['mass'][0] - trajectory['mass'][-1]:.1f} kg")

            # Visualize the complete trajectory
            print("\nGenerating trajectory plots...")
            plot_trajectory(trajectory)

            # Save detailed trajectory data
            save_trajectory_data(trajectory)

            print("\nTrajectory analysis complete!")
            print("Check the generated plots and 'trajectory_collocation_points.txt' file")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback

        traceback.print_exc()


