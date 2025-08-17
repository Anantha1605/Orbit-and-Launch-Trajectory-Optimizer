import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import copy
from constants import R, earth_rotation_rate, burn_time, launch_lat, launch_lon, rocket_mass, \
    ORBITAL_ELEMENTS, a, i, raan, arg_of_perigee, e

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

    return np.array([x, y, z], dtype=np.float64)

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

    for nu in np.linspace(0, 2 * np.pi, 50):  # change num: for fine tuning
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
        dropout_prob = 0.05  # Example: 10% dropout rate

        # Building the layers
        for i in range(len(layers) - 1):
            linear_layer = nn.Linear(layers[i], layers[i + 1])
            self.net.add_module(f"linear{i}", linear_layer)
            # Apply Xavier Uniform Initialization to the weights

            if i == len(layers)-2: # output layer
                init.xavier_uniform_(linear_layer.weight, gain = 0.01)
                init.zeros_(linear_layer.bias)
            else:
                init.xavier_uniform_(linear_layer.weight, gain=1.0)
                init.zeros_(linear_layer.bias)

            if i < len(layers) - 2:
                self.net.add_module(f"tanh{i}", nn.Tanh())
                # add dropout layers
                self.net.add_module(f"dropout{i}", nn.Dropout(p=dropout_prob))

        # Output scaling factors (rough estimates)
        self.scale_r = 1e7  # meters
        self.scale_v = 7e3  # m/s
        self.scale_m = 1e6  # kg

    def forward(self, t):
        # network predicts normalized outputs in [-1, 1]
        x = self.net(t) # t is normalized [0,1]
        # r = x[:, 0:3]  -> normalized position
        # v = x[:, 3:6]  -> normalized velocity
        # m = x[:, 6:7]  -> normalized mass

        return x


# Sampling time points
def sample_time_points(n_points):
    t = torch.linspace(0, 1, n_points)
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
    thrust_unit_dir = torch.tensor([0.0, 0.0, 0.1], device=t_norm.device)  # represents thrust pointing in +ve z axis
    thrust_dir = thrust_unit_dir.unsqueeze(0).expand(r.shape[0], -1)  # move to same device (CPU, GPU) : resize for compatibility
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

    return loss_r + loss_v + loss_m + beta * pos_dm_penalty


# initial and final condition loss (refer PINN design)
def initial_loss(model, t0, r0, v0, m0, pred_r, pred_v, pred_m):
    r_pred = pred_r * model.scale_r
    v_pred = pred_v * model.scale_v
    m_pred = pred_m * model.scale_m

    loss_r = torch.mean((r_pred - r0) ** 2)
    loss_v = torch.mean((v_pred - v0) ** 2)
    loss_m = torch.mean((m_pred - m0) ** 2)

    return loss_r + loss_v + loss_m

def terminal_loss(model, tf, r_target, pred_r, pred_v):
    from constants import orbital_velocity

    # mandating orbital_velocity is tensor with correct shape
    if isinstance(orbital_velocity, (int,float)):
        v_target_magnitude = orbital_velocity
        # For circular orbit, velocity is tangential to position
        r_pred_physical = pred_r * model.scale_r
        r_norm = torch.norm(r_pred_physical, dim=1, keepdim=True)
        # Simplified: assume tangential velocity
        v_target = torch.zeros_like(pred_v * model.scale_v)

    else:
        # if already vector
        v_target = torch.tensor(orbital_velocity, dtype=pred_v.dtype, device=pred_v.device)
        if v_target.dim() == 1:
            v_target = v_target.unsqueeze(0)


    r_pred = pred_r * model.scale_r
    v_pred = pred_v * model.scale_v

    loss_r = torch.mean((r_pred - r_target) ** 2)
    loss_v = torch.mean((v_pred - v_target) ** 2)

    return loss_r + loss_v


def fuel_efficiency_loss(predicted_mf_phys, initial_mass, model):
    """
    Encourage the network to use minimal fuel
    predicted_mf: predicted final mass (torch tensor)
    initial_mass: known starting mass (scalar)
    """
    alpha = 1e3
    predicted_mf = predicted_mf_phys * model.scale_m
    if not isinstance(initial_mass, torch.Tensor):
        initial_mass_tensor = torch.tensor(initial_mass, dtype=predicted_mf.dtype, device=predicted_mf.device)
    else:
        initial_mass_tensor = initial_mass.detach().clone().to(dtype=predicted_mf.dtype, device=predicted_mf.device)

    # Reshape to match predicted_mf
    if initial_mass_tensor.dim() == 0:
        initial_mass_tensor = initial_mass_tensor.unsqueeze(0).unsqueeze(0)
    elif initial_mass_tensor.dim() == 1:
        initial_mass_tensor = initial_mass_tensor.unsqueeze(0)

    initial_mass_tensor = initial_mass_tensor.expand_as(predicted_mf)

    # Minimize fuel usage (maximize final mass)
    # symmetric MSE (keep if you want)
    base = torch.mean((initial_mass_tensor - predicted_mf) ** 2)

    # add asymmetric penalty: penalize predicted_mf > initial_mass heavily
    extra = torch.mean(torch.relu(predicted_mf - initial_mass_tensor) ** 2)

    return base + alpha * extra


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
    loss_phys = physics_loss(t_phys, r_phys, v_phys, m_phys, model.scale_r, model.scale_v, model.scale_m)
    loss_init = initial_loss(model, t0, r0, v0, m0, r_init, v_init, m_init)
    loss_term = terminal_loss(model, tf, r_target, r_final, v_final)
    loss_fuel = fuel_efficiency_loss(m_final, m0, model)

    # Weighted total loss
    total = (weights["phys"] * loss_phys +
             weights["init"] * loss_init +
             weights["term"] * loss_term +
             weights["fuel"] * loss_fuel)

    return total, {
        "physics": loss_phys.item(),
        "initial": loss_init.item(),
        "terminal": loss_term.item(),
        "fuel": loss_fuel.item()
    }


def train(model, epochs, optimizer, scheduler, t_phys, t0, tf, r0, v0, m0, r_target):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from constants import loss_threshold

    gradient_history = {}
    loss_history = []

    model.train()

    relocation = Relocation(patience=10, delta_improvement=3e-3)

    for epoch in range(epochs):
        optimizer.zero_grad()

        loss, loss_breakdown = total_loss(model, t_phys, t0, tf, r0, v0, m0, r_target)

        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
        optimizer.step()

        # Stepping the scheduler so it adapts the LR if loss plateaus
        #scheduler.step(loss.item())

        # Store loss history
        loss_history.append(loss.item())

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
            # print(f"  Physics: {loss_breakdown['physics']:.6f}, Initial: {loss_breakdown['initial']:.6f}")
            # print(f"  Terminal: {loss_breakdown['terminal']:.6f}, Fuel: {loss_breakdown['fuel']:.6f}")

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
        if loss.item() <= loss_threshold:
            print(f"\n Loss target reached: {loss.item():.6f} at epoch {epoch}")
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
    def __init__(self, patience=5, delta_improvement=3e-3):
        self.patience = patience
        self.delta_improvement = delta_improvement  # relative factor
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
            return

        if score == "inf":
            self.counter += 1
            if self.counter >= self.patience:
                model.load_state_dict(self.best_model_state)
                self.counter = 0
                self.change_done = True

        # Compute delta dynamically relative to current best
        delta = self.delta_improvement * abs(self.best_score)

        # Improvement
        if score > self.best_score + delta:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            self.change_done = False  # reset flag
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                model.load_state_dict(self.best_model_state)
                self.counter = 0
                self.change_done = True


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model definition
    model = PINN(layers=[1, 128, 256, 128, 7]).to(device)  # 1 input (time), 7 outputs
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
        lr = 1e-4,
        betas = (0.95, 0.999),
        eps = 1e-6,
        weight_decay = 1e-3,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=30,
        threshold_mode='abs',
        threshold=1e8,
        min_lr=1e-6
    )

    # Time points
    t0 = torch.tensor([[0.0]], requires_grad=True, device=device)
    tf = torch.tensor([[1.0]], requires_grad=True, device=device)
    t_phys = sample_time_points(200).to(device)

    # Initial conditions
    r0_np = convert_to_eci(launch_lat, launch_lon, 0)  # Launch position
    r0 = torch.tensor(r0_np, dtype=torch.float32).view(1, 3).to(device)
    v0 = torch.zeros_like(r0)  # Start at rest
    m0 = torch.tensor([[rocket_mass]], dtype=torch.float32).to(device)

    # Target orbit position (using optimal true anomaly)
    nu = compute_optimal_true_anomaly(ORBITAL_ELEMENTS, launch_lat, launch_lon, 0)
    r_target_np = rotation_matrix(i, raan, arg_of_perigee) @ position_on_orbit(a, e, nu)
    r_target = torch.tensor(r_target_np, dtype=torch.float32).view(1, 3).to(device)

    print("Initial conditions:")
    print(f"  Launch position (ECI): {r0_np}")
    print(f"  Target position (ECI): {r_target_np}")
    print(f"  Initial mass: {rocket_mass} kg")
    print(f"  Burn time: {burn_time} seconds")

    # Training the model
    try:
        train(model, epochs=100000, optimizer=optimizer, scheduler=scheduler,
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

            print("Model Prediction!")

            print(f"Final position: {test_r_phys[-1].cpu().numpy()}")
            print(f"Target position: {r_target.cpu().numpy()}")
            print(f"Final mass: {test_m_phys[-1].item():.1f} kg")
            print(f"Mass used: {rocket_mass - test_m_phys[-1].item():.1f} kg")
            f=open('prediction.txt', 'w')
            f.writelines([
                f"Final position: {test_r_phys[-1].cpu().numpy()}",
                f"\nTarget position: {r_target.cpu().numpy()}",
                f"\nFinal mass: {test_m_phys[-1].item():.1f} kg",
                f"\nMass used: {rocket_mass - test_m_phys[-1].item():.1f} kg"
            ])
            f.close()

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback

        traceback.print_exc()


