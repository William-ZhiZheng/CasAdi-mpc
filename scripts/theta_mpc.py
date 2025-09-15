# Import necessary libraries
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrow, Circle

# Dynamics and integration
# Dynamics definition

# States
x = ca.MX.sym("x")  # type: ignore
y = ca.MX.sym("y")  # type: ignore
phi = ca.MX.sym("phi")  # type: ignore
theta = ca.MX.sym("theta")  # type: ignore

# Inputs
v_x = ca.MX.sym("v_x")  # type: ignore
omega = ca.MX.sym("omega")  # type: ignore
v = ca.MX.sym("v")  # type: ignore

state = ca.vertcat(x, y, phi, theta)
control = ca.vertcat(v_x, omega, v)

ode = ca.vertcat(
    v_x * ca.cos(phi),  # x_dot
    v_x * ca.sin(phi),  # y_dot
    omega,  # phi_dot
    v,  # virtual_speed
)

# Dynamics defined
f = ca.Function("f", [state, control], [ode], ["x", "u"], ["ode"])

N = 10  # Prediction horizon (number of time steps)
T = 0.05  # Sampling time
# Integrator options
intg_options = {"simplify": True, "number_of_finite_elements": 4}

dae = {
    "x": state,  # What are states?
    "p": control,  # What are parameters (fixed during integration horizon)?
    "ode": f(state, control),  # Expression for the right-hand side
}
# integrator(str name, str solver, Function dae, float t0, [float] tout, dict opts) -> Function
intg = ca.integrator("intg", "rk", dae, 0, N * T, intg_options)
x_next = intg(x0=state, p=control)["xf"]

# Big F, the discretized function
F = ca.Function("F", [state, control], [x_next], ["x", "u"], ["x_next"])

sim = F.mapaccum(N)


# Define spiral parameters
r_start = 5.0  # Starting radius (m)
r_end = 10.0  # Ending radius (m)
n_rotations = 3  # Number of full rotations
theta_end = 2 * np.pi * n_rotations  # Total angle


# Create spiral path functions
def create_spiral_path():
    """Create CasADi functions for spiral path"""

    # Symbolic parameter (normalized 0 to 1)
    s = ca.MX.sym("s")

    # Convert normalized parameter to actual angle
    theta_actual = s * theta_end

    # Radius grows linearly from r_start to r_end
    radius = r_start + (r_end - r_start) * s

    # Spiral coordinates
    x_spiral = radius * ca.cos(theta_actual)
    y_spiral = radius * ca.sin(theta_actual)

    # Create CasADi functions
    x_d = ca.Function("x_d", [s], [x_spiral])
    y_d = ca.Function("y_d", [s], [y_spiral])

    # Compute derivatives for tangent angle
    dx_ds = ca.jacobian(x_spiral, s)
    dy_ds = ca.jacobian(y_spiral, s)

    # Tangent angle function
    phi_path = ca.Function("phi_path", [s], [ca.atan2(dy_ds, dx_ds)])

    return x_d, y_d, phi_path


# Create the path functions
x_d, y_d, phi_path = create_spiral_path()

# Test the functions
print("Testing spiral path functions...")
test_params = np.linspace(0, 1, 11)

print("s\t x_d\t y_d\t phi_path")
print("-" * 40)
for s in test_params:
    try:
        x_val = float(x_d(s))
        y_val = float(y_d(s))
        phi_val = float(phi_path(s))
        print(f"{s:.1f}\t{x_val:.2f}\t{y_val:.2f}\t{phi_val:.3f}")
    except Exception as e:
        print(f"ERROR at s={s}: {e}")

# Visualize the spiral path
s_plot = np.linspace(0, 1, 500)
x_plot = [float(x_d(s)) for s in s_plot]
y_plot = [float(y_d(s)) for s in s_plot]

plt.figure(figsize=(10, 8))
plt.plot(x_plot, y_plot, "b-", linewidth=2, label="Spiral path")
plt.plot(x_plot[0], y_plot[0], "go", markersize=10, label="Start (s=0)")
plt.plot(x_plot[-1], y_plot[-1], "ro", markersize=10, label="End (s=1)")

# Add direction arrows
arrow_indices = np.linspace(0, len(s_plot) - 1, 12, dtype=int)
for i in arrow_indices[:-1]:  # Skip last point
    s_val = s_plot[i]
    phi_val = float(phi_path(s_val))
    dx = 0.5 * np.cos(phi_val)
    dy = 0.5 * np.sin(phi_val)
    plt.arrow(
        x_plot[i],
        y_plot[i],
        dx,
        dy,
        head_width=0.3,
        head_length=0.2,
        fc="orange",
        ec="orange",
        alpha=0.7,
    )

plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Spiral Path: 5m to 10m radius, 3 rotations")
plt.grid(True, alpha=0.3)
plt.legend()
plt.axis("equal")
plt.show()


# Define error functions
def compute_errors(x_robot, y_robot, s_param):
    """Compute contouring and lag errors for spiral path"""
    x_path = x_d(s_param)
    y_path = y_d(s_param)
    tangent_angle = phi_path(s_param)

    # Contouring error (perpendicular to path)
    eps_c = ca.sin(tangent_angle) * (x_robot - x_path) - ca.cos(tangent_angle) * (
        y_robot - y_path
    )

    # Lag error (along path direction)
    eps_l = -ca.cos(tangent_angle) * (x_robot - x_path) - ca.sin(tangent_angle) * (
        y_robot - y_path
    )

    return eps_c, eps_l


# Test error computation
print("\nTesting error computation...")
try:
    eps_c, eps_l = compute_errors(6.0, 1.0, 0.25)  # Test point
    print(f"Test successful: eps_c = {float(eps_c):.4f}, eps_l = {float(eps_l):.4f}")
except Exception as e:
    print(f"Error computation failed: {e}")

# Test derivative continuity
print("\nTesting derivative continuity...")
s_test = np.linspace(0.01, 0.99, 20)
prev_phi = None
large_jumps = []

for s in s_test:
    phi_val = float(phi_path(s))
    if prev_phi is not None:
        jump = abs(phi_val - prev_phi)
        if jump > 3.0:  # Large jump (might indicate discontinuity)
            large_jumps.append((s, jump))
    prev_phi = phi_val

if large_jumps:
    print("WARNING: Large tangent angle jumps detected:")
    for s, jump in large_jumps:
        print(f"  At s={s:.3f}, jump = {jump:.3f} rad")
else:
    print("Tangent angle appears continuous.")

print(f"\nSpiral path successfully created!")
print(f"- Starts at radius {r_start}m, ends at radius {r_end}m")
print(f"- {n_rotations} full rotations")
print(f"- Parameter s ∈ [0, 1] maps to θ ∈ [0, {theta_end:.1f}] rad")

opti = ca.Opti()

x = opti.variable(4, N + 1)  # type: ignore
u = opti.variable(3, N)  # type: ignore
p = opti.parameter(4, 1)  # type: ignore

# Cost function with MPCC errors
q_c = 1  # contouring error weight
q_l = 1000  # lag error weight
q_theta = 10  # progress weight
R = ca.diag([1, 1, 100])  # control weights

cost = 0
for k in range(N):
    # Compute errors at each prediction step
    eps_c, eps_l = compute_errors(x[0, k], x[1, k], x[3, k])
    # MPCC cost
    cost += q_c * eps_c**2 + q_l * eps_l**2 - q_theta * x[3, k]
    cost += ca.mtimes([u[:, k].T, R, u[:, k]])

opti.minimize(cost)

opti.subject_to(x[:, 0] == p)

# Set speed limit
opti.subject_to(u[0, :] <= 1)
opti.subject_to(u[0, :] >= 0)
opti.subject_to(u[1, :] <= 1)
opti.subject_to(u[1, :] >= -1)

# Theta limit
opti.subject_to(x[3, :] <= 1)
opti.subject_to(x[3, :] >= 0)

# Set initial condition
opti.subject_to(x[:, 0] == p)
# opti.solver('worhp')
# opti.solver('sqpmethod')
opti.solver("ipopt")
opti.set_value(p, ca.vertcat(0, 0, 0, 0))


M = opti.to_function("M", [p], [u[:, 0]], ["p"], ["u_opt"])


x_log = []
u_log = []

x = ca.vertcat(0, 0, 0, 0)


for i in range(100):
    u = M(x)
    x_log.append(x)
    u_log.append(u)
    x = F(x, u)
