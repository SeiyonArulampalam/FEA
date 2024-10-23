import numpy as np
import utils as fea_utils
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json

# Specify the precision of the print statement
np.set_printoptions(precision=4)

"""Unsteady 1D heat transfer finite element simulation"""


def cprint(flag, *args, **kwargs):
    """Conditionally print if flag is set to true"""
    if flag:
        print(*args, **kwargs)


def compute_steady_state_exact(L, k, u0, beta, apply_convection):
    """
    Compute the analytical solution to the steady-state heat equation

    Parameters
    ----------
    L : float
       Beam length (m)
    k : float
       Thermal conductivitty
    u0 : float
       Root temperature
    beta : float
        Heat transfer coefficient.
    apply_convection : bool
        Flag for whether forced convection was applied at the tip

    Returns
    -------
    x : array
        Coordinates at which the analytical solution was evaluated
    u_exact : array
        Analytical solution at each x coordinate
    """
    if apply_convection == False:
        # Modify beta to equal zero if there is no convection at tip
        beta = 0

    x = np.linspace(0, L, 100)
    m = 20
    num = np.cosh(m * (L - x)) + (beta / (m * k) * np.sinh(m * (L - x)))
    denom = np.cosh(m * L) + (beta / (m * k)) * np.sinh(m * L)
    u_exact = u0 * np.divide(num, denom)
    return x, u_exact


def plot_unsteady(
    xloc,
    unsteady_soln,
    n_steps,
    dt,
    apply_convection,
    x_exact,
    u_steady_exact,
) -> None:
    """Plot the unstady simulation for the entire beam"""
    # Import the steady state solution
    if apply_convection == True:
        steady_soln = json.load(open("steady_state_soln_convection.json"))
        xloc_steady = json.load(open("xloc_convection.json"))
        fname = "unsteady_convection"
    elif apply_convection == False:
        steady_soln = json.load(open("steady_state_soln.json"))
        xloc_steady = json.load(open("xloc.json"))
        fname = "unsteady"

    fig, ax = plt.subplots(nrows=1, ncols=1)

    colors = plt.cm.Reds(np.linspace(0, 1, n_steps))

    # Plot the transient simulation result
    steps = np.linspace(0, n_steps - 1, n_steps).astype(int)
    for i in steps[0::10]:
        ax.plot(
            xloc,
            unsteady_soln[i, :],
            "-",
            color=colors[i],
            # label=f"step = {i}",
        )

    # Plot the steady state simulation result
    ax.plot(xloc_steady, steady_soln, "ko", label="Steady-State")

    # Plot the steady-state analytical result
    ax.plot(x_exact, u_steady_exact, "g:")
    # Draw a red line for the legend
    green_line = Line2D([0], [0], color="g", linestyle=":")
    black_dot = Line2D([0], [0], marker="o", color="black", linestyle="None")
    red_line = Line2D([0], [0], color="red")

    # Draw the legend
    ax.legend(
        [green_line, black_dot, red_line],
        ["Analytical Steady-State", "Steady-State FEA", "Unsteady FEA"],
    )

    # Define the title of the simulation
    ax.set_title(f"Unsteady FEA heat transfer: dt={dt:.2e}s, T={dt*n_steps:.1e}s")

    # Define the x and y axis
    ax.set_xlabel("Location (m)")
    ax.set_ylabel("Temperature (C)")
    plt.grid()

    # Save the figure
    dir = "/Users/seiyonarulampalam/git/FEA/1D_Themal/Figures"
    plt.savefig(dir + "/" + fname + ".jpg", dpi=800)


def plot_tip(unsteady_soln, n_steps, dt, apply_convection):
    """Plot the temperature at the tip of the beam"""

    # Import the steady state solution
    if apply_convection == True:
        steady_soln = json.load(open("steady_state_soln_convection.json"))
        xloc_steady = json.load(open("xloc_convection.json"))
        fname = "unsteady_tip_convection"
    elif apply_convection == False:
        steady_soln = json.load(open("steady_state_soln.json"))
        xloc_steady = json.load(open("xloc.json"))
        fname = "unsteady_tip"

    # Compute the steps array
    steps = np.linspace(0, n_steps - 1, n_steps).astype(int)

    # Plot the data
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(dt * steps, unsteady_soln[:, -1], "r-", label="Transient")
    ax.plot(
        [0, dt * n_steps], [steady_soln[-1], steady_soln[-1]], "k--", label="Steady"
    )

    # Define the x and y axis, and draw the legend
    ax.set_ylim([0, 200])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tip Temperature (C)")
    ax.legend()

    plt.grid()

    # Save the figure
    dir = "/Users/seiyonarulampalam/git/FEA/1D_Themal/Figures"
    plt.savefig(dir + "/" + fname + ".jpg", dpi=800)


# * Flags
apply_convection = False  # apply forced convection at tip of beam
flag_print = False
print("------------------------------------------------")
print("Apply Conection to Beam Tip = ", apply_convection)
print("------------------------------------------------")

# * Establish the total number of elements and nodes and beam length
num_elems = 20
num_nodes = num_elems + 1
L = 0.05  # length of beam [m]
D = 0.02  # diameter of rod [m]
h_e = L / num_elems  # Length of each element

# * Define the physical coefficients of the simulation
rho = 7700  # material density [kg/m3]
beta = 100.0  # heat transfer coefficient [W/(m2.K)]
k = 50.0  # thermal conductivity [W/(m.K)]
A = np.pi * (D / 2) ** 2  # cross sectional area [m2]
P = 2 * np.pi * (D / 2)  # perimeter [m]
cp = 0.452  # specific heat at constant pressure [J/(kg.K)]

# * Define the root temperature of the beam
T_ambient = 20.0  # ambient temperature [C]
T_root = 320.0  # temperature at the root [C]
u_root = T_root - T_ambient  # temperature solution at the root [C]

# * Define coefficienct found in the heat unsteady 1d heat equation
c1 = rho * cp * A
a = k * A
c0 = P * beta

# * Generate the mesh
xloc = np.linspace(0, L, num_nodes)  # location of x nodes

# * Create an array that maps node tag to node position: | node tag | node location
node_array = np.zeros((num_nodes, 2))
for i in range(num_nodes):
    node_array[i][0] = i
    node_array[i][1] = xloc[i]

cprint(flag_print, "Mapping node tag -> node location:")
cprint(flag_print, node_array)
print()

# * Create element array: | element # | node 1 position | node 2 position |
element_array = np.zeros((num_elems, 3))
for i in range(num_elems):
    element_array[i][0] = i
    element_array[i][1] = node_array[i][1]
    element_array[i][2] = node_array[i + 1][1]

cprint(flag_print, "Mapping element tag -> node position:")
cprint(flag_print, element_array)
print()

# * Create element node tag array: | element # | node 1 tag | node 2 tag |
element_node_tag_array = np.zeros((num_elems, 3))
for i in range(num_elems):
    element_node_tag_array[i][0] = int(i)
    element_node_tag_array[i][1] = int(node_array[i][0])
    element_node_tag_array[i][2] = int(node_array[i + 1][0])

cprint(flag_print, "Mapping element tag -> node tag:")
cprint(flag_print, element_node_tag_array)
print()

# * Assemble K matrix, F vector, and C Matrix
# FEA: K*u + C*u_dot = F

# Excitation of the beam
g = np.zeros(num_elems)  # initialize excitation vector

# Define key function handles for gaussian integration
wts, xi_pts = fea_utils.GaussOrder4()
jac_func = fea_utils.Order1Jacobian
shape_func_derivatives = fea_utils.Order1_dNdx
shape_func_vals = fea_utils.Order1ShapeFunc

K_global, F_global, C_global = fea_utils.assembleFEAmat(
    g=g,
    wts=wts,
    xi_pts=xi_pts,
    jac_func=jac_func,
    shape_func_derivatives=shape_func_derivatives,
    shape_func_vals=shape_func_vals,
    num_elems=num_elems,
    num_nodes=num_nodes,
    element_array=element_array,
    a=a,
    c0=c0,
    c1=c1,
    h_e=h_e,
    element_node_tag_array=element_node_tag_array,
)

# * Simulate time-march
dt = 1e-3
time = 1.0
n_steps = int(time / dt)
u = fea_utils.time_march(
    simulation_time=time,
    n_steps=n_steps,
    n_nodes=num_nodes,
    Kmat=K_global,
    Cmat=C_global,
    Fvec=F_global,
    alpha=0.5,
    u_root=u_root,
    beta=beta,
    A=A,
    apply_convection=apply_convection,
)

# * Compute the exact steady state simulation
x_exact, u_steady_exact = compute_steady_state_exact(
    L, k, u_root, beta, apply_convection
)

# * Plot simulation
dt = time / n_steps
plot_unsteady(xloc, u, n_steps, dt, apply_convection, x_exact, u_steady_exact)
plot_tip(u, n_steps, dt, apply_convection)
plt.show()
