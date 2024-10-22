import numpy as np
import utils as fea_utils
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import niceplots

# Specify the precision of the print statement
np.set_printoptions(precision=4)
plt.style.use(niceplots.get_style())
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10
plt.rcParams["text.color"] = "black"

"""Steady-State Simulation Benchmark Against Textbook"""


def plot(
    xloc,
    u_steady,
    u_unsteady,
    n_steps,
    apply_convection,
) -> None:
    """
    Plot the steady and unsteady FEA solution
    """
    # Compute the total steps vector for the simulation
    steps = np.linspace(0, n_steps - 1, n_steps).astype(int)

    # Define the exact solution from the textbook
    if apply_convection == False:
        colors = plt.cm.coolwarm(steps[0::10])
        line_color = "#7553BF"
        fname = "simulation"

    elif apply_convection == True:
        colors = plt.cm.coolwarm(steps[0::10])
        line_color = "#7553BF"
        fname = "simulation_tip_convection"

    num_elems = len(xloc) - 1  # Compute the total number of elements

    # Plot Results
    fig, ax = plt.subplots(nrows=1, ncols=2)
    for i, step in enumerate(steps[0::10]):
        # Loop through each step and plot the result
        ax[0].plot(
            xloc,
            u_unsteady[step, :],
            "r-",
            label="Unsteady",
            color=colors[i],
            linewidth=2,
        )

    # Create legend object
    transient_line = Line2D(
        [0],
        [0],
        color=line_color,
        linestyle="-",
        linewidth=3,
        label="Transient Tip Temp.",
    )
    steady_state_line = Line2D(
        [0],
        [0],
        color="k",
        linestyle="--",
        linewidth=3,
        label="Steady-State Tip Temp",
    )
    steady_state_data = Line2D(
        [0],
        [0],
        linestyle="--",
        linewidth=3,
        color="c",
        label="Steady-State FEA",
    )

    # Add legend to the second plot
    ax[1].legend(
        handles=[transient_line, steady_state_line, steady_state_data],
    )

    # Create transient history time plot
    ax[0].plot(xloc, u_steady, "c--", label="Steady")
    ax[0].set_xlabel("Location (m)")
    ax[0].set_ylabel("Temp. (C)")
    ax[0].xaxis.label.set_color("black")
    ax[0].grid()

    # Plot the time history of the temperature at the tip
    ax[1].plot(dt * steps, u_unsteady[:, -1], line_color)
    ax[1].plot([0, dt * steps[-1]], [u_steady[-1], u_steady[-1]], "k--")
    ax[1].set_ylim([0, u_steady[0]])
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Tip Temp. (C)")
    ax[1].xaxis.label.set_color("black")
    ax[1].grid()

    # Add a horizontal colorbar below the x-axis of ax[0]
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=plt.cm.coolwarm), ax=ax[0], orientation="horizontal"
    )
    cbar.set_label("Time (s)")

    plt.suptitle(f"Simulation: {num_elems} elements, convection = {apply_convection}")
    dir = "/Users/seiyonarulampalam/git/FEA/1D_Themal/Figures"
    plt.savefig(dir + "/" + fname + ".jpg", dpi=800)


# * Flags
apply_convection = False  # apply forced convection at tip of beam

# * Establish the total number of elements and nodes and beam length
num_elems = 20
num_nodes = num_elems + 1
L = 0.08  # length of beam [m]
D = 0.02  # diameter of rod [m]
h_e = L / num_elems  # Length of each element

# * Define the physical coefficients of the simulation
rho = 7700  # material density [kg/m3]
beta = 100.0  # heat transfer coefficient [W/(m2.K)]
k = 20.0  # thermal conductivity [W/(m.K)]
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

print("Mapping node tag -> node location:")
print(node_array)
print()

# * Create element array: | element # | node 1 position | node 2 position |
element_array = np.zeros((num_elems, 3))
for i in range(num_elems):
    element_array[i][0] = i
    element_array[i][1] = node_array[i][1]
    element_array[i][2] = node_array[i + 1][1]

print("Mapping element tag -> node position:")
print(element_array)
print()

# * Create element node tag array: | element # | node 1 tag | node 2 tag |
element_node_tag_array = np.zeros((num_elems, 3))
for i in range(num_elems):
    element_node_tag_array[i][0] = int(i)
    element_node_tag_array[i][1] = int(node_array[i][0])
    element_node_tag_array[i][2] = int(node_array[i + 1][0])

print("Mapping element tag -> node tag:")
print(element_node_tag_array)
print()


# * Assemble K matrix, F vector, C Matrix
# FEA: K*u + C*u_dot = F

# Excitation of the beam
excitation = 1.0e3
g = np.zeros(num_elems) + excitation  # initialize excitation vector

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

# * Solve steady-state problem
u_steady = fea_utils.solve_steady_state(
    K_global=K_global.copy(),
    F_global=F_global.copy(),
    u_root=u_root,
    beta=beta,
    A=A,
    apply_convection=apply_convection,
)

# * Solve unsteady simulation
dt = 1e-3
time = 1.5
n_steps = int(time / dt)
u_unsteady = fea_utils.time_march(
    simulation_time=time,
    n_steps=n_steps,
    n_nodes=num_nodes,
    Kmat=K_global.copy(),
    Cmat=C_global.copy(),
    Fvec=F_global.copy(),
    alpha=0.5,
    u_root=u_root,
    beta=beta,
    A=A,
    apply_convection=apply_convection,
)


# * Plot result
plot(
    xloc=xloc,
    u_steady=u_steady,
    u_unsteady=u_unsteady,
    apply_convection=apply_convection,
    n_steps=n_steps,
)
