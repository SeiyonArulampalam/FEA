import numpy as np
import utils as fea_utils
import json
import matplotlib.pyplot as plt

# Specify the precision of the print statement
np.set_printoptions(precision=4)

"""Steady-State Simulation Benchmark Against Textbook"""


def compute_analytical(L, k, u0, beta, apply_convection):
    """
    Compute the analytical solution to the heat equation

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


def plot_steady_txtbook(
    xloc,
    u_fea,
    x_exact,
    u_exact,
    apply_convection=False,
) -> None:
    """
    Plot the steady state FEA solution

    Parameters
    ----------
    xloc : array
        Node coordinates
    u_fea : array
        Temperature at each node
    """
    # Define the exact solution from the textbook
    if apply_convection == False:
        xloc_nelems_2 = np.array([0.0, 0.025, 0.05])
        xloc_nelems_4 = np.array([0.0, 0.0125, 0.025, 0.0375, 0.05])
        soln_nelems_2 = np.array([300.0, 217.98, 192.83])
        soln_nelems_4 = np.array([300.0, 251.52, 218.92, 200.16, 194.03])
    elif apply_convection == True:
        xloc_nelems_2 = np.array([0.0, 0.025, 0.05])
        xloc_nelems_4 = np.array([0.0, 0.0125, 0.025, 0.0375, 0.05])
        soln_nelems_2 = np.array([300.0, 211.97, 179.24])
        soln_nelems_4 = np.array([300.0, 248.57, 212.85, 190.56, 180.31])

    num_elems = len(xloc) - 1  # Compute the total number of elements

    # Plot Results
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(xloc, u_fea, "k-", label="FEA")
    ax.plot(x_exact, u_exact, "g--", label="Exact")
    ax.plot(xloc_nelems_2, soln_nelems_2, "D", label="2 Elems. Txtbook")
    ax.plot(xloc_nelems_4, soln_nelems_4, "x", label="4 Elems. Txtbook")
    ax.set_xlabel("Location (m)")
    ax.set_ylabel("Temperature (C)")
    ax.set_title(f"Steady-state heat transfer simulation with {num_elems} elements")
    plt.grid()
    plt.legend()
    plt.savefig("steady.jpg", dpi=800)
    plt.show()


# * Flags
# NOTE: Set to False to ensure simulation matches tesxtbook
apply_convection = True  # apply forced convection at tip of beam

# * Establish the total number of elements and nodes and beam length
num_elems = 4
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

# * Apply Dirichlet B.C.
# modify K_global to be a pivot
K_global[0, 0] = 1.0
K_global[0, 1:] = 0.0  # zero the 1st row of the K matrix

# modify the F_global due to dirichlet BC
F_global[0] = u_root
F_global[1:] = F_global[1:] - K_global[1:, 0] * u_root

# Zero the 1st col of the K matrix
K_global[1:, 0] = 0.0

# * Apply convection BC at beam tip
if apply_convection == True:
    K_global[-1, -1] += beta * A

print("K matrix Modified:")
print(K_global)
print()
print("F vector Modified:")
print(F_global)
print()

# * Solve system fo Equations
u_fea = np.linalg.solve(K_global, F_global)
print("Solution Steady State:")
print(u_fea)
print()

# * Compute the exact solution
x_exact, u_exact = compute_analytical(L, k, u_root, beta, apply_convection)

# * Plot result
plot_steady_txtbook(xloc, u_fea, x_exact, u_exact, apply_convection)

# *Write solution to json
save = u_fea.tolist()
with open("steady_state_soln.json", "w") as f:
    json.dump(save, f)

save = xloc.tolist()
with open("xloc_steady.json", "w") as f:
    json.dump(save, f)
