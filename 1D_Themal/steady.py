import numpy as np
import utils as fea_utils


# Specify the precision of the print statement
np.set_printoptions(precision=4)

"""Steady-state finite element simulation"""

# Flags
# NOTE: Set to False to ensure simulation matches tesxtbook
apply_convection = False  # apply forced convection at tip of beam

# Establish the total number of elements and nodes and beam length
num_elems = 4
num_nodes = num_elems + 1
L = 0.05  # length of beam [m]
D = 0.02  # diameter of rod [m]

# Define the physical coefficients of the simulation
rho = 7700  # material density [kg/m3]
beta = 100.0  # heat transfer coefficient [W/(m2.K)]
k = 50.0  # thermal conductivity [W/(m.K)]
A = np.pi * (D / 2) ** 2  # cross sectional area [m2]
P = 2 * np.pi * (D / 2)  # perimeter [m]
cp = 0.452  # specific heat at constant pressure [J/(kg.K)]

# Define the root temperature of the beam
T_ambient = 20.0  # ambient temperature [C]
T_root = 320.0  # temperature at the root [C]
u_root = T_root - T_ambient  # temperature solution at the root [C]

# Define coefficienct found in the heat unsteady 1d heat equation
c1 = rho * cp * A
a = k * A
c0 = P * beta

# Generate the mesh
xloc = np.linspace(0, L, num_nodes)  # location of x nodes

# Create an array that maps node tag to node position: | node tag | node location
node_array = np.zeros((num_nodes, 2))
for i in range(num_nodes):
    node_array[i][0] = i
    node_array[i][1] = xloc[i]

print("Mapping node tag -> node location:")
print(node_array)
print()

# Create element array: | element # | node 1 position | node 2 position |
element_array = np.zeros((num_elems, 3))
for i in range(num_elems):
    element_array[i][0] = i
    element_array[i][1] = node_array[i][1]
    element_array[i][2] = node_array[i + 1][1]

print("Mapping element tag -> node position:")
print(element_array)
print()

# Create element node tag array: | element # | node 1 tag | node 2 tag |
element_node_tag_array = np.zeros((num_elems, 3))
for i in range(num_elems):
    element_node_tag_array[i][0] = int(i)
    element_node_tag_array[i][1] = int(node_array[i][0])
    element_node_tag_array[i][2] = int(node_array[i + 1][0])

print("Mapping element tag -> node tag:")
print(element_node_tag_array)
print()


# Assemble K matrix, F vector, C Matrix
# FEA: K*u + C*u_dot = F

# Excitation of the beam
g = np.zeros(num_elems)  # initialize excitation vector

# Initialize the vectors and matrixes for the finite element analysis
K_global = np.zeros((num_nodes, num_nodes))
F_global = np.zeros(num_nodes)
C_global = np.zeros((num_nodes, num_nodes))

# define key function handles for gaussian integration
wts, xi_pts = fea_utils.GaussOrder4()
jac_func = fea_utils.Order1Jacobian
shape_func_derivatives = fea_utils.Order1_dNdx
shape_func_vals = fea_utils.Order1ShapeFunc

for e in range(num_elems):
    # loop through each element and update the global matrix
    x_left = element_array[e][1]
    x_right = element_array[e][2]
    x_vec = np.array([x_left, x_right])

    K_local = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            K_local[i, j] = fea_utils.Integrate_K(
                wts,
                xi_pts,
                jac_func,
                shape_func_derivatives,
                shape_func_vals,
                x_vec,
                a,
                c0,
                i,
                j,
            )
            n_i_e = element_node_tag_array[e][i + 1]
            n_j_e = element_node_tag_array[e][j + 1]
            K_global[int(n_i_e), int(n_j_e)] += K_local[i, j]

    f_local = np.zeros(2)
    for i in range(2):
        f_local[i] = fea_utils.Integrate_f(
            wts,
            xi_pts,
            jac_func,
            shape_func_vals,
            x_vec,
            g[e],
            i,
        )
        n_i_e = element_node_tag_array[e][i + 1]
        F_global[int(n_i_e)] += f_local[i]

    C_local = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            C_local[i, j] = fea_utils.Integrate_C(
                wts,
                xi_pts,
                jac_func,
                x_vec,
                shape_func_vals,
                c1,
                i,
                j,
            )
            n_i_e = element_node_tag_array[e][i + 1]
            n_j_e = element_node_tag_array[e][j + 1]
            C_global[int(n_i_e), int(n_j_e)] += C_local[i, j]

print("K matrix Unmodified:")
print(K_global)
print()
print("F vector Unmodified:")
print(F_global)
print()
print("C matrix Unmodified:")
print(C_global)
print()

# Apply Dirichlet B.C.
# modify K_global to be a pivot
K_global[0, 0] = 1.0
K_global[0, 1:] = 0.0  # zero the 1st row of the K matrix

# modify the F_global due to dirichlet BC
F_global[0] = u_root
F_global[1:] = F_global[1:] - K_global[1:, 0] * u_root

# Zero the 1st col of the K matrix
K_global[1:, 0] = 0.0

# # Apply convection BC at beam tip
if apply_convection == True:
    K_global[-1, -1] += beta * A

print("K matrix Modified:")
print(K_global)
print()
print("F vector Modified:")
print(F_global)
print()

# Solve system fo Equations
steady_state_soln = np.linalg.solve(K_global, F_global)
print("Solution Steady State:")
print(steady_state_soln)
print()

# Plot result
fea_utils.plot_steady_state_result(xloc, steady_state_soln)
