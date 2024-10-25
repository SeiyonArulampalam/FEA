import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import niceplots
import utils
import plot_utils
import triangle_element
import line_element

# Specify the precision of the print statement
np.set_printoptions(precision=4)

"""Run the 2D heat trasnfer simulation"""
# * Flags
apply_convection = True 
open_gmsh = False

# * Define model parameters
a_xx = 40.0  # Thermal conductivitty [W/mC]
a_yy = 40.0  # Thermal conductivitty [W/mC]
if apply_convection == True:
    beta = 75.0  # Heat trasnf. coefficent [W/m^2 C]
    q_hat = 0.0  # Heat flux on convection boundary
else:
    beta = 0.0
    q_hat = 0.0  # Heat flux on convection boundary
T_ambient = 20.0  # Ambient temperature [C]
u_hat = 300.0  # Temperature on dirichlet boundary [C]
c1 = 1.0

# * Generate the mesh
height = 1.0  # Length of analysis domain [m]
length = 1.0  # Height analysis domain [m]
lc1 = 0.6e-1  # Mesh refinement of nodes on base
lc = 0.6e-1  # Mesh refinement of nodes on top edge
mesh_info = utils.generate_mesh(
    lenght=length,
    height=height,
    open_gmsh=open_gmsh,
    lc=lc,
    lc1=lc1,
)

# Extract the variables from the dictionary
print()
print("Mesh Information Keys: ", mesh_info.keys())
nodeTags = mesh_info["nodeTags"]
nodeCoords = mesh_info["nodeCoords"]
elemTags = mesh_info["elemTags"]
elemNodeTags = mesh_info["elemNodeTags"]
nodeTags_s1 = utils.reorder_node_tags(mesh_info["nodeTags_s1"].tolist())
nodeTags_s2 = utils.reorder_node_tags(mesh_info["nodeTags_s2"].tolist())
nodeTags_s3 = utils.reorder_node_tags(mesh_info["nodeTags_s3"].tolist())
nodeTags_s4 = utils.reorder_node_tags(mesh_info["nodeTags_s4"].tolist())

print()
print("Reordered node tags:")
print(f"{nodeTags_s1=}")
print(f"{nodeTags_s2=}")
print(f"{nodeTags_s3=}")
print(f"{nodeTags_s4=}")

n_elems = len(elemTags)
n_nodes = len(nodeTags)
print(f"\nNumber of elements:", n_elems)
print(f"Number of nodes:", n_nodes)

# * Define boundary condition node tags
# Case 1
if apply_convection == False:
    dirichlet_bc_tags = []
    convection_bc_tags = []
    dirichlet_bc_tags.extend(nodeTags_s1)
    dirichlet_bc_tags.extend(nodeTags_s2)
    dirichlet_bc_tags.extend(nodeTags_s3)
    dirichlet_bc_tags.extend(nodeTags_s4)


# Case 2
elif apply_convection == True:
    dirichlet_bc_tags = []
    convection_bc_tags = []
    dirichlet_bc_tags = nodeTags_s1.copy()
    convection_bc_tags.extend(nodeTags_s2[:])
    convection_bc_tags.extend(nodeTags_s3[1:])
    convection_bc_tags.extend(nodeTags_s4[1:])

print("\nConvection BC tags:", convection_bc_tags)
print("Dirichlet BC tags:", dirichlet_bc_tags)

# * Assemble FEA matrices
# Define excitation for each element
if apply_convection == True:
    g = np.zeros(n_elems)
else:
    g = np.zeros(n_elems) + 100.0

# Define handles for numerical integration on triangle elements
wts_triangle, xi_eta_triangle = triangle_element.GaussOrder3()
jac_func_triangle = triangle_element.jac
shape_func_derivatives_triangle = triangle_element.shape_func_derivatives
shape_func_vals_triangle = triangle_element.shape_func_vals

# Define handles for numerical integration on line elements
wts_line, xi_line = line_element.GaussOrder4()
jac_func_line = line_element.jac
shape_func_vals_line = line_element.shape_func_vals

# Assemble K
K = utils.assemble_K(
    wts=wts_triangle,
    xi_eta=xi_eta_triangle,
    jac_func=jac_func_triangle,
    shape_func_derivatives=shape_func_derivatives_triangle,
    shape_func_vals=shape_func_vals_triangle,
    a_xx=a_xx,
    a_yy=a_yy,
    num_elems=n_elems,
    num_nodes=n_nodes,
    elemNodeTags=elemNodeTags,
    nodeCoords=nodeCoords,
)

# Assemble K analytically
K_analytical = utils.assemble_analytical_K(
    a_xx=a_xx,
    a_yy=a_yy,
    num_elems=n_elems,
    num_nodes=n_nodes,
    elemNodeTags=elemNodeTags,
    nodeCoords=nodeCoords,
)

# Check if K and K_analytical are close
if not np.allclose(K, K_analytical):
    raise ValueError("ERROR: K and K_analytical do not match")
else:
    print()
    print("Assembly Check:")
    print("PASSED ANALYTICAL TEST: K")

# Assemble H
H = utils.assemble_H(
    wts=wts_line,
    xi_pts=xi_line,
    jac_func=jac_func_line,
    shape_func_vals=shape_func_vals_line,
    beta=beta,
    num_nodes=n_nodes,
    nodeCoords=nodeCoords,
    convection_bc_tags=convection_bc_tags,
)

# Assemble H analytically
H_analytical = utils.assemble_analytical_H(
    beta=beta,
    num_nodes=n_nodes,
    nodeCoords=nodeCoords,
    convection_bc_tags=convection_bc_tags,
)

# Check if H and H_analytical are close
if not np.allclose(H, H_analytical):
    raise ValueError("ERROR: H and H_analytical do not match")
else:
    print("PASSED ANALYTICAL TEST: H")

# Assemble f
f = utils.assemble_f(
    wts=wts_triangle,
    xi_eta=xi_eta_triangle,
    jac_func=jac_func_triangle,
    shape_func_vals=shape_func_vals_triangle,
    g=g,
    num_elems=n_elems,
    num_nodes=n_nodes,
    elemNodeTags=elemNodeTags,
    nodeCoords=nodeCoords,
)

# Assemble f analytically
f_analytical = utils.assemble_analytical_f(
    g=g,
    num_elems=n_elems,
    num_nodes=n_nodes,
    elemNodeTags=elemNodeTags,
    nodeCoords=nodeCoords,
)

# Check if f and f_analytical are close
if not np.allclose(f, f_analytical):
    raise ValueError("ERROR: f and f_analytical do not match")
else:
    print("PASSED ANALYTICAL TEST: f")

# Assemble P
P = utils.assemble_P(
    wts=wts_line,
    xi_pts=xi_line,
    jac_func=jac_func_line,
    shape_func_vals=shape_func_vals_line,
    beta=beta,
    T_ambient=T_ambient,
    num_nodes=n_nodes,
    nodeCoords=nodeCoords,
    convection_bc_tags=convection_bc_tags,
)

# Assemble P analytically
P_analytical = utils.assemble_analytical_P(
    beta=beta,
    T_ambient=T_ambient,
    num_nodes=n_nodes,
    nodeCoords=nodeCoords,
    convection_bc_tags=convection_bc_tags,
)

# Check if P and P_analytical are close
if not np.allclose(P, P_analytical):
    raise ValueError("ERROR: P and P_analytical do not match")
else:
    print("PASSED ANALYTICAL TEST: P")

# Assemble Q
Q = utils.assemble_Q(
    wts=wts_line,
    xi_pts=xi_line,
    jac_func=jac_func_line,
    shape_func_vals=shape_func_vals_line,
    q_hat=q_hat,
    num_nodes=n_nodes,
    nodeCoords=nodeCoords,
    convection_bc_tags=convection_bc_tags,
)

# Assemble Q analytically
Q_analytical = utils.assemble_analytical_Q(
    q_hat=q_hat,
    num_nodes=n_nodes,
    nodeCoords=nodeCoords,
    convection_bc_tags=convection_bc_tags,
)

# Assemble C
C = utils.assemble_C(
    wts=wts_triangle,
    xi_eta=xi_eta_triangle,
    jac_func=jac_func_triangle,
    shape_func_vals=shape_func_vals_triangle,
    c1=c1,
    num_elems=n_elems,
    num_nodes=n_nodes,
    elemNodeTags=elemNodeTags,
    nodeCoords=nodeCoords,
)

# Check if f and f_analytical are close
if not np.allclose(Q, Q_analytical):
    raise ValueError("ERROR: Q and Q_analytical do not match")
else:
    print("PASSED ANALYTICAL TEST: Q")

# * Steady State Simulation
if apply_convection == True:
    dirichlet_val = u_hat
elif apply_convection == False:
    dirichlet_val = T_ambient
u_steady = utils.steady_state_simulation(
    K.copy(),
    H.copy(),
    f.copy(),
    P.copy(),
    Q.copy(),
    n_nodes,
    dirichlet_bc_tags.copy(),
    dirichlet_val,
)

# * Solve the transient simulation
dt = 0.8e-3
simulation_time = 0.5
n_steps = int(simulation_time / dt)
print(f"\nnum of step = {n_steps}")
dirichlet_bc_tags_tmp = np.array(dirichlet_bc_tags.copy()) - 1
u0 = np.zeros(n_nodes) + T_ambient
if apply_convection == True:
    u0[dirichlet_bc_tags_tmp] = T_ambient
elif apply_convection == False:
    u_hat = T_ambient

alpha = 0.5
u_transient = utils.time_march(
    simulation_time=simulation_time,
    n_steps=n_steps,
    n_nodes=n_nodes,
    alpha=alpha,
    u0=u0,
    C=C.copy(),
    K=K.copy(),
    H=H.copy(),
    f=f.copy(),
    Q=Q.copy(),
    P=P.copy(),
    dirichlet_bc_nodes=dirichlet_bc_tags,
    u_hat=u_hat,
)

# * Compute relative error from steady state and final transient simulation
rel_err = np.divide(u_steady - u_transient[-1, :], u_steady) * 100
print("\nMax Rel. Err % =", max(rel_err))

if max(rel_err) > 1.0:
    raise ValueError("Relative Error of steady state and transient is greater than 1%")

# * Visualize simulations
if apply_convection == True:
    fname = "steady_state_simulation_convection.jpg"
elif apply_convection == False:
    fname = "steady_state_simulation_dirichlet.jpg"
plot_utils.contour_mpl(
    xyz_nodeCoords=nodeCoords.reshape(-1, 3),
    z=u_steady,
    fname=fname,
    flag_save=True,
)

plot_utils.contour_mpl_animate(
    xyz_nodeCoords=nodeCoords.reshape(-1, 3),
    u=u_transient,
    flag_save=False,
    max_rel_err=max(rel_err),
)
