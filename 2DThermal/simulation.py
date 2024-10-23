import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import niceplots
import utils
import triangle_element
import line_element

# Specify the precision of the print statement
np.set_printoptions(precision=4)

"""Run the 2D heat trasnfer simulation"""
# * Flags
apply_convection = False
apply_dirichlet = False
open_gmsh = False

# * Define model parameters
a_xx = 5.0  # Thermal conductivitty [W/mC]
a_yy = 5.0  # Thermal conductivitty [W/mC]
if apply_convection == True:
    beta = 40.0  # Heat trasnf. coefficent [W/m^2 C]
else:
    beta = 0.0
T_ambient = 20.0  # Ambient temperature [C]
u_hat = 300.0  # Temperature on dirichlet boundary [C]
q_hat = 0.0  # Heat flux on convection boundary

# * Generate the mesh
height = 0.08  # Length of analysis domain [m]
length = 0.02  # Hieght of analysis domain [m]
mesh_info = utils.generate_mesh(lenght=length, height=height, open_gmsh=open_gmsh)

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

# * Define boundary condition node tags
dirichlet_bc_tags = nodeTags_s1.copy()

convection_bc_tags = []
convection_bc_tags.extend(nodeTags_s2[:])
convection_bc_tags.extend(nodeTags_s3[1:])
convection_bc_tags.extend(nodeTags_s4[1:])
print()
print("Convection BC tags:", convection_bc_tags)
print("Dirichlet BC tags:", dirichlet_bc_tags)

# * Assemble FEA matrices
# Define excitation for each element
g = np.zeros(n_elems) + 10

# Define handles for numerical integration on triangle elements
wts_triangle, xi_eta_triangle = triangle_element.GaussOrder3()
jac_func_triangle = triangle_element.jac
shape_func_derivatives_triangle = triangle_element.shape_func_derivatives
shape_func_vals_triangle = triangle_element.shape_func_vals

# Define handles for numerical integration on line elements
wts_line, xi_line = line_element.GaussOrder4()
jac_func_line = line_element.jac
# shape_func_derivatives_line = line_element.shape_func_derivatives
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

# Check is K and K_analytical are close
if not np.allclose(K, K_analytical):
    raise ValueError("ERROR: K and K_analytical do not match")

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

# Check is K and K_analytical are close
if not np.allclose(f, f_analytical):
    raise ValueError("ERROR: f and f_analytical do not match")

# Assemble P

# Assemble Q

# * Solve the steady state problem

# * Solve the transient simulation

# * Visualize simulations
