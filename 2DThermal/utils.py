import numpy as np
import gmsh
import json
import sys

# Specify the precision of the print statement
np.set_printoptions(precision=4)

"""
Summary:
--------
Solve the 2D heat trasnfer problem using the finite element method.
Assumption: orthotropic solid medium

PDE: 
---
    c1*∂u/∂t -∂/∂x(a_xx ∂u/∂x) -∂/∂x(a_yy ∂u/∂y) = g(x,y) in Ω

Dirichlet BC: 
-------------
    u = u_hat on boundary

Convection BC: 
--------------
    (a_xx ∂u/∂x + a_yy ∂u/∂y) + β(u-u_inf) = q on boundary 

Key Coefficiencts:
-----------------
    β = heat transfer coeff
    k = thermal conductivity
    g = internal excitation

Model Analysis Geometry:
------------------------
                    Side 3
            --------------------
            |                  |
            |                  |
            |                  |
 .  Side 4  |       Ω          | Side 2
            |                  |
            |                  |
            |                  |
            --------------------
                    Side 1
"""

#############################################################
"""Functions for generating the mesh"""


def get_nodes_on_line(tag):
    """
    Get the nodes on line the line tag
    """
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes(
        dim=1,
        tag=tag,
        includeBoundary=True,
        returnParametricCoord=False,
    )
    return nodeTags, nodeCoords


def reorder_node_tags(nodeTags):
    """Move the 2nd to last element in the array to index position 0"""
    # Compute the index of the 2nd to last element
    index = len(nodeTags) - 2

    # Determine the tag number
    tag = nodeTags[index]

    # Remove tag at position = index
    nodeTags.pop(index)

    # Insert tag at index position 0
    nodeTags.insert(0, tag)

    return nodeTags


def generate_mesh(lenght=1.0, height=1.0, lc=1.0, lc1=1.0, open_gmsh=True):
    """
    Generate the mesh for the finite element analysis

    Parameters
    ----------
    lenght : float, optional
        length of analysis domain in meters, by default 1.0
    width : float, optional
        width of analysis domain in meters, by default 1.0
    """
    gmsh.initialize()  # Start up GMSH api

    gmsh.model.add("geometry_gmsh")

    # Define the nodes that define the boundary
    gmsh.model.geo.addPoint(0.0, 0.0, 0, lc1, 0)
    gmsh.model.geo.addPoint(lenght, 0.0, 0, lc1, 1)
    gmsh.model.geo.addPoint(lenght, height, 0, lc, 2)
    gmsh.model.geo.addPoint(0.0, height, 0, lc, 3)

    # Draw the lines that connect points.
    gmsh.model.geo.addLine(0, 1, 1)
    gmsh.model.geo.addLine(1, 2, 2)
    gmsh.model.geo.addLine(2, 3, 3)
    gmsh.model.geo.addLine(3, 0, 4)

    # Define the loops
    gmsh.model.geo.addCurveLoop([1, 2, 3, 4], 1, reorient=True)

    # Define surfaces
    gmsh.model.geo.addPlaneSurface([1], 1)

    # Required to call synchronize in order to be meshed
    gmsh.model.geo.synchronize()

    # Generate the 2D mesh
    gmsh.model.mesh.generate(2)

    if "-nopopup" not in sys.argv and open_gmsh == True:
        # open Gmsh GUI
        gmsh.fltk.run()

    # Extract mesh information
    nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes(-1, -1)
    elemTags, elemNodeTags = gmsh.model.mesh.getElementsByType(2, tag=-1)
    nodeTags_s1, nodeCoords_s1 = get_nodes_on_line(1)  # Node tags on side 1
    nodeTags_s2, nodeCoords_s2 = get_nodes_on_line(2)  # Node tags on side 2
    nodeTags_s3, nodeCoords_s3 = get_nodes_on_line(3)  # Node tags on side 3
    nodeTags_s4, nodeCoords_s4 = get_nodes_on_line(4)  # Node tags on side 4

    # Store output as a dictionary
    mesh_info = {
        "nodeTags": nodeTags,
        "nodeCoords": nodeCoords,
        "elemTags": elemTags,
        "elemNodeTags": elemNodeTags,
        "nodeTags_s1": nodeTags_s1,
        "nodeCoords_s1": nodeCoords_s1,
        "nodeTags_s2": nodeTags_s2,
        "nodeCoords_s2": nodeCoords_s2,
        "nodeTags_s3": nodeTags_s3,
        "nodeCoords_s3": nodeCoords_s3,
        "nodeTags_s4": nodeTags_s4,
        "nodeCoords_s4": nodeCoords_s4,
    }

    gmsh.finalize()  # Terminate GMSH api

    return mesh_info


##############################################################################
"""Functions for applying Dirichlet and convection boundary conditions."""

##############################################################################
"""Assemble the FEA matrices and vectors."""


def Integrate_K_ij(
    wts,
    xi_eta,
    jac_func,
    shape_func_derivatives,
    x_vec,
    y_vec,
    a_xx,
    a_yy,
    i,
    j,
):
    """Numerical integration of local matrix K."""
    I = 0.0
    for k in range(len(wts)):
        weight = wts[k]  # quadrature weight
        xi = xi_eta[k][0]  # quadrature point
        eta = xi_eta[k][1]  # quadrature point
        jacobian = jac_func(x_vec, y_vec)
        dN_dx, dN_dy = shape_func_derivatives(jacobian, xi, eta)
        # N = shape_func_vals(xi, eta)
        comp1 = a_xx * dN_dx[i] * dN_dx[j]
        comp2 = a_yy * dN_dy[i] * dN_dy[j]
        detJ = np.linalg.det(jacobian)
        if detJ <= 0:
            raise ValueError("Integrate K: detJ <= 0")
        I += weight * (comp1 + comp2) * detJ
    return I


def Integrate_H_ij(
    wts,
    xi_pts,
    shape_func_vals,
    jac_func,
    x_vec,
    y_vec,
    beta,
    i,
    j,
):
    """Numerical integration of local matrix H"""
    I = 0.0
    for k in range(len(wts)):
        weight = wts[k]  # quadrature weight
        xi = xi_pts[k]  # quadrature point
        jac = jac_func(x_vec, y_vec)  # length of line segment
        N = shape_func_vals(xi)
        comp1 = beta * N[i] * N[j]
        I += weight * comp1 * jac
    return I


def Integrate_f_i(
    wts,
    xi_eta,
    jac_func,
    shape_func_vals,
    x_vec,
    y_vec,
    g_e,
    i,
):
    """Numerical integration of local element forcing vector f"""
    I = 0.0
    for k in range(len(wts)):
        weight = wts[k]  # quadrature weight
        xi = xi_eta[k][0]  # quadrature point
        eta = xi_eta[k][1]  # quadrature point
        jacobian = jac_func(x_vec, y_vec)
        N = shape_func_vals(xi, eta)
        comp1 = g_e * N[i]
        detJ = np.linalg.det(jacobian)
        if detJ <= 0:
            raise ValueError("Integrate f : detJ <= 0")
        I += weight * comp1 * detJ
    return I


def Integrate_P_i(
    wts,
    xi_pts,
    jac_func,
    shape_func_vals,
    x_vec,
    y_vec,
    beta,
    T_ambient,
    i,
):
    """Numerical interhation of local vector P"""
    I = 0.0
    for k in range(len(wts)):
        weight = wts[k]  # quadrature weight
        xi = xi_pts[k]  # quadrature point
        jac = jac_func(x_vec, y_vec)
        N = shape_func_vals(xi)
        comp1 = beta * T_ambient * N[i]
        I += weight * comp1 * jac
    return I


def Integrate_Q_i(
    wts,
    xi_pts,
    jac_func,
    shape_func_vals,
    x_vec,
    y_vec,
    q_hat,
    i,
):
    """Numerical integration of local vector Q_i"""
    I = 0.0
    for k in range(len(wts)):
        weight = wts[k]  # quadrature weight
        xi = xi_pts[k]  # quadrature point
        jac = jac_func(x_vec, y_vec)
        N = shape_func_vals(xi)
        comp1 = q_hat * N[i]
        I += weight * comp1 * jac
    return I


def assemble_K(
    wts,
    xi_eta,
    jac_func,
    shape_func_derivatives,
    shape_func_vals,
    a_xx,
    a_yy,
    num_elems,
    num_nodes,
    elemNodeTags,
    nodeCoords,
):
    """Assemble global K matrix."""
    K = np.zeros((num_nodes, num_nodes))

    for e in range(num_elems):
        # Extract each node for element 'e'.
        # Subtract one for zero indexed node
        n1 = int(elemNodeTags[e * 3] - 1)
        n2 = int(elemNodeTags[e * 3 + 1] - 1)
        n3 = int(elemNodeTags[e * 3 + 2] - 1)

        # Determine the x-y coordinates of each node
        n1_x = nodeCoords[n1 * 3]
        n1_y = nodeCoords[n1 * 3 + 1]
        n2_x = nodeCoords[n2 * 3]
        n2_y = nodeCoords[n2 * 3 + 1]
        n3_x = nodeCoords[n3 * 3]
        n3_y = nodeCoords[n3 * 3 + 1]

        # Define the x_vec and y_vec
        x_vec = np.array([n1_x, n2_x, n3_x])
        y_vec = np.array([n1_y, n2_y, n3_y])

        # Initialize K_local
        K_local = np.zeros((3, 3))

        for i in range(3):
            for j in range(3):
                K_local[i, j] = Integrate_K_ij(
                    wts=wts,
                    xi_eta=xi_eta,
                    jac_func=jac_func,
                    shape_func_derivatives=shape_func_derivatives,
                    x_vec=x_vec,
                    y_vec=y_vec,
                    a_xx=a_xx,
                    a_yy=a_yy,
                    i=i,
                    j=j,
                )

                n_row_e = int(elemNodeTags[3 * e + i] - 1)  # Global row index
                n_col_e = int(elemNodeTags[3 * e + j] - 1)  # Global col index
                K[n_row_e, n_col_e] += K_local[i, j]  # Update global K

    return K


def assemble_analytical_K(
    a_xx,
    a_yy,
    num_elems,
    num_nodes,
    elemNodeTags,
    nodeCoords,
):
    """Assemble global K matrix using the anlytical equations."""
    K = np.zeros((num_nodes, num_nodes))
    for e in range(num_elems):
        # Extract each node for element 'e'.
        # Subtract one for zero indexed node
        n1 = int(elemNodeTags[e * 3] - 1)
        n2 = int(elemNodeTags[e * 3 + 1] - 1)
        n3 = int(elemNodeTags[e * 3 + 2] - 1)

        # Determine the x-y coordinates of each node
        n1_x = nodeCoords[n1 * 3]
        n1_y = nodeCoords[n1 * 3 + 1]
        n2_x = nodeCoords[n2 * 3]
        n2_y = nodeCoords[n2 * 3 + 1]
        n3_x = nodeCoords[n3 * 3]
        n3_y = nodeCoords[n3 * 3 + 1]

        # Compute the area of each element
        a_e = 0.5 * (n1_x * (n2_y - n3_y) + n2_x * (n3_y - n1_y) + n3_x * (n1_y - n2_y))
        if a_e <= 0:
            raise ValueError(f"NEGATIVE AREA: Element {e} has area {a_e}")

        # Compute elementatl coefficients
        b1_e = n2_y - n3_y
        b2_e = n3_y - n1_y
        b3_e = n1_y - n2_y

        c1_e = n3_x - n2_x
        c2_e = n1_x - n3_x
        c3_e = n2_x - n1_x

        b_e = [b1_e, b2_e, b3_e]
        c_e = [c1_e, c2_e, c3_e]

        for i in range(3):
            for j in range(3):
                K_local = (a_xx * b_e[i] * b_e[j] + a_yy * c_e[i] * c_e[j]) / (4 * a_e)
                n_row_e = int(elemNodeTags[3 * e + i] - 1)  # Global row index
                n_col_e = int(elemNodeTags[3 * e + j] - 1)  # Global col index
                K[n_row_e, n_col_e] += K_local  # Update global K

    return K


def assemble_H(
    wts,
    xi_pts,
    shape_func_vals,
    jac_func,
    beta,
    num_nodes,
    nodeCoords,
    convection_bc_tags,
):
    """Assemble global H matrix using numerical integration."""

    H = np.zeros((num_nodes, num_nodes))
    n_line_segs = len(convection_bc_tags) - 1

    for segment_i in range(n_line_segs):
        # Extract the left and right end node tags for the line segment
        n1 = int(convection_bc_tags[segment_i]) - 1  # Left node
        n2 = int(convection_bc_tags[segment_i + 1]) - 1  # Right node
        segment_node_tags = [n1, n2]

        # Extract the x and y coordinates for each node
        n1_x = nodeCoords[n1 * 3]
        n1_y = nodeCoords[n1 * 3 + 1]
        n2_x = nodeCoords[n2 * 3]
        n2_y = nodeCoords[n2 * 3 + 1]

        # x and y vectors
        x_vec = [n1_x, n2_x]
        y_vec = [n1_y, n2_y]

        # Compute length of line segment
        # l_e = np.sqrt(np.power(n2_x - n1_x, 2) + np.power(n2_y - n1_y, 2))

        # Initialize K_local
        H_local = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                H_local[i, j] = Integrate_H_ij(
                    wts,
                    xi_pts,
                    shape_func_vals,
                    jac_func,
                    x_vec,
                    y_vec,
                    beta,
                    i,
                    j,
                )
                n_row_e = int(segment_node_tags[i])  # Global row index
                n_col_e = int(segment_node_tags[j])  # Global col index
                H[int(n_row_e), int(n_col_e)] += H_local[i, j]
    return H


def assemble_analytical_H(
    beta,
    num_nodes,
    nodeCoords,
    convection_bc_tags,
):
    """Assemble global H using analytical formualtion."""
    H = np.zeros((num_nodes, num_nodes))
    n_line_segs = len(convection_bc_tags) - 1

    for segment_i in range(n_line_segs):
        # Extract the left and right end node tags for the line segment
        n1 = int(convection_bc_tags[segment_i]) - 1  # Left node
        n2 = int(convection_bc_tags[segment_i + 1]) - 1  # Right node
        segment_node_tags = [n1, n2]

        # Extract the x and y coordinates for each node
        n1_x = nodeCoords[n1 * 3]
        n1_y = nodeCoords[n1 * 3 + 1]
        n2_x = nodeCoords[n2 * 3]
        n2_y = nodeCoords[n2 * 3 + 1]

        # x and y vectors
        x_vec = [n1_x, n2_x]
        y_vec = [n1_y, n2_y]

        # Compute length of line segment
        l_e = np.sqrt(np.power(n2_x - n1_x, 2) + np.power(n2_y - n1_y, 2))

        # Initialize K_local
        H_local = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                if i == j:
                    delta_ij = 1.0
                elif i != j:
                    delta_ij = 0.0
                H_local[i, j] = beta * (l_e / 6.0) * (1 + delta_ij)

                n_row_e = int(segment_node_tags[i])  # Global row index
                n_col_e = int(segment_node_tags[j])  # Global col index
                H[int(n_row_e), int(n_col_e)] += H_local[i, j]
    return H


def assemble_f(
    wts,
    xi_eta,
    jac_func,
    shape_func_vals,
    g,
    num_elems,
    num_nodes,
    elemNodeTags,
    nodeCoords,
):
    """Assemble global f vector"""
    f = np.zeros(num_nodes)
    for e in range(num_elems):
        # Extract each node for element 'e'.
        # Subtract one for zero indexed node
        n1 = int(elemNodeTags[e * 3] - 1)
        n2 = int(elemNodeTags[e * 3 + 1] - 1)
        n3 = int(elemNodeTags[e * 3 + 2] - 1)

        # Determine the x-y coordinates of each node
        n1_x = nodeCoords[n1 * 3]
        n1_y = nodeCoords[n1 * 3 + 1]
        n2_x = nodeCoords[n2 * 3]
        n2_y = nodeCoords[n2 * 3 + 1]
        n3_x = nodeCoords[n3 * 3]
        n3_y = nodeCoords[n3 * 3 + 1]

        # Define the x_vec and y_vec
        x_vec = np.array([n1_x, n2_x, n3_x])
        y_vec = np.array([n1_y, n2_y, n3_y])

        # Initialize f local
        f_local = np.zeros(3)

        for i in range(3):
            f_local[i] = Integrate_f_i(
                wts=wts,
                xi_eta=xi_eta,
                jac_func=jac_func,
                shape_func_vals=shape_func_vals,
                x_vec=x_vec,
                y_vec=y_vec,
                g_e=g[e],
                i=i,
            )
            n_row_e = int(elemNodeTags[3 * e + i] - 1)  # Global row index
            f[n_row_e] += f_local[i]
    return f


def assemble_analytical_f(g, num_elems, num_nodes, elemNodeTags, nodeCoords):
    """Assemble global f vector analytically."""
    f = np.zeros(num_nodes)
    for e in range(num_elems):
        # Extract each node for element 'e'.
        # Subtract one for zero indexed node
        n1 = int(elemNodeTags[e * 3] - 1)
        n2 = int(elemNodeTags[e * 3 + 1] - 1)
        n3 = int(elemNodeTags[e * 3 + 2] - 1)

        # Determine the x-y coordinates of each node
        n1_x = nodeCoords[n1 * 3]
        n1_y = nodeCoords[n1 * 3 + 1]
        n2_x = nodeCoords[n2 * 3]
        n2_y = nodeCoords[n2 * 3 + 1]
        n3_x = nodeCoords[n3 * 3]
        n3_y = nodeCoords[n3 * 3 + 1]

        # Compute the area of each element
        a_e = 0.5 * (n1_x * (n2_y - n3_y) + n2_x * (n3_y - n1_y) + n3_x * (n1_y - n2_y))
        if a_e <= 0:
            raise ValueError(f"NEGATIVE AREA: Element {e} has area {a_e}")

        # Initialize f local
        f_local = np.zeros(3)
        for i in range(3):
            f_local[i] = a_e * g[e] * (1.0 / 3.0)
            n_row_e = int(elemNodeTags[3 * e + i] - 1)  # Global row index
            f[n_row_e] += f_local[i]
    return f


def assemble_P(
    wts,
    xi_pts,
    jac_func,
    shape_func_vals,
    beta,
    T_ambient,
    num_nodes,
    nodeCoords,
    convection_bc_tags,
):
    """Assemble the global P vector."""
    n_line_segs = len(convection_bc_tags) - 1
    P = np.zeros(num_nodes)
    for segment_i in range(n_line_segs):
        # Extract the left and right end node tags for the line segment
        n1 = int(convection_bc_tags[segment_i]) - 1  # Left node
        n2 = int(convection_bc_tags[segment_i + 1]) - 1  # Right node
        segment_node_tags = [n1, n2]

        # Extract the x and y coordinates for each node
        n1_x = nodeCoords[n1 * 3]
        n1_y = nodeCoords[n1 * 3 + 1]
        n2_x = nodeCoords[n2 * 3]
        n2_y = nodeCoords[n2 * 3 + 1]

        # x and y vectors
        x_vec = [n1_x, n2_x]
        y_vec = [n1_y, n2_y]

        P_local = np.zeros(2)
        for i in range(2):
            P_local[i] = Integrate_P_i(
                wts=wts,
                xi_pts=xi_pts,
                jac_func=jac_func,
                shape_func_vals=shape_func_vals,
                x_vec=x_vec,
                y_vec=y_vec,
                beta=beta,
                T_ambient=T_ambient,
                i=i,
            )
            n_row_e = int(segment_node_tags[i])  # Global row index
            P[n_row_e] += P_local[i]
    return P


def assemble_analytical_P(
    beta,
    T_ambient,
    num_nodes,
    nodeCoords,
    convection_bc_tags,
):
    """Assemble global P vector analytically"""
    n_line_segs = len(convection_bc_tags) - 1
    P = np.zeros(num_nodes)
    for segment_i in range(n_line_segs):
        # Extract the left and right end node tags for the line segment
        n1 = int(convection_bc_tags[segment_i]) - 1  # Left node
        n2 = int(convection_bc_tags[segment_i + 1]) - 1  # Right node
        segment_node_tags = [n1, n2]

        # Extract the x and y coordinates for each node
        n1_x = nodeCoords[n1 * 3]
        n1_y = nodeCoords[n1 * 3 + 1]
        n2_x = nodeCoords[n2 * 3]
        n2_y = nodeCoords[n2 * 3 + 1]

        # Compute length of line segment
        l_e = np.sqrt(np.power(n2_x - n1_x, 2) + np.power(n2_y - n1_y, 2))

        P_local = np.zeros(2)
        for i in range(2):
            P_local[i] = beta * T_ambient * l_e * 0.5
            n_row_e = int(segment_node_tags[i])  # Global row index
            P[n_row_e] += P_local[i]
    return P


def assemble_Q(
    wts,
    xi_pts,
    jac_func,
    shape_func_vals,
    q_hat,
    num_nodes,
    nodeCoords,
    convection_bc_tags,
):
    """Assemble global Q matrix using numerical integration"""
    n_line_segs = len(convection_bc_tags) - 1

    Q = np.zeros(num_nodes)
    for segment_i in range(n_line_segs):
        # Extract the left and right end node tags for the line segment
        n1 = int(convection_bc_tags[segment_i]) - 1  # Left node
        n2 = int(convection_bc_tags[segment_i + 1]) - 1  # Right node
        segment_node_tags = [n1, n2]

        # Extract the x and y coordinates for each node
        n1_x = nodeCoords[n1 * 3]
        n1_y = nodeCoords[n1 * 3 + 1]
        n2_x = nodeCoords[n2 * 3]
        n2_y = nodeCoords[n2 * 3 + 1]

        # x and y vectors
        x_vec = [n1_x, n2_x]
        y_vec = [n1_y, n2_y]

        Q_local = np.zeros(2)
        for i in range(2):
            Q_local[i] = Integrate_Q_i(
                wts=wts,
                xi_pts=xi_pts,
                jac_func=jac_func,
                shape_func_vals=shape_func_vals,
                x_vec=x_vec,
                y_vec=y_vec,
                q_hat=q_hat,
                i=i,
            )
            n_row_e = int(segment_node_tags[i])  # Global row index
            Q[n_row_e] += Q_local[i]
    return Q


def assemble_analytical_Q(
    q_hat,
    num_nodes,
    nodeCoords,
    convection_bc_tags,
):
    """Assemble Q matrix analytically."""
    n_line_segs = len(convection_bc_tags) - 1
    Q = np.zeros(num_nodes)
    for segment_i in range(n_line_segs):
        # Extract the left and right end node tags for the line segment
        n1 = int(convection_bc_tags[segment_i]) - 1  # Left node
        n2 = int(convection_bc_tags[segment_i + 1]) - 1  # Right node
        segment_node_tags = [n1, n2]

        # Extract the x and y coordinates for each node
        n1_x = nodeCoords[n1 * 3]
        n1_y = nodeCoords[n1 * 3 + 1]
        n2_x = nodeCoords[n2 * 3]
        n2_y = nodeCoords[n2 * 3 + 1]

        # Compute length of line segment
        l_e = np.sqrt(np.power(n2_x - n1_x, 2) + np.power(n2_y - n1_y, 2))

        Q_local = np.zeros(2)
        for i in range(2):
            Q_local[i] = q_hat * l_e * 0.5
            n_row_e = int(segment_node_tags[i])  # Global row index
            Q[n_row_e] += Q_local[i]
    return Q


def apply_dirichlet_bc(K_global, b_global, numNodes, nodesBC, u_hat):
    """
    Implement the dirichlet BC at the specified nodes.

    Parameters
    ----------
        K_global : array
            Unmodified left hand side matrix
        b_global : array
            Unmodified right hand side vector
        nodesBC : array
            Vector of dirichlect BC node tags
        u_hat : float
            The value at the specified BC node tags

    Returns
    -------
        K_global : array
            modified LHS matrix
        b_gloabl : array
            modified RHS vector
    """
    for i in range(len(nodesBC)):
        # loop through each node BC will be applied to
        nd_i = int(nodesBC[i] - 1)
        # print(f"BC global node : {nd_i+1}, phi : {p_i}") # matches figure count
        for j in range(numNodes):
            # print(j)
            if nd_i == j:
                K_global[j, j] = 1.0
                b_global[j] = u_hat
            else:
                b_global[j] -= K_global[j, nd_i] * u_hat
                K_global[nd_i, j] = 0.0
                K_global[j, nd_i] = 0.0
    return K_global, b_global
