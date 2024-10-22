import numpy as np
import matplotlib.pyplot as plt
import json

"""
PDE: c1*∂u/∂t - ∂/∂x(a*∂u/∂x) + c0*u = g(x,t)

Insulated BC: k*A*∂u/∂t = 0 

Convection BC: k*A*∂u/∂t + β*A*u = 0

Solution: u = T - T_ambient

c1 = rho*cp*A
a = k*A
c0 = P*β

rho = mass density
β = heat transfer coeff
k = thermal conductivity
A = cross-section area
P = perimeter of rod end (curcumprence of rod)
cp =specific heat at constant pressure
g = internal excitation


The FEM discretization of a beam leads to:

Element (M)      (0)         (1)         (2)
            o-----------o-----------o-----------o-----> (x-axis)
Node (N)    0           1           2           3
"""


def Order1ShapeFunc(xi) -> tuple[np.ndarray, np.ndarray[np.any]]:
    """
    Definition of the shape functions for a linear element

    Parameters
    ----------
    xi : float
        quadrature point in [0,1] frame

    Returns
    -------
    arrays
        f : the value of the shape functions at xi
        df: derivative of shape functions at xi
    """
    # shape functions
    f1 = 1 - xi
    f2 = xi
    f = np.array([f1, f2])

    # shape function derivatives
    df1 = -1
    df2 = 1
    df = np.array([df1, df2])

    return f, df


def Order1Map(xi, x_vec) -> float:
    """
    Map the local reference frame to the global reference fram

    Parameters
    ----------
    xi : float
        quadrature point of interest [0,1]
    x_vec : array
        first entry is the left node position,
        and the second entry is the 2nd node of the element

    Returns
    -------
    float
        the mapped global coordinate of the quadrature point
    """
    x1 = x_vec[0]
    x2 = x_vec[1]

    f, _ = Order1ShapeFunc(xi)
    N1 = f[0]
    N2 = f[1]

    return x1 * N1 + x2 * N2


def Order1Jacobian(x_vec) -> float:
    """
    Compute the jacobian of the element

    Parameters
    ----------
    x_vec : array
        first entry is the left node position,
        and the second entry is the 2nd node of the element

    Returns
    -------
    float
        Jacobian is the scaling factor for a change of frame for integration.
        In our case we want to change our integration fram from the global coordiantes
        to the [0,1] reference frame. The jacobian acts as a scaling term.

    """
    x1 = x_vec[0]
    x2 = x_vec[1]
    return x2 - x1


def Order1_dNdx(xi, x_vec) -> np.ndarray:
    """
    Compute the shape function derivatives w.r.t the global frame

    Parameters
    ----------
    xi : float
        quadrature point of interest [0,1]
    x_vec : array
        first entry is the left node position,
        and the second entry is the 2nd node of the element


    Returns
    -------
    array
        First element is the derivative of the first shape function w.r.t. x.
        Second element is the derivative of the second shape function w.r.t. x.
    """
    jac = Order1Jacobian(x_vec)

    _, df = Order1ShapeFunc(xi)
    dN1_dxi = df[0]
    dN2_dxi = df[1]

    dN1_dx = dN1_dxi / jac
    dN2_dx = dN2_dxi / jac

    dNdx = np.array([dN1_dx, dN2_dx])
    return dNdx


def GaussOrder4():
    """
    Define the Gaussian Quadrature points and weights exact up to order 4. The
    weights are multiplied by 0.5 to shift the weights from wiki that are from the [-1,1] frame.
    Similarly, the quadrature points are shifted as well.

    var_hat -> [0,1] frame. While w and xi are in the [-1,1] frame
    w_hat = 0.5*w
    xi_hat = 0.5*xi + 0.5

    Returns
    -------
    arrays
        wts: quadrature weight in [0,1] frame
        xi : quadrature pts in the [0,1] frame
    """
    # Weights and quad points shifted s.t. xi in [0,1] coordinates
    wts = np.array([0.568889, 0.478629, 0.478629, 0.236927, 0.236927]) * 0.5
    xi = np.array([0.0, 0.538469, -0.538469, 0.90618, -0.90618]) * 0.5 + 0.5
    return wts, xi


def Integrate_K(
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
) -> float:
    """
    Evaluate the integral that defines the elemental stiffness matrix input.

    Parameters
    ----------
    wts : array
        quadrature weights
    xi_pts : array
        quadrature points
    jac_func : function
        jacobian function handle
    shape_func_derivatives : function
        shape function derivatives handle that returns the derivative at quad point
    shape_func_vals : function
        shape function handle that returns the derivative at quad point
    x_vec : array
        first entry is the left node position,
        and the second entry is the 2nd node of the element
    a : float
        elemental coeff
    c0 : float
        elemental coeff
    i : int
        index for the local stiffness matrix (row)
    j : int
        index for the local stiffness matrix (col)

    Returns
    -------
    float
        Gaussian quadrature integration result
    """
    I = 0.0
    for k in range(len(wts)):
        weight = wts[k]
        xi = xi_pts[k]
        jac = jac_func(x_vec)
        dN_dx = shape_func_derivatives(xi, x_vec)
        N, _ = shape_func_vals(xi)

        comp1 = a * dN_dx[i] * dN_dx[j]
        comp2 = c0 * N[i] * N[j]

        I += weight * (comp1 + comp2) * jac

    return I


def Integrate_f(
    wts,
    xi_pts,
    jac_func,
    shape_func_vals,
    x_vec,
    g,
    i,
) -> float:
    """
    Evaluate the elemental f values to build the global F vector.

    Parameters
    ----------
    wts : array
        quadrature weights
    xi_pts : array
        quadrature points
    jac_func : function
        jacobian function handle
    shape_func_vals : function
        shape function handle that returns the derivative at quad point
    x_vec : array
        first entry is the left node position,
        and the second entry is the 2nd node of the element
    g : float
        excitation value for the element
    i : int
        local node number

    Returns
    -------
    float
        Gaussian quadrature integration result
    """
    I = 0.0
    for k in range(len(wts)):
        weight = wts[k]
        xi = xi_pts[k]
        jac = jac_func(x_vec)
        N, _ = shape_func_vals(xi)

        comp1 = N[i] * g

        I += weight * comp1 * jac

    return I


def Integrate_C(
    wts,
    xi_pts,
    jac_func,
    x_vec,
    shape_func_vals,
    c1,
    i,
    j,
) -> float:
    """
    Compute the matrix responsible for the time dependence.

    Parameters
    ----------
    wts : array
        quadrature weights
    xi_pts : array
        quadrature points
    jac_func : function
        jacobian function handle
    shape_func_vals : function
        shape function handle that returns the derivative at quad point
    x_vec : array
        first entry is the left node position,
        and the second entry is the 2nd node of the element
    c1 : float
        physics coeff for the element
    i : int
        local index of matrix (row)
    j : int
        local index of matrix (col)

    Returns
    -------
    float
        Gaussian quadrature integration result
    """
    I = 0.0
    for k in range(len(wts)):
        weight = wts[k]
        xi = xi_pts[k]
        jac = jac_func(x_vec)
        N, _ = shape_func_vals(xi)

        comp1 = c1 * N[i] * N[j]

        I += weight * comp1 * jac
    return I


def time_march(
    simulation_time,
    n_steps,
    n_nodes,
    Kmat,
    Cmat,
    Fvec,
    alpha,
    u_root,
    beta,
    A,
    apply_convection,
):
    """
    Direct numerical integration of the semi-discrete FE system.

    Parameters
    ----------
    simulation_time : float
       Total simulation time
    n_steps : int
       Number of steps to take in the simulation
    n_nodes : int
       Number of nodes in the mesh
    Kmat :  2d array
       Original K matrix for the system
    Cmat : 2d array
       Original C matrix for the system
    Fvec : vector
       Orifinagl F vector for the system
    alpha : float
       Parameter to tune the solver
       0 = Forward difference,
       0.5 = Crank-Nicolson (Stable),
       1.0 = Backward difference
    u_root : float
       Temperature of root above the reference temp
    beta : float
       Heat trasnfer coefficient
    A : float
       Cross sectional area
    apply_convection : bool
       Flag to enable convection BC modification to system

    Returns
    -------
    u : 2d array
       Time history solution to the system.
       Each row corresopnds to the solution to a time-step.
    """
    # Compute Δt (dt)
    dt = simulation_time / n_steps
    print(f"{dt=:.2e}")

    # Initialize solution shape (row = step, col = node temperature)
    u = np.zeros((n_steps, n_nodes))

    # Enforce the root temeprature value at t=0
    u[:, 0] = u_root

    # Compute the time-stepping coefficiencts
    a1 = alpha * dt
    a2 = (1.0 - alpha) * dt

    # Compute the modified stifness matrices
    K_hat_base = Cmat + a2 * Kmat
    K_bar_base = Cmat - a1 * Kmat

    # Step through each time-step
    for i in range(1, n_steps):
        # Extract the previous solution
        u_prev = u[i - 1, :]

        # Initialize copy
        K_hat = K_hat_base.copy()
        K_bar = K_bar_base.copy()

        # Apply Convection
        if apply_convection == True:
            K_hat[-1, -1] += a1 * beta * A
            K_bar[-1, -1] -= a2 * beta * A

        # Compute the RHS
        RHS = K_bar @ u_prev

        # Dirichlet BC
        K_hat[0, 0] = 1.0
        K_hat[0, 1:] = 0.0  # zero the 1st row of the K matrix

        # Modify the RHS vector
        RHS[0] = u_root
        RHS[1:] = RHS[1:] - K_hat[1:, 0] * u_root

        # Zero the 1st col of the K matrix
        K_hat[1:, 0] = 0.0

        # Update solution
        u_new = np.linalg.solve(K_hat, RHS)
        u[i, :] = u_new
    return u


def assembleFEAmat(
    g,
    wts,
    xi_pts,
    jac_func,
    shape_func_derivatives,
    shape_func_vals,
    num_elems,
    num_nodes,
    element_array,
    a,
    c0,
    c1,
    h_e,
    element_node_tag_array,
):
    # Initialize the vectors and matrixes for the finite element analysis
    K_global = np.zeros((num_nodes, num_nodes))
    F_global = np.zeros(num_nodes)
    C_global = np.zeros((num_nodes, num_nodes))

    for e in range(num_elems):
        # loop through each element and update the global matrix
        x_left = element_array[e][1]
        x_right = element_array[e][2]
        x_vec = np.array([x_left, x_right])

        K_local = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                K_local[i, j] = Integrate_K(
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
        # Compute analytical K
        analyical_K = np.array(
            [[1, -1], [-1, 1]],
        ) * (a / h_e) + np.array(
            [[2, 1], [1, 2]]
        ) * (c0 * h_e / 6.0)

        f_local = np.zeros(2)
        for i in range(2):
            f_local[i] = Integrate_f(
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
                C_local[i, j] = Integrate_C(
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
        # Compute analytical C
        analytical_C = np.array([[2, 1], [1, 2]]) * (c1 * h_e / 6.0)

        # Check intergated matrices equals the expected analytical result
        if np.allclose(analytical_C, C_local) != True:
            raise Exception("Matrix C incorrectly computed")

        if np.allclose(analyical_K, K_local) != True:
            raise Exception("Matrix K incorrectly computed")

    # print("K matrix Unmodified:")
    # print(K_global)
    # print()
    # print("F vector Unmodified:")
    # print(F_global)
    # print()
    # print("C matrix Unmodified:")
    # print(C_global)
    # print()

    return K_global, F_global, C_global
